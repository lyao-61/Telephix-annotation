"""
视频级评估：输入完整视频，输出 subtask 时间线，与真值对比。

用法:
    python evaluate_video.py --lora-path checkpoints_video/lora_final
    python evaluate_video.py --base-only
"""

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

TEST_DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
TEST_SESSION = "session_20261010_173611_lerobot"
VIDEO_NAME = "observation.images.global.mp4"
SAMPLE_FPS = 2

SUBTASK_LABELS = [
    "hold the cube in the starting grip",
    "rotate the whole cube",
    "turn the right layer clockwise",
    "turn the right layer counterclockwise",
    "turn the up layer clockwise",
    "turn the up layer counterclockwise",
]

PROMPT = (
    "Watch this video of a robot manipulating a Rubik's cube. "
    "Segment the video into subtasks and output the timeline.\n\n"
    "There are exactly 6 possible subtasks:\n"
    "1. 'hold the cube in the starting grip' - cube is static, hands holding still\n"
    "2. 'rotate the whole cube' - the whole cube is passed between hands "
    "(left to right and back to left); cube orientation changes but no layer is turned\n"
    "3. 'turn the right layer clockwise' - right face rotates 90° clockwise\n"
    "4. 'turn the right layer counterclockwise' - right face rotates 90° counterclockwise\n"
    "5. 'turn the up layer clockwise' - top face rotates 90° clockwise\n"
    "6. 'turn the up layer counterclockwise' - top face rotates 90° counterclockwise\n\n"
    "Focus on: which face rotates 90°? Is there a hand-transfer motion? "
    "Is the cube static?\n\n"
    "Output format (one subtask per line, [start_s-end_s] label):\n"
    "[0.0-5.2] hold the cube in the starting grip\n"
    "[5.2-12.8] rotate the whole cube\n"
    "..."
)


def parse_timeline(text: str) -> list[dict]:
    """解析模型输出的时间线文本。"""
    segments = []
    # 匹配 [start-end] label 格式
    pattern = r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.+)'
    for match in re.finditer(pattern, text):
        t_start = float(match.group(1))
        t_end = float(match.group(2))
        label = match.group(3).strip()
        # 匹配到已知标签
        matched = False
        for known in SUBTASK_LABELS:
            if known.lower() in label.lower() or label.lower() in known.lower():
                label = known
                matched = True
                break
        segments.append({
            "t_start_ns": int(t_start * 1e9),
            "t_end_ns": int(t_end * 1e9),
            "text": label,
        })
    return segments


def compute_frame_accuracy(gt_segments, pred_segments, duration_s, fps=SAMPLE_FPS, tolerance_s=1.0):
    """逐帧对比真值和预测，返回准确率统计。"""
    n_frames = int(duration_s * fps)
    frame_ns = int(1e9 / fps)
    tolerance_ns = int(tolerance_s * 1e9)

    # 收集边界时间点
    gt_boundaries = set()
    for i in range(len(gt_segments) - 1):
        gt_boundaries.add(gt_segments[i]["t_end_ns"])
    pred_boundaries = set()
    for i in range(len(pred_segments) - 1):
        pred_boundaries.add(pred_segments[i]["t_end_ns"])
    all_boundaries = gt_boundaries | pred_boundaries

    def get_label_at(segments, t_ns):
        for seg in segments:
            if seg["t_start_ns"] <= t_ns < seg["t_end_ns"]:
                return seg["text"]
        return None

    def is_boundary_frame(t_ns):
        for b in all_boundaries:
            if abs(t_ns - b) < tolerance_ns:
                return True
        return False

    core_correct = 0; core_total = 0
    bnd_correct = 0; bnd_total = 0
    all_correct = 0

    for fi in range(n_frames):
        t_ns = fi * frame_ns
        gt_label = get_label_at(gt_segments, t_ns)
        pred_label = get_label_at(pred_segments, t_ns)
        if gt_label is None:
            continue

        match = (gt_label == pred_label)
        if match:
            all_correct += 1

        if is_boundary_frame(t_ns):
            bnd_total += 1
            if match:
                bnd_correct += 1
        else:
            core_total += 1
            if match:
                core_correct += 1

    return {
        "core_correct": core_correct,
        "core_total": core_total,
        "boundary_correct": bnd_correct,
        "boundary_total": bnd_total,
        "all_correct": all_correct,
        "all_total": core_total + bnd_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/home/liuyao/Telephix/Telephix-annotation/cube_subtask_finetune/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora-path", default="/home/liuyao/projects/cube_subtask_finetune/checkpoints_video/lora_final")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--output-dir", default="predictions_video")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--fps", type=float, default=1.42, help="视频采样帧率 (default: 1.42, 128帧覆盖~90s)")
    parser.add_argument("--max-frames", type=int, default=128)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"加载模型: {args.base_model}")
    processor = Qwen3VLProcessor.from_pretrained(args.base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if not args.base_only:
        if Path(args.lora_path).exists():
            print(f"加载 LoRA: {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
            model = model.merge_and_unload()
        else:
            print("警告: 未找到 LoRA 权重")

    session_dir = Path(TEST_DATA_ROOT) / TEST_SESSION
    episodes = sorted(session_dir.glob("episode_*"))
    print(f"测试: {len(episodes)} episodes\n")

    total_stats = defaultdict(int)
    total_gen_time = 0

    for ep_dir in episodes:
        ep_name = ep_dir.name
        video_path = ep_dir / "videos" / VIDEO_NAME
        subtasks_path = ep_dir / "subtasks.json"

        if not video_path.exists() or not subtasks_path.exists():
            continue

        with open(subtasks_path) as f:
            gt_data = json.load(f)
        gt_segments = gt_data["segments"]
        duration_s = gt_data["duration_s"]

        # 推理
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": PROMPT},
            ]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[str(video_path)], fps=args.fps,
                          max_frames=args.max_frames,
                          size={"shortest_edge": 4587520, "longest_edge": 18350080},
                          return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen_time = time.time() - t0
        total_gen_time += gen_time

        in_len = inputs["input_ids"].shape[1]
        pred_text = processor.decode(outputs[0][in_len:], skip_special_tokens=True).strip()
        pred_segments = parse_timeline(pred_text)

        # 计算准确率
        stats = compute_frame_accuracy(gt_segments, pred_segments, duration_s, fps=SAMPLE_FPS)
        for k, v in stats.items():
            total_stats[k] += v

        core_acc = stats["core_correct"] / max(stats["core_total"], 1)
        print(f"  {ep_name}: core={core_acc:.1%} ({stats['core_correct']}/{stats['core_total']}), "
              f"gt_segments={len(gt_segments)}, pred_segments={len(pred_segments)}, "
              f"gen_time={gen_time:.1f}s")

        # 保存预测
        ep_out = output_dir / TEST_SESSION / ep_name
        ep_out.mkdir(parents=True, exist_ok=True)
        result = {
            "version": 1,
            "duration_s": duration_s,
            "segments": pred_segments,
            "annotator": "Qwen3-VL-2B-Instruct (video, finetuned)" if not args.base_only else "Qwen3-VL-2B-Instruct (video, base)",
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(ep_out / "subtasks_pred.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # 汇总
    n_eps = len(episodes)
    core_acc = total_stats["core_correct"] / max(total_stats["core_total"], 1)
    bnd_acc = total_stats["boundary_correct"] / max(total_stats["boundary_total"], 1)
    all_acc = total_stats["all_correct"] / max(total_stats["all_total"], 1)

    print(f"\n{'='*60}")
    print(f"Overall Results ({n_eps} episodes)")
    print(f"{'='*60}")
    print(f"Core accuracy:     {core_acc:.1%} ({total_stats['core_correct']}/{total_stats['core_total']})")
    print(f"Boundary accuracy: {bnd_acc:.1%} ({total_stats['boundary_correct']}/{total_stats['boundary_total']})")
    print(f"Overall accuracy:  {all_acc:.1%} ({total_stats['all_correct']}/{total_stats['all_total']})")
    print(f"Avg gen time:      {total_gen_time/max(n_eps,1):.1f}s per episode")
    print(f"Predictions saved to: {output_dir}/")


if __name__ == "__main__":
    main()