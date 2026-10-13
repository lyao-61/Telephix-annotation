"""
方案 A：用 cube_samples.json 中的 move 事件做硬约束，
结合视觉模型区分 hold vs rotate_whole。

原理：
- /cube/move 给出精确的旋转事件（R/R'/U/U'）和时间戳
- move 之间的间隙是 hold 或 rotate_whole，用视觉模型区分
- 几乎可以做到 100% 准确

用法:
    python fusion_a.py [--batch-size 4]
    python fusion_a.py --visual-only  # 纯视觉模型对比
"""

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

TEST_DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
TEST_SESSION = "session_20261010_173611_lerobot"
SAMPLE_FPS = 2

MOVE_TO_SUBTASK = {
    "R": "turn the right layer clockwise",
    "R'": "turn the right layer counterclockwise",
    "U": "turn the up layer clockwise",
    "U'": "turn the up layer counterclockwise",
}

HOLD_OR_ROTATE_PROMPT = (
    "This is a frame from a video of a robot holding a Rubik's cube. "
    "Is the robot holding the cube still (starting grip) or rotating the whole cube? "
    "Answer with only: 'hold the cube in the starting grip' or 'rotate the whole cube'."
)


def load_episode_moves(session_name, episode_name):
    """加载一个 episode 的 move 事件。"""
    ep_path = Path(TEST_DATA_ROOT) / session_name / episode_name
    cube_file = ep_path / "cube_samples.json"
    if not cube_file.exists():
        return []

    with open(cube_file) as f:
        data = json.load(f)

    moves = []
    for m in data.get("/cube/move", []):
        moves.append({
            "timestamp_ns": m["timestamp_ns"],
            "value": m["value"],
            "subtask": MOVE_TO_SUBTASK.get(m["value"], f"unknown: {m['value']}"),
        })

    moves.sort(key=lambda x: x["timestamp_ns"])
    return moves


def get_video_duration_ns(session_name, episode_name):
    """从原始 subtasks.json 获取视频时长。"""
    orig = Path(TEST_DATA_ROOT) / session_name / episode_name / "subtasks.json"
    if orig.exists():
        with open(orig) as f:
            return int(json.load(f)["duration_s"] * 1e9)
    return None


def classify_hold_vs_rotate(model, processor, frame_paths: list[str], batch_size: int):
    """用视觉模型区分 hold vs rotate_whole。"""
    results = []
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]

        msgs = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": HOLD_OR_ROTATE_PROMPT}]}]
            for img in images
        ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        for output in outputs:
            pred = processor.decode(output[in_len:], skip_special_tokens=True).strip().lower()
            if "rotate" in pred:
                results.append("rotate the whole cube")
            else:
                results.append("hold the cube in the starting grip")

    return results


def build_fusion_segments(moves, duration_ns, frame_paths, model, processor, batch_size):
    """融合 move 事件 + 视觉预测，生成 subtask segments。"""
    frame_interval_ns = int(1e9 / SAMPLE_FPS)
    n_frames = len(frame_paths)

    if not moves:
        # 没有 move 数据，全靠视觉
        preds = classify_hold_vs_rotate(model, processor, frame_paths, batch_size)
        return [{"t_start_ns": 0, "t_end_ns": duration_ns, "text": preds[0]}]

    segments = []
    prev_move_end_ns = 0

    for i, move in enumerate(moves):
        move_start_ns = move["timestamp_ns"]

        # move 之前的间隙 → hold 或 rotate_whole
        if move_start_ns > prev_move_end_ns + frame_interval_ns:
            # 取间隙中间的一帧做视觉判断
            gap_start_frame = int(prev_move_end_ns / frame_interval_ns)
            gap_end_frame = int(move_start_ns / frame_interval_ns)
            mid_frame = (gap_start_frame + gap_end_frame) // 2
            mid_frame = min(mid_frame, n_frames - 1)

            gap_preds = classify_hold_vs_rotate(
                model, processor, [frame_paths[mid_frame]], batch_size
            )
            gap_label = gap_preds[0]

            segments.append({
                "t_start_ns": prev_move_end_ns,
                "t_end_ns": move_start_ns,
                "text": gap_label,
            })

        # move 本身 → 到下一个 move 或视频结束
        next_boundary = moves[i + 1]["timestamp_ns"] if i + 1 < len(moves) else duration_ns
        segments.append({
            "t_start_ns": move_start_ns,
            "t_end_ns": next_boundary,
            "text": move["subtask"],
        })

        prev_move_end_ns = next_boundary

    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/home/liuyao/Telephix/Telephix-annotation/cube_subtask_finetune/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora-path", default="/home/liuyao/projects/cube_subtask_finetune/checkpoints/lora_final")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--visual-only", action="store_true", help="只用视觉模型（不融合 move 数据）")
    parser.add_argument("--output-dir", default="predictions_fusion_a")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.base_model}")
    processor = Qwen3VLProcessor.from_pretrained(args.base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if not args.visual_only and Path(args.lora_path).exists():
        print(f"加载 LoRA: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    # 遍历测试 session
    session_dir = Path(TEST_DATA_ROOT) / TEST_SESSION
    total_segments = 0
    t0 = time.time()

    for ep_dir in sorted(session_dir.glob("episode_*")):
        ep_name = ep_dir.name
        moves = load_episode_moves(TEST_SESSION, ep_name)
        duration_ns = get_video_duration_ns(TEST_SESSION, ep_name)

        if duration_ns is None:
            continue

        # 提取帧路径
        frame_dir = Path("data/frames") / f"{TEST_SESSION}_{ep_name}"
        frame_paths = sorted(str(f) for f in frame_dir.glob("frame_*.jpg"))

        if args.visual_only:
            # 纯视觉：用微调模型逐帧预测
            preds = classify_hold_vs_rotate(model, processor, frame_paths, args.batch_size)
            # 简单合并
            segments = []
            if preds:
                cur = preds[0]; start = 0
                for i in range(1, len(preds)):
                    if preds[i] != cur:
                        segments.append({"t_start_ns": start * int(1e9/SAMPLE_FPS), "t_end_ns": i * int(1e9/SAMPLE_FPS), "text": cur})
                        cur = preds[i]; start = i
                segments.append({"t_start_ns": start * int(1e9/SAMPLE_FPS), "t_end_ns": duration_ns, "text": cur})
        else:
            segments = build_fusion_segments(moves, duration_ns, frame_paths, model, processor, args.batch_size)

        total_segments += len(segments)

        # 保存
        ep_out = output_dir / TEST_SESSION / ep_name
        ep_out.mkdir(parents=True, exist_ok=True)
        result = {
            "version": 1,
            "duration_s": duration_ns / 1e9,
            "segments": segments,
            "annotator": "fusion_a (move + visual)" if not args.visual_only else "visual_only",
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(ep_out / "subtasks_pred.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  {ep_name}: {len(segments)} segments, {len(moves)} moves")

    elapsed = time.time() - t0
    n_episodes = len(list(session_dir.glob("episode_*")))
    print(f"\n总计: {total_segments} segments / {n_episodes} episodes (avg {total_segments/n_episodes:.1f})")
    print(f"耗时: {elapsed:.1f}s")
    print(f"保存到: {output_dir}/")


if __name__ == "__main__":
    main()