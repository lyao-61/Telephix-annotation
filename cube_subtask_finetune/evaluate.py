"""
评估微调后的 Qwen3-VL 在测试集上的 subtask 分类准确率。

边界容忍：segment 交界附近（±tolerance_frames）的帧单独统计，
不计入核心准确率，避免因标注边界模糊而误判模型能力。

用法:
    python evaluate.py [--batch-size 4] [--tolerance-frames 2]
    python evaluate.py --base-only   # baseline
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

TEST_DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
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
    "This is a frame from a video of a robot manipulating a Rubik's cube. "
    "What subtask is the robot performing? "
    "Choose from: hold the cube in the starting grip, rotate the whole cube, "
    "turn the right layer clockwise, turn the right layer counterclockwise, "
    "turn the up layer clockwise, turn the up layer counterclockwise. "
    "Answer with only the subtask label."
)


def match_label(predicted: str) -> str:
    pred = predicted.lower().strip().rstrip(".")
    for label in SUBTASK_LABELS:
        if label.lower() in pred or pred in label.lower():
            return label
    pred_clean = pred.replace(".", "").replace(",", "")
    for label in SUBTASK_LABELS:
        if label.lower() in pred_clean or pred_clean in label.lower():
            return label
    return predicted.strip()


def build_boundary_map(tolerance_frames: int) -> set[str]:
    """构建边界帧集合。返回 sample_id 集合，这些帧在 segment 交界 ±tolerance 内。"""
    boundary_ids = set()
    tolerance_ns = int(tolerance_frames * 1e9 / SAMPLE_FPS)

    for session_dir in sorted(Path(TEST_DATA_ROOT).glob("session_*")):
        for episode_dir in sorted(session_dir.glob("episode_*")):
            subtasks_file = episode_dir / "subtasks.json"
            if not subtasks_file.exists():
                continue

            with open(subtasks_file) as f:
                subtasks = json.load(f)

            segments = subtasks["segments"]
            if len(segments) < 2:
                continue

            # 收集所有边界时间点（相邻 segment 的 t_end / t_start）
            boundaries = []
            for i in range(len(segments) - 1):
                boundaries.append(segments[i]["t_end_ns"])

            episode_id = f"{session_dir.name}_{episode_dir.name}"
            frame_interval_ns = int(1e9 / SAMPLE_FPS)

            for b_ns in boundaries:
                # 找出 ±tolerance 范围内的帧
                start_frame = max(0, int((b_ns - tolerance_ns) / frame_interval_ns))
                end_frame = int((b_ns + tolerance_ns) / frame_interval_ns)
                for fi in range(start_frame, end_frame + 1):
                    boundary_ids.add(f"{episode_id}_frame{fi:04d}")

    return boundary_ids


def run_inference(model, processor, samples: list, batch_size: int, max_pixels: int):
    """跑推理，返回 (predictions, ground_truths, sample_ids, elapsed)。"""
    processor.max_pixels = max_pixels
    predictions = []
    ground_truths = []
    sample_ids = []

    t0 = time.time()
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = [Image.open(s["image"]).convert("RGB") for s in batch]
        gts = [s["conversations"][1]["value"] for s in batch]
        ids = [s["id"] for s in batch]

        msgs = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
            for img in images
        ]
        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        in_len = inputs["input_ids"].shape[1]
        for j, output in enumerate(outputs):
            pred_text = processor.decode(output[in_len:], skip_special_tokens=True).strip()
            predictions.append(match_label(pred_text))
            ground_truths.append(gts[j])
            sample_ids.append(ids[j])

        if (i // batch_size) % 50 == 0:
            print(f"  [{i + len(batch)}/{len(samples)}]")

    elapsed = time.time() - t0
    return predictions, ground_truths, sample_ids, elapsed


def print_report(tag, predictions, ground_truths, sample_ids, boundary_ids, elapsed, total_samples):
    """打印评估报告。"""
    # 分类：核心帧 vs 边界帧
    core_correct = 0; core_total = 0
    boundary_correct = 0; boundary_total = 0
    all_correct = 0

    for gt, pred, sid in zip(ground_truths, predictions, sample_ids):
        match = (gt == pred)
        if match:
            all_correct += 1
        if sid in boundary_ids:
            boundary_total += 1
            if match:
                boundary_correct += 1
        else:
            core_total += 1
            if match:
                core_correct += 1

    print(f"\n{'='*60}")
    print(f"[{tag}] Results")
    print(f"{'='*60}")
    print(f"Total samples:   {total_samples}")
    print(f"Core frames:     {core_total}  (away from boundaries)")
    print(f"Boundary frames: {boundary_total}  (within ±tolerance of transitions)")
    print(f"Inference time:  {elapsed:.1f}s  ({elapsed/total_samples*1000:.0f}ms/sample)")

    print(f"\n--- Core Accuracy (excl. boundary) ---")
    if core_total > 0:
        print(f"Accuracy: {core_correct}/{core_total} = {core_correct/core_total:.1%}")
    else:
        print(f"No core frames")

    print(f"\n--- Boundary Accuracy (tolerance zone) ---")
    if boundary_total > 0:
        print(f"Accuracy: {boundary_correct}/{boundary_total} = {boundary_correct/boundary_total:.1%}")
    else:
        print(f"No boundary frames")

    print(f"\n--- Overall Accuracy (all frames) ---")
    print(f"Accuracy: {all_correct}/{total_samples} = {all_correct/total_samples:.1%}")

    # 每类准确率（核心帧）
    print(f"\nPer-class accuracy (core frames only):")
    label_correct = Counter()
    label_total = Counter()
    for gt, pred, sid in zip(ground_truths, predictions, sample_ids):
        if sid not in boundary_ids:
            label_total[gt] += 1
            if gt == pred:
                label_correct[gt] += 1

    short = {
        "hold the cube in the starting grip": "hold",
        "rotate the whole cube": "rotate",
        "turn the right layer clockwise": "R_cw",
        "turn the right layer counterclockwise": "R_ccw",
        "turn the up layer clockwise": "U_cw",
        "turn the up layer counterclockwise": "U_ccw",
    }
    for label in SUBTASK_LABELS:
        if label_total[label] > 0:
            acc = label_correct[label] / label_total[label]
            print(f"  {short[label]:>8}: {label_correct[label]:>5}/{label_total[label]:<5} = {acc:.1%}")

    # 混淆矩阵（核心帧）
    confusion = defaultdict(lambda: defaultdict(int))
    for gt, pred, sid in zip(ground_truths, predictions, sample_ids):
        if sid not in boundary_ids:
            confusion[gt][pred] += 1

    print(f"\nConfusion Matrix (core frames, true ↓ vs predicted →):")
    print(f"{'':>10}", end="")
    for s in short.values():
        print(f" {s:>6}", end="")
    print()
    for label in SUBTASK_LABELS:
        print(f"{short[label]:>10}", end="")
        for pred_label in SUBTASK_LABELS:
            c = confusion[label][pred_label]
            print(f" {c:>6}" if c > 0 else f" {'·':>6}", end="")
        print()

    return {
        "core_accuracy": core_correct / core_total if core_total > 0 else None,
        "core_correct": core_correct,
        "core_total": core_total,
        "boundary_accuracy": boundary_correct / boundary_total if boundary_total > 0 else None,
        "boundary_correct": boundary_correct,
        "boundary_total": boundary_total,
        "overall_accuracy": all_correct / total_samples,
        "overall_correct": all_correct,
        "total": total_samples,
        "inference_time_s": round(elapsed, 1),
        "ms_per_sample": round(elapsed / total_samples * 1000),
    }


def save_predictions_as_subtasks(sample_ids, predictions, output_dir):
    """将帧级预测按 episode 分组，合并为 subtasks.json 格式保存。"""
    from datetime import datetime
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 episode 分组，保持帧顺序
    episodes = defaultdict(list)
    for sid, pred in zip(sample_ids, predictions):
        # sample_id: session_xxx_episode_yyyy_frameNNNN
        parts = sid.rsplit("_frame", 1)
        episode_id = parts[0]       # session_xxx_episode_yyyy
        frame_idx = int(parts[1])   # NNNN
        episodes[episode_id].append((frame_idx, pred))

    frame_interval_ns = int(1e9 / SAMPLE_FPS)
    saved = 0

    for episode_id, frames in episodes.items():
        frames.sort(key=lambda x: x[0])

        # 从原始数据读取 duration_s
        # episode_id 格式: session_20261010_173611_lerobot_episode_0002
        # 需要拆分出 session 和 episode
        parts = episode_id.split("_episode_")
        session_name = parts[0]  # session_20261010_173611_lerobot
        ep_name = "episode_" + parts[1]  # episode_0002
        orig_file = Path(TEST_DATA_ROOT) / session_name / ep_name / "subtasks.json"

        if orig_file.exists():
            with open(orig_file) as f:
                orig = json.load(f)
            duration_s = orig["duration_s"]
        else:
            duration_s = (frames[-1][0] + 1) / SAMPLE_FPS

        # 合并连续相同标签为 segments
        segments = []
        if frames:
            cur_label = frames[0][1]
            cur_start_frame = frames[0][0]

            for fi in range(1, len(frames)):
                if frames[fi][1] != cur_label:
                    segments.append({
                        "t_start_ns": cur_start_frame * frame_interval_ns,
                        "t_end_ns": frames[fi][0] * frame_interval_ns,
                        "text": cur_label,
                    })
                    cur_label = frames[fi][1]
                    cur_start_frame = frames[fi][0]

            # 最后一个 segment 延伸到视频结尾
            segments.append({
                "t_start_ns": cur_start_frame * frame_interval_ns,
                "t_end_ns": int(duration_s * 1e9),
                "text": cur_label,
            })

        # 保存
        session_dir = output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        ep_dir = session_dir / ep_name
        ep_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "version": 1,
            "duration_s": duration_s,
            "segments": segments,
            "annotator": "Qwen3-VL-2B-Instruct (finetuned)",
            "annotated_at": datetime.now().isoformat(timespec="seconds"),
        }

        with open(ep_dir / "subtasks_pred.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved += 1

    print(f"\n已保存 {saved} 个 episode 的预测结果到 {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="/home/liuyao/projects/cube_subtask_finetune/data/test.jsonl")
    parser.add_argument("--base-model", default="/home/liuyao/Telephix/Telephix-annotation/cube_subtask_finetune/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--lora-path", default="/home/liuyao/projects/cube_subtask_finetune/checkpoints/lora_final")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tolerance-frames", type=int, default=2,
                        help="边界容忍帧数 (default: 2, 即 ±1s at 2fps)")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--save-predictions", type=str, default=None,
                        help="保存微调模型预测为 subtasks_pred.json 的输出目录")
    parser.add_argument("--output", default="/home/liuyao/projects/cube_subtask_finetune/eval_results.json")
    args = parser.parse_args()

    # 加载测试数据
    samples = [json.loads(l) for l in open(args.test_file)]
    print(f"测试集: {len(samples)} 样本")

    # 构建边界帧集合
    print(f"构建边界帧集合 (tolerance=±{args.tolerance_frames} frames = ±{args.tolerance_frames/SAMPLE_FPS:.1f}s)...")
    boundary_ids = build_boundary_map(args.tolerance_frames)
    print(f"边界帧: {len(boundary_ids)} 个")

    # 加载模型
    print(f"加载模型: {args.base_model}")
    processor = Qwen3VLProcessor.from_pretrained(args.base_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # Baseline
    preds, gts, sids, elapsed = run_inference(model, processor, samples, args.batch_size, max_pixels=120*28*28)
    r1 = print_report("Baseline", preds, gts, sids, boundary_ids, elapsed, len(samples))

    # Fine-tuned
    r2 = None
    ft_preds, ft_gts, ft_sids = None, None, None
    if not args.base_only:
        if Path(args.lora_path).exists():
            print(f"\n加载 LoRA: {args.lora_path}")
            model = PeftModel.from_pretrained(model, args.lora_path)
            model = model.merge_and_unload()
            ft_preds, ft_gts, ft_sids, elapsed = run_inference(model, processor, samples, args.batch_size, max_pixels=120*28*28)
            r2 = print_report("Fine-tuned", ft_preds, ft_gts, ft_sids, boundary_ids, elapsed, len(samples))
        else:
            print("警告: 未找到 LoRA 权重")

    # 保存预测结果
    if args.save_predictions and ft_preds:
        save_predictions_as_subtasks(ft_sids, ft_preds, args.save_predictions)

    # 汇总
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'':>14} {'Core':>10} {'Boundary':>10} {'Overall':>10}")
    print(f"{'Baseline':>14} {r1['core_accuracy']:.1%}" if r1['core_accuracy'] else f"{'Baseline':>14} {'N/A':>10}",
          f"{r1['boundary_accuracy']:.1%}" if r1['boundary_accuracy'] else f"{'N/A':>10}",
          f"{r1['overall_accuracy']:.1%}")
    if r2:
        print(f"{'Fine-tuned':>14} {r2['core_accuracy']:.1%}" if r2['core_accuracy'] else f"{'Fine-tuned':>14} {'N/A':>10}",
              f"{r2['boundary_accuracy']:.1%}" if r2['boundary_accuracy'] else f"{'N/A':>10}",
              f"{r2['overall_accuracy']:.1%}")
        if r1['core_accuracy'] and r2['core_accuracy']:
            delta = r2['core_accuracy'] - r1['core_accuracy']
            print(f"{'Δ':>14} {delta:>+10.1%}")

    results = {"baseline": r1, "finetuned": r2, "tolerance_frames": args.tolerance_frames}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {args.output}")


if __name__ == "__main__":
    main()