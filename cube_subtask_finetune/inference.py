"""
用微调后的 Qwen3-VL 对长视频进行 subtask 时间分割推理。

用法:
    python inference.py <video_path> [--fps 2] [--output result.json]

输出格式 (JSON):
{
    "video": "path/to/video.mp4",
    "fps": 2.0,
    "segments": [
        {"t_start_s": 0.0, "t_end_s": 10.5, "text": "hold the cube in the starting grip"},
        {"t_start_s": 10.5, "t_end_s": 25.0, "text": "rotate the whole cube"},
        ...
    ]
}
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# 6个 subtask 标签
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


def extract_frames(video_path: str, output_dir: str, fps: float) -> list[str]:
    """从视频中按指定 fps 提取帧。"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg"),
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    return sorted(str(f) for f in Path(output_dir).glob("frame_*.jpg"))


def get_video_duration(video_path: str) -> float:
    """获取视频时长(秒)。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def classify_frames(
    frame_paths: list[str],
    model,
    processor,
    fps: float,
    batch_size: int = 8,
) -> list[dict]:
    """对每帧进行 subtask 分类。"""
    results = []

    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        # 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": PROMPT},
                ]
            }
            for img in batch_images
        ]

        # 批量处理
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        inputs = processor(
            text=texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # 推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # 解码输出
        for j, output in enumerate(outputs):
            # 只取新生成的 token
            input_length = inputs["input_ids"].shape[1]
            generated = output[input_length:]
            label = processor.decode(generated, skip_special_tokens=True).strip()

            # 匹配最近的标签
            matched_label = match_label(label)
            t_start_s = (i + j) / fps

            results.append({
                "frame_idx": i + j,
                "t_start_s": t_start_s,
                "t_end_s": t_start_s + 1.0 / fps,
                "text": matched_label,
            })

    return results


def match_label(predicted: str) -> str:
    """将预测文本匹配到最近的已知标签。"""
    predicted_lower = predicted.lower().strip()
    for label in SUBTASK_LABELS:
        if label.lower() in predicted_lower or predicted_lower in label.lower():
            return label
    # 如果没有精确匹配，返回原始预测
    return predicted


def merge_segments(results: list[dict], fps: float) -> list[dict]:
    """将连续相同标签的帧合并成 segment。"""
    if not results:
        return []

    segments = []
    current_label = results[0]["text"]
    current_start = results[0]["t_start_s"]

    for i in range(1, len(results)):
        if results[i]["text"] != current_label:
            segments.append({
                "t_start_s": round(current_start, 2),
                "t_end_s": round(results[i]["t_start_s"], 2),
                "text": current_label,
            })
            current_label = results[i]["text"]
            current_start = results[i]["t_start_s"]

    # 添加最后一个 segment
    segments.append({
        "t_start_s": round(current_start, 2),
        "t_end_s": round(results[-1]["t_end_s"], 2),
        "text": current_label,
    })

    return segments


def main():
    parser = argparse.ArgumentParser(description="Subtask 时间分割推理")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("--fps", type=float, default=2.0, help="采样帧率 (默认: 2)")
    parser.add_argument("--output", help="输出 JSON 路径")
    parser.add_argument("--base-model", default="/home/liuyao/Telephix/Telephix-annotation/cube_subtask_finetune/model/Qwen3-VL-2B-Instruct",
                        help="基础模型名称")
    parser.add_argument("--lora-path",
                        default="/home/liuyao/projects/cube_subtask_finetune/checkpoints/lora_final",
                        help="LoRA 权重路径")
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.base_model}")
    processor = Qwen3VLProcessor.from_pretrained(args.base_model)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if os.path.exists(args.lora_path):
        print(f"加载 LoRA: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        model = model.merge_and_unload()
    else:
        print("警告: 未找到 LoRA 权重，使用基础模型")
        model = base_model

    # 提取帧
    duration = get_video_duration(args.video)
    print(f"视频时长: {duration:.1f}s, 采样 {args.fps}fps")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_paths = extract_frames(args.video, tmpdir, args.fps)
        print(f"提取了 {len(frame_paths)} 帧")

        # 分类
        print("开始分类...")
        results = classify_frames(frame_paths, model, processor, args.fps)

    # 合并 segment
    segments = merge_segments(results, args.fps)

    # 输出
    output = {
        "video": args.video,
        "fps": args.fps,
        "duration_s": round(duration, 2),
        "segments": segments,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n结果保存到: {args.output}")
    else:
        print("\n" + json.dumps(output, indent=2, ensure_ascii=False))

    # 打印摘要
    print(f"\n共识别出 {len(segments)} 个 segment:")
    for seg in segments:
        print(f"  [{seg['t_start_s']:.1f}s - {seg['t_end_s']:.1f}s] {seg['text']}")


if __name__ == "__main__":
    main()
