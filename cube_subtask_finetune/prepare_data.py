"""
从 1010_six_90 数据集中提取帧并构建训练/测试数据。

按 session 划分 train/test，避免同一 episode 的帧泄漏到两个集合。

用法:
    python prepare_data.py
"""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
OUTPUT_DIR = "/home/liuyao/projects/cube_subtask_finetune/data"
SAMPLE_FPS = 2  # 每秒采样帧数

# 用最后 1 个 session 做测试集
TEST_SESSION = "session_20261010_173611_lerobot"


def extract_frames(video_path: str, output_dir: str, sample_fps: int) -> list[str]:
    """用 ffmpeg 从视频中按指定 fps 提取帧，返回帧文件路径列表。"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={sample_fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg"),
        "-y", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    return [str(f) for f in frames]


def get_label_at_time(segments: list, t_ns: int) -> str | None:
    """根据纳秒时间戳查找对应的 subtask 标签。"""
    for seg in segments:
        if seg["t_start_ns"] <= t_ns < seg["t_end_ns"]:
            return seg["text"]
    return None


def collect_episodes():
    """收集所有有效 episode 并按 session 分组。"""
    sessions = {}
    for session_dir in sorted(Path(DATA_ROOT).glob("session_*")):
        session_name = session_dir.name
        for episode_dir in sorted(session_dir.glob("episode_*")):
            subtasks_file = episode_dir / "subtasks.json"
            video_file = episode_dir / "videos" / "observation.images.global.mp4"
            if subtasks_file.exists() and video_file.exists():
                sessions.setdefault(session_name, []).append({
                    "session": session_name,
                    "episode": episode_dir.name,
                    "episode_path": str(episode_dir),
                    "video_path": str(video_file),
                    "subtasks_path": str(subtasks_file),
                })
    return sessions


def process_episodes(episodes: list, split_name: str) -> list[dict]:
    """处理一组 episode，提取帧并构建训练样本。"""
    dataset = []
    skipped = 0

    for ep in episodes:
        with open(ep["subtasks_path"]) as f:
            subtasks = json.load(f)

        segments = subtasks["segments"]
        episode_id = f"{ep['session']}_{ep['episode']}"

        frame_dir = os.path.join(OUTPUT_DIR, "frames", episode_id)
        frame_paths = extract_frames(ep["video_path"], frame_dir, SAMPLE_FPS)

        frame_interval_ns = int(1e9 / SAMPLE_FPS)
        for i, frame_path in enumerate(frame_paths):
            t_ns = int(i * frame_interval_ns)
            label = get_label_at_time(segments, t_ns)

            if label is None:
                skipped += 1
                continue

            dataset.append({
                "id": f"{episode_id}_frame{i:04d}",
                "image": frame_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nThis is a frame from a video of a robot manipulating a Rubik's cube. What subtask is the robot performing? Choose from: hold the cube in the starting grip, rotate the whole cube, turn the right layer clockwise, turn the right layer counterclockwise, turn the up layer clockwise, turn the up layer counterclockwise."
                    },
                    {
                        "from": "gpt",
                        "value": label
                    }
                ]
            })

    # 保存
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 统计
    labels = [item["conversations"][1]["value"] for item in dataset]
    print(f"\n[{split_name}] 总计: {len(dataset)} 样本, 跳过: {skipped} 帧")
    print(f"[{split_name}] 标签分布:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

    return dataset


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)

    sessions = collect_episodes()
    print(f"找到 {len(sessions)} 个 session:")
    for name, eps in sessions.items():
        tag = " [TEST]" if name == TEST_SESSION else " [TRAIN]"
        print(f"  {name}: {len(eps)} episodes{tag}")

    # 按 session 划分
    train_episodes = []
    test_episodes = []
    for name, eps in sessions.items():
        if name == TEST_SESSION:
            test_episodes.extend(eps)
        else:
            train_episodes.extend(eps)

    print(f"\n训练集: {len(train_episodes)} episodes (session)")
    print(f"测试集: {len(test_episodes)} episodes (session: {TEST_SESSION})")

    train_data = process_episodes(train_episodes, "train")
    test_data = process_episodes(test_episodes, "test")

    # 汇总
    print(f"\n{'='*50}")
    print(f"train.jsonl: {len(train_data)} 样本")
    print(f"test.jsonl:  {len(test_data)} 样本")
    print(f"保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()