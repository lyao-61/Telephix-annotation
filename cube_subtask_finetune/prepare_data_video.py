"""
视频输入数据准备：直接用 Qwen3-VL 的视频理解能力做 subtask 时间分割。

每个样本：
- 输入: 完整 episode 视频（global 相机）+ 提示词
- 输出: subtask 时间线文本（模型学会按时序输出所有 segment）

用法:
    python prepare_data_video.py
"""

import json
import os
from collections import Counter
from pathlib import Path

DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
OUTPUT_DIR = "/home/liuyao/projects/cube_subtask_finetune/data"
TEST_SESSION = "session_20261010_173611_lerobot"
VIDEO_NAME = "observation.images.global.mp4"

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


def format_timeline(segments: list, duration_s: float) -> str:
    """将 subtasks.json 的 segments 转为文本时间线。"""
    lines = []
    for seg in segments:
        t_start = seg["t_start_ns"] / 1e9
        t_end = seg["t_end_ns"] / 1e9
        lines.append(f"[{t_start:.1f}-{t_end:.1f}] {seg['text']}")
    return "\n".join(lines)


def collect_episodes():
    sessions = {}
    for session_dir in sorted(Path(DATA_ROOT).glob("session_*")):
        session_name = session_dir.name
        for episode_dir in sorted(session_dir.glob("episode_*")):
            subtasks_file = episode_dir / "subtasks.json"
            video_file = episode_dir / "videos" / VIDEO_NAME
            if subtasks_file.exists() and video_file.exists():
                sessions.setdefault(session_name, []).append({
                    "session": session_name,
                    "episode": episode_dir.name,
                    "video_path": str(video_file),
                    "subtasks_path": str(subtasks_file),
                })
    return sessions


def process_episodes(episodes, split_name):
    dataset = []

    for ep in episodes:
        with open(ep["subtasks_path"]) as f:
            subtasks = json.load(f)

        segments = subtasks["segments"]
        duration_s = subtasks["duration_s"]
        episode_id = f"{ep['session']}_{ep['episode']}"

        # 输出时间线
        timeline = format_timeline(segments, duration_s)

        # 输入：完整视频
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": ep["video_path"]},
                {"type": "text", "text": PROMPT},
            ]},
            {"role": "assistant", "content": timeline},
        ]

        dataset.append({
            "id": episode_id,
            "video": ep["video_path"],
            "duration_s": duration_s,
            "messages": messages,
            # 兼容旧格式
            "conversations": [
                {"from": "human", "value": f"<video>\n{PROMPT}"},
                {"from": "gpt", "value": timeline},
            ],
        })

    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    n_segments = sum(len(json.loads(open(ep["subtasks_path"]).read())["segments"]) for ep in episodes)
    print(f"\n[{split_name}] {len(dataset)} episodes, avg {n_segments/max(len(episodes),1):.1f} segments/ep")
    print(f"  Duration range: {min(e for e in [json.loads(open(ep['subtasks_path']).read())['duration_s'] for ep in episodes]):.1f}s"
          f" - {max(e for e in [json.loads(open(ep['subtasks_path']).read())['duration_s'] for ep in episodes]):.1f}s")

    return dataset


def main():
    sessions = collect_episodes()
    train_eps, test_eps = [], []
    for name, eps in sessions.items():
        (test_eps if name == TEST_SESSION else train_eps).extend(eps)

    print(f"训练集: {len(train_eps)} episodes")
    print(f"测试集: {len(test_eps)} episodes")
    print(f"视频: {VIDEO_NAME}")

    process_episodes(train_eps, "train_video")
    process_episodes(test_eps, "test_video")
    print(f"\n保存到: {OUTPUT_DIR}/train_video.jsonl, test_video.jsonl")


if __name__ == "__main__":
    main()