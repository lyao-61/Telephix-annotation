"""
多帧时序数据准备：连续帧序列 + IMU quaternion 图 + 改进的提示词。

原理：
- 每个样本取连续 K 帧（默认4帧 = 2秒），模型能"看到"运动
- IMU quaternion 时序图辅助判断旋转
- 提示词引导模型关注：换手过程、面旋转方向

用法:
    python prepare_data_seq.py
"""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

DATA_ROOT = "/run/media/liuyao/Y/data_studio/prj_20260923_185840/data/1010_six_90"
OUTPUT_DIR = "/home/liuyao/projects/cube_subtask_finetune/data"
SAMPLE_FPS = 2
TEST_SESSION = "session_20261010_173611_lerobot"

NUM_FRAMES = 4        # 每个样本取4帧连续帧
FRAME_STRIDE = 1      # 帧间距（1 = 连续，2 = 隔一帧取）
IMU_WINDOW_S = 3.0    # IMU 图时间窗口

os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "imu_plots"), exist_ok=True)

PROMPT = (
    "You are watching a robot manipulating a Rubik's cube. "
    "The images show a sequence of frames from a video (in temporal order) "
    "followed by an IMU quaternion plot (qw/qx/qy/qz over time, red line = current moment).\n\n"
    "Analyze the MOTION between frames to determine the subtask:\n"
    "- 'hold the cube in the starting grip': cube is static, hands are holding still\n"
    "- 'rotate the whole cube': the whole cube passes from one hand to the other "
    "(left → right → left is one complete rotate); cube orientation changes but no layer is turned\n"
    "- 'turn the right layer clockwise': the right face of the cube rotates 90° clockwise "
    "(looking at the right face)\n"
    "- 'turn the right layer counterclockwise': the right face rotates 90° counterclockwise\n"
    "- 'turn the up layer clockwise': the top face rotates 90° clockwise (looking from above)\n"
    "- 'turn the up layer counterclockwise': the top face rotates 90° counterclockwise\n\n"
    "Focus on: which face of the cube has rotated 90°? Which hand is holding the cube? "
    "Is there a hand-transfer motion? "
    "Answer with only the subtask label."
)


def extract_frames(video_path, output_dir, sample_fps):
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


def load_quaternion_data(cube_samples_path):
    with open(cube_samples_path) as f:
        data = json.load(f)
    quats = data.get("/cube/quaternion", [])
    if not quats:
        return None, None
    timestamps = np.array([q["timestamp_ns"] for q in quats])
    values = np.array([q["value"] for q in quats])
    return timestamps, values


def plot_quaternion_window(timestamps, values, current_t_ns, output_path, window_s=IMU_WINDOW_S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_ns = int(window_s * 1e9)
    mask = (timestamps >= current_t_ns - window_ns) & (timestamps <= current_t_ns + window_ns)
    t_win = timestamps[mask]
    v_win = values[mask]

    fig, axes = plt.subplots(4, 1, figsize=(4, 3), dpi=50, sharex=True)

    if len(t_win) < 2:
        axes[0].text(0.5, 0.5, "No IMU data", ha="center", va="center", transform=axes[0].transAxes)
        for ax in axes:
            ax.axis("off")
    else:
        t_sec = (t_win - current_t_ns) / 1e9
        labels = ["qw", "qx", "qy", "qz"]
        colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
        for i in range(4):
            axes[i].plot(t_sec, v_win[:, i], color=colors[i], linewidth=1.5)
            axes[i].axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)
            axes[i].set_ylabel(labels[i], fontsize=8, rotation=0, labelpad=20)
            axes[i].tick_params(labelsize=6)
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
        axes[-1].set_xlabel("t (s)", fontsize=8)
        axes[0].set_title("IMU Quaternion", fontsize=9, pad=3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def get_label_at_time(segments, t_ns):
    for seg in segments:
        if seg["t_start_ns"] <= t_ns < seg["t_end_ns"]:
            return seg["text"]
    return None


def collect_episodes():
    sessions = {}
    for session_dir in sorted(Path(DATA_ROOT).glob("session_*")):
        session_name = session_dir.name
        for episode_dir in sorted(session_dir.glob("episode_*")):
            subtasks_file = episode_dir / "subtasks.json"
            video_file = episode_dir / "videos" / "observation.images.global.mp4"
            cube_file = episode_dir / "cube_samples.json"
            if subtasks_file.exists() and video_file.exists():
                sessions.setdefault(session_name, []).append({
                    "session": session_name,
                    "episode": episode_dir.name,
                    "episode_path": str(episode_dir),
                    "video_path": str(video_file),
                    "subtasks_path": str(subtasks_file),
                    "cube_path": str(cube_file) if cube_file.exists() else None,
                })
    return sessions


def process_episodes(episodes, split_name):
    dataset = []
    skipped = 0

    for ep in episodes:
        with open(ep["subtasks_path"]) as f:
            subtasks = json.load(f)
        segments = subtasks["segments"]
        episode_id = f"{ep['session']}_{ep['episode']}"

        frame_dir = os.path.join(OUTPUT_DIR, "frames", episode_id)
        frame_paths = extract_frames(ep["video_path"], frame_dir, SAMPLE_FPS)

        imu_timestamps, imu_values = None, None
        if ep["cube_path"]:
            imu_timestamps, imu_values = load_quaternion_data(ep["cube_path"])

        imu_dir = os.path.join(OUTPUT_DIR, "imu_plots", episode_id)
        os.makedirs(imu_dir, exist_ok=True)

        frame_interval_ns = int(1e9 / SAMPLE_FPS)
        n_frames = len(frame_paths)

        # 滑动窗口：每个样本取连续 NUM_FRAMES 帧
        for center in range(0, n_frames):
            # 构建帧序列
            seq_indices = []
            for k in range(NUM_FRAMES):
                fi = center + k * FRAME_STRIDE
                if fi < n_frames:
                    seq_indices.append(fi)

            if len(seq_indices) < 2:
                continue

            # 标签：用中心帧的时间（第一帧的时间）
            center_t_ns = seq_indices[0] * frame_interval_ns
            label = get_label_at_time(segments, center_t_ns)

            if label is None:
                skipped += 1
                continue

            # IMU 图（以中心帧时刻为中心）
            mid_t_ns = seq_indices[len(seq_indices)//2] * frame_interval_ns
            imu_plot_path = os.path.join(imu_dir, f"imu_{seq_indices[0]:04d}.jpg")
            if imu_timestamps is not None:
                plot_quaternion_window(imu_timestamps, imu_values, mid_t_ns, imu_plot_path)
            else:
                imu_plot_path = None

            # 图片列表: [frame1, frame2, ..., frameK, imu_plot]
            images_paths = [frame_paths[fi] for fi in seq_indices]
            if imu_plot_path:
                images_paths.append(imu_plot_path)

            # 提示词中标明帧数
            n_img = len(images_paths)
            seq_prompt = f"{PROMPT}\n\nThere are {NUM_FRAMES} consecutive camera frames followed by 1 IMU plot ({n_img} images total)."

            dataset.append({
                "id": f"{episode_id}_frame{seq_indices[0]:04d}",
                "images": images_paths,
                "conversations": [
                    {"from": "human", "value": "<image>" * n_img + "\n" + seq_prompt},
                    {"from": "gpt", "value": label},
                ],
                "image": frame_paths[seq_indices[0]],  # 兼容旧格式
                "imu_plot": imu_plot_path,
            })

    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    labels = [item["conversations"][1]["value"] for item in dataset]
    print(f"\n[{split_name}] 总计: {len(dataset)} 样本, 跳过: {skipped}")
    print(f"[{split_name}] 标签分布:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")
    return dataset


def main():
    sessions = collect_episodes()
    train_eps, test_eps = [], []
    for name, eps in sessions.items():
        (test_eps if name == TEST_SESSION else train_eps).extend(eps)

    print(f"训练集: {len(train_eps)} episodes, 测试集: {len(test_eps)} episodes")
    print(f"每个样本: {NUM_FRAMES} 帧 + 1 IMU 图")

    process_episodes(train_eps, "train_seq")
    process_episodes(test_eps, "test_seq")
    print(f"\n保存到: {OUTPUT_DIR}/train_seq.jsonl, test_seq.jsonl")


if __name__ == "__main__":
    main()