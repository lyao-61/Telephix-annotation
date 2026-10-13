"""
方案 B 数据准备：为每个训练/测试帧生成 IMU quaternion 时序图。

原理：
- 在当前帧前后取一个时间窗口的 quaternion 数据
- 画成 4 通道时序图（qw, qx, qy, qz）
- 标注当前帧位置（红色竖线）
- 保存为图片，和摄像头画面一起输入 VLM

用法:
    python prepare_data_b.py
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

# quaternion 时序图的窗口大小（秒）
IMU_WINDOW_S = 2.0  # 前后各 2 秒，共 4 秒

os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "imu_plots"), exist_ok=True)


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
    """加载 quaternion 数据，返回 (timestamps_ns, values) 数组。"""
    with open(cube_samples_path) as f:
        data = json.load(f)

    quats = data.get("/cube/quaternion", [])
    if not quats:
        return None, None

    timestamps = np.array([q["timestamp_ns"] for q in quats])
    values = np.array([q["value"] for q in quats])  # (N, 4) [qw, qx, qy, qz]
    return timestamps, values


def plot_quaternion_window(timestamps, values, current_t_ns, output_path, window_s=IMU_WINDOW_S):
    """画当前帧前后 window_s 秒的 quaternion 时序图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_ns = int(window_s * 1e9)
    t_start = current_t_ns - window_ns
    t_end = current_t_ns + window_ns

    # 取窗口内的数据
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    t_win = timestamps[mask]
    v_win = values[mask]

    if len(t_win) < 2:
        # 数据不足，画空白图
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=50)
        ax.text(0.5, 0.5, "No IMU data", ha="center", va="center", fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return

    # 归一化时间到秒
    t_sec = (t_win - current_t_ns) / 1e9
    labels = ["qw", "qx", "qy", "qz"]
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    fig, axes = plt.subplots(4, 1, figsize=(4, 3), dpi=50, sharex=True)
    for i in range(4):
        ax = axes[i]
        ax.plot(t_sec, v_win[:, i], color=colors[i], linewidth=1.5)
        ax.axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_ylabel(labels[i], fontsize=8, rotation=0, labelpad=20)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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

        # 提取视频帧
        frame_dir = os.path.join(OUTPUT_DIR, "frames", episode_id)
        frame_paths = extract_frames(ep["video_path"], frame_dir, SAMPLE_FPS)

        # 加载 IMU 数据
        imu_timestamps, imu_values = None, None
        if ep["cube_path"]:
            imu_timestamps, imu_values = load_quaternion_data(ep["cube_path"])

        # 生成 IMU 图
        imu_dir = os.path.join(OUTPUT_DIR, "imu_plots", episode_id)
        os.makedirs(imu_dir, exist_ok=True)

        frame_interval_ns = int(1e9 / SAMPLE_FPS)
        for i, frame_path in enumerate(frame_paths):
            t_ns = int(i * frame_interval_ns)
            label = get_label_at_time(segments, t_ns)

            if label is None:
                skipped += 1
                continue

            # 画 IMU 图
            imu_plot_path = os.path.join(imu_dir, f"imu_{i:04d}.jpg")
            if imu_timestamps is not None:
                plot_quaternion_window(imu_timestamps, imu_values, t_ns, imu_plot_path)
            else:
                imu_plot_path = None

            # 构建样本（两张图）
            content = []
            if imu_plot_path:
                content.append({"type": "image"})  # IMU plot (先传)
            content.append({"type": "image"})  # 摄像头画面
            content.append({
                "type": "text",
                "text": (
                    "The first image shows IMU quaternion data (qw/qx/qy/qz) around the current frame "
                    "(red dashed line = current time). The second image is the camera view. "
                    "What subtask is the robot performing? Choose from: "
                    "hold the cube in the starting grip, rotate the whole cube, "
                    "turn the right layer clockwise, turn the right layer counterclockwise, "
                    "turn the up layer clockwise, turn the up layer counterclockwise."
                ),
            })

            images_paths = []
            if imu_plot_path:
                images_paths.append(imu_plot_path)
            images_paths.append(frame_path)

            dataset.append({
                "id": f"{episode_id}_frame{i:04d}",
                "images": images_paths,
                "conversations": [
                    {"from": "human", "value": "<image>" * len(images_paths) + "\n" + content[-1]["text"]},
                    {"from": "gpt", "value": label},
                ],
                # 保留旧格式兼容
                "image": frame_path,
                "imu_plot": imu_plot_path,
            })

    # 保存
    output_file = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    with open(output_file, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    labels = [item["conversations"][1]["value"] for item in dataset]
    print(f"\n[{split_name}] 总计: {len(dataset)} 样本, 跳过: {skipped} 帧")
    print(f"[{split_name}] 标签分布:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

    return dataset


def main():
    sessions = collect_episodes()
    print(f"找到 {len(sessions)} 个 session")

    train_eps, test_eps = [], []
    for name, eps in sessions.items():
        if name == TEST_SESSION:
            test_eps.extend(eps)
        else:
            train_eps.extend(eps)

    print(f"训练集: {len(train_eps)} episodes")
    print(f"测试集: {len(test_eps)} episodes")

    process_episodes(train_eps, "train_b")
    process_episodes(test_eps, "test_b")

    print(f"\n保存到: {OUTPUT_DIR}/train_b.jsonl, test_b.jsonl")


if __name__ == "__main__":
    main()