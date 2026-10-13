"""
对视频逐帧 subtask 预测进行时序后处理，消除碎片化。

方法：
1. 滑动窗口多数投票（中值滤波）
2. 最短 segment 合并
3. （可选）Viterbi 解码 — 考虑转移代价

用法:
    # 基础：中值滤波 + 最短 segment
    python postprocess.py predictions/

    # 调参
    python postprocess.py predictions/ --window 7 --min-duration 1.0

    # Viterbi 解码
    python postprocess.py predictions/ --method viterbi
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

SUBTASK_LABELS = [
    "hold the cube in the starting grip",
    "rotate the whole cube",
    "turn the right layer clockwise",
    "turn the right layer counterclockwise",
    "turn the up layer clockwise",
    "turn the up layer counterclockwise",
]

LABEL2IDX = {l: i for i, l in enumerate(SUBTASK_LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}


def smooth_majority_vote(labels: list[str], window: int = 5) -> list[str]:
    """滑动窗口多数投票。"""
    half = window // 2
    smoothed = []
    for i in range(len(labels)):
        start = max(0, i - half)
        end = min(len(labels), i + half + 1)
        window_labels = labels[start:end]
        # 多数投票，平局取中间帧的原标签
        counts = Counter(window_labels)
        most_common = counts.most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed


def smooth_viterbi(labels: list[str], penalty: float = 3.0) -> list[str]:
    """
    Viterbi 解码：在观测序列（逐帧预测）上加入转移代价。
    penalty: 切换 subtask 的代价（越大越倾向保持相同标签）。
    """
    n = len(labels)
    if n == 0:
        return labels

    n_labels = len(SUBTASK_LABELS)
    obs = [LABEL2IDX.get(l, 0) for l in labels]

    # dp[t][j] = 到第 t 帧、状态 j 的最大 log-prob
    # backptr[t][j] = 前一帧的最佳状态
    dp = [[0.0] * n_labels for _ in range(n)]
    backptr = [[0] * n_labels for _ in range(n)]

    # 初始：观测匹配得分
    for j in range(n_labels):
        dp[0][j] = 1.0 if j == obs[0] else 0.0

    for t in range(1, n):
        for j in range(n_labels):
            best_score = float("-inf")
            best_prev = 0
            obs_score = 1.0 if j == obs[t] else 0.0
            for k in range(n_labels):
                trans = -penalty if k != j else 0.0
                score = dp[t - 1][k] + trans + obs_score
                if score > best_score:
                    best_score = score
                    best_prev = k
            dp[t][j] = best_score
            backptr[t][j] = best_prev

    # 回溯
    path = [0] * n
    path[-1] = max(range(n_labels), key=lambda j: dp[-1][j])
    for t in range(n - 2, -1, -1):
        path[t] = backptr[t + 1][path[t + 1]]

    return [IDX2LABEL[i] for i in path]


def merge_short_segments(segments: list[dict], min_duration_ns: int) -> list[dict]:
    """合并过短的 segment 到相邻 segment。"""
    if len(segments) <= 1:
        return segments

    merged = []
    for seg in segments:
        duration = seg["t_end_ns"] - seg["t_start_ns"]
        if duration < min_duration_ns and merged:
            # 合并到前一个 segment
            merged[-1]["t_end_ns"] = seg["t_end_ns"]
        else:
            merged.append(dict(seg))

    # 反向也做一轮（最后一个 segment 可能太短）
    if len(merged) > 1:
        final = []
        for i, seg in enumerate(merged):
            duration = seg["t_end_ns"] - seg["t_start_ns"]
            if duration < min_duration_ns and i < len(merged) - 1:
                # 合并到后一个
                merged[i + 1]["t_start_ns"] = seg["t_start_ns"]
            else:
                final.append(seg)
        merged = final

    return merged


def labels_to_segments(labels: list[str], fps: float, duration_s: float) -> list[dict]:
    """将帧级标签序列转换为 segment 列表。"""
    if not labels:
        return []

    frame_ns = int(1e9 / fps)
    segments = []
    cur_label = labels[0]
    cur_start = 0

    for i in range(1, len(labels)):
        if labels[i] != cur_label:
            segments.append({
                "t_start_ns": cur_start * frame_ns,
                "t_end_ns": i * frame_ns,
                "text": cur_label,
            })
            cur_label = labels[i]
            cur_start = i

    segments.append({
        "t_start_ns": cur_start * frame_ns,
        "t_end_ns": int(duration_s * 1e9),
        "text": cur_label,
    })
    return segments


def process_episode(ep_dir: Path, args):
    """对一个 episode 的预测做后处理。"""
    pred_file = ep_dir / "subtasks_pred.json"
    if not pred_file.exists():
        return

    with open(pred_file) as f:
        pred_data = json.load(f)

    segments = pred_data["segments"]
    fps = 2.0
    duration_s = pred_data["duration_s"]

    # 还原帧级标签
    labels = []
    frame_ns = int(1e9 / fps)
    for seg in segments:
        start_frame = int(seg["t_start_ns"] / frame_ns)
        end_frame = int(seg["t_end_ns"] / frame_ns)
        for _ in range(end_frame - start_frame):
            labels.append(seg["text"])

    n_before = len(segments)

    # 步骤 1：平滑
    if args.method == "viterbi":
        labels = smooth_viterbi(labels, penalty=args.penalty)
    else:
        labels = smooth_majority_vote(labels, window=args.window)

    # 步骤 2：转换为 segments
    segments = labels_to_segments(labels, fps, duration_s)

    # 步骤 3：合并短 segment
    if args.min_duration > 0:
        min_ns = int(args.min_duration * 1e9)
        segments = merge_short_segments(segments, min_ns)

    n_after = len(segments)

    # 保存
    result = {
        "version": 1,
        "duration_s": duration_s,
        "segments": segments,
        "annotator": f"Qwen3-VL-2B-Instruct (postprocess: {args.method}, window={args.window}, min_dur={args.min_duration}s)",
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
    }

    out_file = ep_dir / "subtasks_pred.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return n_before, n_after


def main():
    parser = argparse.ArgumentParser(description="Subtask 预测后处理")
    parser.add_argument("pred_dir", help="predictions 目录（含各 session/episode 子目录）")
    parser.add_argument("--method", choices=["median", "viterbi"], default="median")
    parser.add_argument("--window", type=int, default=7, help="中值滤波窗口大小 (default: 7, 即 3.5s)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="最短 segment 时长(秒) (default: 0.5)")
    parser.add_argument("--penalty", type=float, default=3.0, help="Viterbi 切换代价 (default: 3.0)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    total_before = 0
    total_after = 0

    for session_dir in sorted(pred_dir.glob("session_*")):
        for ep_dir in sorted(session_dir.glob("episode_*")):
            result = process_episode(ep_dir, args)
            if result:
                n_before, n_after = result
                total_before += n_before
                total_after += n_after
                if n_before != n_after:
                    print(f"  {ep_dir.name}: {n_before} → {n_after} segments")

    print(f"\n总计: {total_before} → {total_after} segments")
    print(f"平均每个 episode: {total_before/60:.1f} → {total_after/60:.1f} segments")


if __name__ == "__main__":
    main()