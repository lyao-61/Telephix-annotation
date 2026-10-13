# 魔方遥操作数据 Subtask 标注方法

## 概述

遥操作采集的魔方数据每条 episode 可分为 6 个子任务阶段。标注依据来自两个传感器源：

- **`/cube/quaternion`**（IMU 四元数）→ 检测 hold 和 rotate 阶段边界
- **`/cube/move`**（魔方固件转动检测）→ 检测 turn 阶段边界

数据位于每条 episode 的 `cube_samples.json` 中。

```
┌─────────┐  ┌──────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  hold   │→│  rotate  │→│ turn right  │→│ turn up     │→│ turn right  │→│ turn up     │
│         │  │          │  │ layer CW    ││ layer CW    ││ layer CCW   ││ layer CCW   │
└─────────┘  └──────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
frame=0    hold_end      rotate_end        move[0]          move[1]          move[2]       move[3]
```

---

## 一、Hold 阶段检测

### 物理含义

操作者从桌面拿起魔方并握稳。此阶段 quaternion 基本不变（噪声底 ~18°/s）。

### 检测方法

找 quaternion 的**第一次大角度跳变**（拿起动作导致角速度飙升）。

```python
def find_hold_end(quats, t0, threshold=35):
    """找第一次角速度超过阈值的时刻 = 拿起魔方的瞬间。

    Args:
        quats: cube_samples.json 中 /cube/quaternion 的采样列表
        t0: 视频起始的绝对时间戳 (ns)，通常取 quats[0]['timestamp_ns']
        threshold: 角速度阈值 (°/s)，默认 35

    Returns:
        hold_end 相对时间 (秒)，未找到返回 None
    """
    window_ns = 1_000_000_000  # 1 秒滑动窗口
    i = 0
    for j in range(1, len(quats)):
        if quats[j]['timestamp_ns'] - quats[i]['timestamp_ns'] >= window_ns:
            q1, q2 = quats[i]['value'], quats[j]['value']
            dot = min(abs(sum(a * b for a, b in zip(q1, q2))), 1.0)
            angle = 2 * math.degrees(math.acos(dot))
            dur = (quats[j]['timestamp_ns'] - quats[i]['timestamp_ns']) / 1e9
            if angle / dur > threshold:
                return (quats[j]['timestamp_ns'] - t0) / 1e9
            i = j
    return None
```

### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `threshold` | 35°/s | 角速度超过此值判定为拿起动作 |

### 典型时间范围

- 视频开始后 **1~6 秒**
- 误差范围：**±1.7 秒**

### 物理过程

```
0s ──────── hold_end ────────
quiescent    │ spike (拿起)
~18°/s       │ >35°/s
```

---

## 二、Rotate 阶段检测

### 物理含义

操作者整体翻转魔方（连续旋转，非离散 90° 步进）。此阶段 quaternion 持续大幅变化。

### 检测方法

从 hold_end 到第一个 move 时间戳之间，找 quaternion 角速度的**最后一个 spike 及其稳定点**。

```python
def find_rotate_end(quats, t0, hold_end_s, move0_s):
    """在 hold_end 和 move0 之间找旋转结束时刻。

    策略：找最后一个角速度 spike (>30°/s)，然后找之后
    角速度回落到噪声底 (<22°/s) 的时刻。

    Args:
        quats: /cube/quaternion 采样列表
        t0: 视频起始绝对时间戳 (ns)
        hold_end_s: hold 阶段结束的相对时间 (秒)
        move0_s: 第一个 move 的相对时间 (秒)

    Returns:
        rotate_end 相对时间 (秒)
    """
    window_ns = 1_000_000_000
    rates = []
    i = 0
    for j in range(1, len(quats)):
        if quats[j]['timestamp_ns'] - quats[i]['timestamp_ns'] >= window_ns:
            rel_s = (quats[j]['timestamp_ns'] - t0) / 1e9
            if hold_end_s <= rel_s <= move0_s:
                q1, q2 = quats[i]['value'], quats[j]['value']
                dot = min(abs(sum(a * b for a, b in zip(q1, q2))), 1.0)
                angle = 2 * math.degrees(math.acos(dot))
                dur = (quats[j]['timestamp_ns'] - quats[i]['timestamp_ns']) / 1e9
                rates.append((rel_s, angle / dur))
            i = j

    if not rates:
        return hold_end_s + 2.0

    # 找最后一个 spike
    last_spike = None
    for t, r in rates:
        if r > 30:
            last_spike = t

    if last_spike is None:
        return hold_end_s + 2.0

    # spike 之后找稳定点
    for t, r in rates:
        if t > last_spike and r < 22:
            return t
    return last_spike
```

### 参数说明

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `spike_threshold` | 30°/s | 旋转 spike 的角速度阈值 |
| `stable_threshold` | 22°/s | 旋转结束后稳定的角速度阈值 |
| `min_stable` | 2 | 连续稳定窗口数 |

### 典型时间范围

- hold_end 后 **8~20 秒**
- 误差范围：**±10 秒**（不稳定，需人工抽查）

### 物理过程

```
hold_end ──── rotate ──── rotate_end ──── prepare ──── move0
              │  spikes   │  stabilizes    │  held still
              │  >30°/s   │  <22°/s        │  ~18°/s
```

### 旋转的轴角特征

以 hold_end 为参考点，rotate 过程中的轴角变化：

| 时间段 | 角度变化 | 含义 |
|--------|----------|------|
| 0~6s | 0°→~2° | 微调握姿 |
| 6~13s | 2°→~17° | 第一次旋转 |
| 13~14s | 17°→~90° | 过渡 |
| 14~25s | 90°→~128° | 第二次旋转 |
| 25s+ | ~128° 稳定 | 旋转结束 |

总旋转角度范围：**90°~143°**，平均 **126.2°**。

---

## 三、Turn 阶段标注

### 物理含义

操作者转动魔方的单层。此阶段 quaternion 不变（魔方整体朝向不变），但 `/cube/state` 的 facelet 会变化。

### 检测方法

直接读取 `cube_samples.json` 中 `/cube/move` 的时间戳。

```python
def load_cube_moves(cube_samples_path, t0):
    """读取 /cube/move 时间戳，转为相对时间。

    Args:
        cube_samples_path: cube_samples.json 路径
        t0: 视频起始绝对时间戳 (ns)

    Returns:
        相对时间列表 (秒)，按时间排序
    """
    with open(cube_samples_path) as f:
        data = json.load(f)
    moves = data.get("/cube/move", [])
    times = [(m['timestamp_ns'] - t0) / 1e9 for m in moves]
    return sorted(times)
```

### 标注规则

每个 move 对应一个 turn 阶段的**起始时间**。标注顺序固定为：

| 段 | Subtask | 边界 |
|----|---------|------|
| 3 | turn the right layer clockwise | rotate_end → move[0] |
| 4 | turn the up layer clockwise | move[0] → move[1] |
| 5 | turn the right layer counterclockwise | move[1] → move[2] |
| 6 | turn the up layer counterclockwise | move[2] → move[3] |

### 注意事项

- `/cube/move` 的 `value` 字段（转动记号如 "R", "U'"）可能不准确，需以实际操作为准
- 如果 move 数量不是 4，该 episode 需要人工标注或跳过
- move 时间戳是绝对纳秒时间，需减去视频起始时间转为相对时间

---

## 四、完整标注流程

```python
import json, math

def annotate_episode(cube_samples_path, meta_path, output_path):
    """为一条 episode 生成 6 段 subtask 标注。"""
    with open(cube_samples_path) as f:
        cs = json.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    quats = cs["/cube/quaternion"]
    moves = cs["/cube/move"]
    t0 = quats[0]['timestamp_ns']

    # 1. 检测 hold 和 rotate 边界
    hold_end_s = find_hold_end(quats, t0) or 3.0
    move_times = sorted([(m['timestamp_ns'] - t0) / 1e9 for m in moves])

    if len(move_times) < 4:
        return  # 跳过 move 不足的 episode

    rotate_end_s = find_rotate_end(quats, t0, hold_end_s, move_times[0])

    # 2. 构建 6 段 subtask
    turn_labels = [
        "turn the right layer clockwise",
        "turn the up layer clockwise",
        "turn the right layer counterclockwise",
        "turn the up layer counterclockwise",
    ]

    boundaries = [0, hold_end_s, rotate_end_s] + move_times[:4]
    texts = [
        "hold the cube in the starting grip",
        "rotate the whole cube",
    ] + turn_labels

    segments = []
    for i in range(len(texts)):
        segments.append({
            "t_start_ns": int(boundaries[i] * 1e9),
            "t_end_ns": int(boundaries[i + 1] * 1e9),
            "text": texts[i],
        })

    # 3. 输出 subtasks.json
    n_frames = meta.get("n_frames", 0)
    fps = meta.get("fps", 30)
    payload = {
        "version": 1,
        "duration_s": n_frames / fps,
        "segments": segments,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
```

---

## 五、阈值参数汇总

| 参数 | 值 | 用途 |
|------|-----|------|
| hold spike threshold | 35°/s | 检测拿起动作 |
| rotate spike threshold | 30°/s | 检测旋转 spike |
| rotate stable threshold | 22°/s | 检测旋转结束 |
| 滑动窗口 | 1 秒 | 角速度计算窗口 |
| 噪声底 | ~18°/s | quaternion 精度限制导致的量化噪声 |

---

## 六、数据结构

### cube_samples.json

```json
{
  "/cube/quaternion": [
    {"timestamp_ns": 1791598237601701257, "value": [0.831, 0.006, 0.552, -0.015]},
    ...
  ],
  "/cube/move": [
    {"timestamp_ns": 1791598237601701257, "value": "R"},
    ...
  ],
  "/cube/state": [
    {"timestamp_ns": ..., "value": "54-char facelet string"},
    ...
  ]
}
```

| Topic | 含义 | 采样频率 | 标注用途 |
|-------|------|----------|----------|
| `/cube/quaternion` | IMU 四元数 (x,y,z,w) | ~60ms | **检测 hold/rotate 边界** |
| `/cube/move` | 固件检测到的单层转动 | 事件触发 | **检测 turn 边界** |
| `/cube/state` | 54 个 facelet 颜色编码 | ~150ms | 识别魔方内部状态（标注未使用） |

### subtasks.json

```json
{
  "version": 1,
  "duration_s": 41.61,
  "segments": [
    {"t_start_ns": 0, "t_end_ns": 5100000000, "text": "hold the cube in the starting grip"},
    {"t_start_ns": 5100000000, "t_end_ns": 19500000000, "text": "rotate the whole cube"},
    {"t_start_ns": 19500000000, "t_end_ns": 22770000000, "text": "turn the right layer clockwise"},
    {"t_start_ns": 22770000000, "t_end_ns": 28710000000, "text": "turn the up layer clockwise"},
    {"t_start_ns": 28710000000, "t_end_ns": 34680000000, "text": "turn the right layer counterclockwise"},
    {"t_start_ns": 34680000000, "t_end_ns": 41610000000, "text": "turn the up layer counterclockwise"}
  ],
  "annotator": "auto_quaternion",
  "annotated_at": "2026-10-10"
}
```

---

## 七、已知限制

1. **rotate_end 检测不稳定**：误差可达 ±10 秒，需人工抽查确认
2. **四元数符号问题**：部分 episode 的四元数 w 分量符号不一致，需预处理为 w≥0
3. **move value 不准确**：`/cube/move` 的 `value` 字段（转动记号）可能与实际操作不符，标注时以实际动作为准
4. **非标准 episode**：move 数量不是 4 的 episode 需要人工标注