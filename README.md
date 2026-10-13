# Telephix 数据标注（独立工具）

从 Telephix-teleop 抽出来的**纯数据标注**工具,自包含、不依赖 ROS,也不依赖
teleop 代码/数据库。放在标注员自己的电脑上跑,标完用「传输本 session」把数据
rsync 回收数据的电脑。

> 给标注员的完整使用说明：[使用文档.md](使用文档.md)。
> 一页版流程图：[标注流程图.png](标注流程图.png)（可打印贴工位）；
> 培训用讲义：[标注员手册.pptx](标注员手册.pptx)。随包一起分发。
> 注意：PPT 已按实际使用做过删减，比「使用文档.md」和流程图 PNG 简略，以 md 为准。

## 新电脑部署

要求:Ubuntu/Debian(apt)、Python ≥ 3.10、能联网(装依赖)。

```bash
# 1. 在你的电脑上打包(生成 dist/Telephix-annotation-<日期>.tar.gz)
bash pack.sh

# 2. 拷到新电脑,解压并启动
tar xzf Telephix-annotation-*.tar.gz
cd Telephix-annotation
bash run.sh                # → http://localhost:8097
```

`run.sh` 第一次运行会自动调用 `install.sh`:
- 缺什么系统工具就用 `sudo apt-get` 装(`zenity` 选文件夹、`rsync` + `sshpass` 传输、
  必要时 `python3-venv`),会要 sudo 密码;
- 在项目目录建 `.venv`,`pip install -r requirements.txt`(只有 `nicegui`)。

之后每次直接 `bash run.sh`。

- pip 下载慢:`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash install.sh`
- 端口被占用:`TELEPHIX_ANNOTATION_PORT=8098 bash run.sh`
- 打包不含 `.venv/`、`.nicegui/`(本机登录信息)、`dist/`、`__pycache__`。

## 用法

1. **登录**:输入采集员和标注员名字。标注员写进每个 sidecar 的 `annotator` 字段;
   传输时两个名字都会加进对方那边的文件夹名。
2. **方案**(`/projects`):新建一个「标注方案」,定义要标哪些字段 + 类型:
   - `string` → subtask 长时程语言(时间轴分段)
   - `detection` → 检测框 bbox(在画面上拖框 + 物体类别)
   - `int / float / bool` → 数值 typed 分段
3. **进入标注**(`/annotate`):
   - 填「Scan folder」指向 lerobot 数据根(内含 `*_lerobot/episode_*`),Apply。
   - 三路视频 + 时间轴 + 接触模式 C1–C6 + subtask + 检测框/数值标注 + episode 导航。
   - **快捷键**:`Space` 播放/暂停(视频轴上也可)、`←/→` 快退快进(Shift 微调)、
     `1–0` subtask 预设速填(在「数字键速填」里先给 1–0 各设一句;临时,退出清空)、
     `Enter` 在输入框内添加到光标。
   - **保存**:写出 `subtasks.json` / `contact_modes.json` / `<field>.json` /
     `bbox.json` 到每个 episode 目录(和 teleop 标注格式一致)。
4. **传输**(页面底部「传输本 session」):填对方 `user@ip`、文件位置、密码,rsync
   把当前 session 文件夹传过去,对方那边的文件夹名自动加上姓名:
   `session_xxx_lerobot` → `session_xxx_采集-张三_标注-李四_lerobot`(本地文件夹名不变)。

## 数据位置

- **标注方案**(schema)存在 `~/telephix_annotation/projects.json`
  (可用环境变量 `TELEPHIX_ANNOTATION_DIR` 改)。不随项目打包,每台电脑自己建。
- **标注结果**(sidecar)直接写在你选的数据文件夹的各 episode 目录里,不进本工具的库。
- **登录信息**(采集员/标注员名字、上次的 Scan folder)存在项目目录 `.nicegui/`。

## 目录结构

```
Telephix-annotation/
├── app.py              入口 (:8097)
├── run.sh              启动(首次自动安装)
├── install.sh          安装系统工具 + .venv
├── pack.sh             打包成 dist/*.tar.gz
├── requirements.txt    Python 依赖
├── auth.py             登录守卫
├── state.py            S.active_project(标注 UI 唯一共享状态)
├── schema_store.py     标注方案(字段定义)JSON 存储
├── rsync_transfer.py   rsync 传输 + 带姓名的文件夹名
├── styles.py           CSS
├── media.py            episode 视频媒体 URL 注册
├── dirpick.py          zenity 文件夹选择器
├── pages/
│   ├── login.py        '/'          登录
│   ├── projects.py     '/projects'  方案列表 + 编辑
│   └── annotate.py     '/annotate'  标注主界面 + 传输
└── panels/             标注 UI(从 teleop 复制,已去掉 ROS 相关的 Re-bake 按钮)
    ├── annotate_stage.py
    ├── project_annotate.py
    ├── bbox_annotator.py
    └── value_annotator.py
```

## 与 teleop 的关系

- 代码是**复制**过来的快照,改这里不影响 teleop,反之亦然。
- 已移除 teleop 专有的「Re-bake into training dataset」按钮(它依赖 ROS 转换器
  `scripts/rosbag2_to_lerobot.py`)。标注只负责写 sidecar;要烤进 LeRobot 数据集
  仍回 teleop 用转换器。
