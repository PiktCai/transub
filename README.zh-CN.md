# Transub 使用指南

[English README](https://github.com/PiktCai/transub/blob/main/README.md)

Transub 通过 Typer 命令行，**提取**视频字幕并加以**翻译**：使用 `ffmpeg` 抽取音频，借助 Whisper 完成转写，并由 LLM 进行翻译，生成可直接使用的字幕文件。

## 目录

- [概览](#概览)
- [功能亮点](#功能亮点)
- [安装](#安装)
  - [1. 基础依赖](#1-基础依赖)
  - [2. 安装 Transub](#2-安装-transub)
  - [3. 转写引擎](#3-转写引擎)
  - [4. 初始化 Transub](#4-初始化-transub)
  - [5. 运行流水线](#5-运行流水线)
- [配置总览](#配置总览)
- [常用命令速查](#常用命令速查)
- [开发者指南](#开发者指南)
- [目录结构](#目录结构)
- [许可协议](#许可协议)

## 概览

Transub 的标准流水线如下：

1. 使用 `ffmpeg` 从视频中提取音频。
2. 通过本地 `faster-whisper` 生成带时间戳的语音转写。
3. 将字幕分批发送给 LLM，使用 JSON 约束确保输出稳定。
4. 输出 `.srt` / `.vtt` 文件，控制行长、断句和时间轴偏移。

所有中间状态都会写入工作目录，意外中断后可以就地恢复。

## 功能亮点

- **一键处理**：`transub run <视频文件>` 即可完成提取 → 转写 → 翻译 → 导出。
- **专一的转写引擎**：只使用 `faster-whisper`，默认保留词级时间戳，方便后续字幕切分和时间轴优化。
- **稳定翻译**：JSON 约束、自动重试、可调节批量大小。
- **字幕排版友好**：智能断句、时间轴微调，可选的多脚本间距优化。
- **断点续跑**：默认缓存目录位于 `~/.cache/transub`，可保存音频、分段和翻译进度，避免重复计算。

## 安装

### 1. 基础依赖

- **Python 3.10+**
- **ffmpeg**：需安装并确保在系统 `PATH` 中可用。
  - **Windows**：`winget install Gyan.FFmpeg` 或 `choco install ffmpeg`
  - **macOS**：`brew install ffmpeg`
  - **Linux**：`sudo apt update && sudo apt install ffmpeg`（Debian/Ubuntu）或 `sudo pacman -S ffmpeg`（Arch）

### 2. 安装 Transub

使用 `uv`（推荐）

`uv` 是一个快速的 Python 包安装器和解析器。它会在隔离环境中安装命令行工具。

```bash
uv tool install transub
```

后续升级可执行：

```bash
uv tool upgrade transub
```

### 3. 转写引擎

Transub 现在只使用 **faster-whisper**。

这样用户不需要理解 API 转写、`whisper.cpp` 模型文件、MLX 转换或其他 ASR 包。源码开发时安装依赖即可：

```bash
uv sync
```

### 4. 初始化 Transub

运行交互式向导生成配置文件：

```bash
transub init
```

向导会引导你选择 faster-whisper 模型尺寸以及翻译所需的 LLM 提供方。

### 5. 运行流水线

```bash
transub run /path/to/video.mp4
```

生成的字幕默认保存在原视频所在目录，支持 `.srt` 与 `.vtt`。如仅需原始转写，可在命令中追加 `--transcribe-only`。

第一次使用本地模型时，建议先准备模型：

```bash
transub prepare-model
```

它会下载或初始化当前配置的本地 ASR 模型。后续运行会复用本地缓存。

> [!TIP]
> 更换视频或切换 Whisper 配置前，可清理默认缓存目录 `~/.cache/transub`，或直接通过 `--work-dir` 指向临时位置以避免旧缓存干扰。
如需调整导出目录，可在配置中设置 `pipeline.output_dir`；缓存位置可通过 `--work-dir` 指定。

## 配置总览

运行时配置存放于 `transub.conf`（TOML），主要包含：

- `[whisper]`：faster-whisper 模型尺寸、设备、源语言和时间戳选项。
- `[llm]`：翻译模型、批大小、温度、重试策略等。
- `[pipeline]`：输出格式、行长限制、时间轴修正、标点与空格控制。

示例：

```toml
[pipeline]
output_format = "srt"
translation_max_chars_per_line = 26
translation_min_chars_per_line = 16
normalize_cjk_spacing = true
timing_offset_seconds = 0.05
```

执行 `transub configure` 可进入交互式编辑，或直接修改文件。配置文件属于用户环境，不建议提交至版本库。

## 常用命令速查

```bash
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub  # 覆盖默认缓存目录（默认使用 ~/.cache/transub）
transub prepare-model                 # 下载/初始化当前配置的本地 ASR 模型
transub show-config
transub init --config ./transub.conf   # 重新运行初始化向导
transub configure                      # 编辑配置（0 保存，Q 放弃）
transub run demo.mp4 --transcribe-only # 仅输出原始转写结果
transub run demo.mp4 -T               # 使用短参数启用仅转写
transub --version                     # 查看当前安装的版本号
```

默认缓存目录为 `~/.cache/transub`，其中存放音频、分段 JSON、翻译进度与流水线状态；如执行中断，重新运行即可继续。需要时可使用 `--work-dir` 指定自定义缓存路径。

## 开发者指南

如果希望参与贡献，可按以下步骤搭建本地环境。

### 桌面 GUI

桌面前端是 `desktop/` 下的 **Electron + React + TypeScript** 应用，提供可视化的流水线配置、凭据管理、转录/翻译运行和字幕预览界面。

启动开发环境：

```bash
cd desktop
npm install
npm run electron:dev
```

`npm run electron:dev` 会启动真正的 Electron 应用，可测试原生文件选择器和 Python 后端桥接。`npm run dev` 只是浏览器预览，不能测试这些桌面能力。

Electron 会自动启动本地 FastAPI 后端。默认优先使用 `localhost:18789`，如果端口被占用会自动尝试附近端口。运行页只展示面向用户的阶段进度和输出路径；详细后端日志会保存到 `~/.cache/transub/logs/`，用于排查问题。

后端已经加入 provider 级别的凭据管理：

- 默认 auth 文件位于 `~/.transub/auth.toml`；
- 可用 `TRANSUB_AUTH` 覆盖 auth 文件路径；
- 环境变量中的 key 优先级高于 auth 文件；
- 不要提交 auth 文件，也不要在日志中打印 API key。

### 从源码安装

1. **克隆仓库**
   ```bash
   git clone https://github.com/PiktCai/transub.git
   cd transub
   ```
2. **创建并激活虚拟环境**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **可编辑方式安装并拉取开发依赖**
   ```bash
   uv sync --extra dev
   ```
4. **准备本地转写模型**
   ```bash
   uv sync
   uv run transub prepare-model
   ```

### 运行测试

```bash
python -m unittest
```

性能和演示测试默认跳过，避免日常回归过慢。如需运行：

```bash
TRANSUB_RUN_PERF_TESTS=1 python -m unittest transub.test_concurrent_performance transub.test_retry_performance transub.test_concurrent_demo
```

### 代码结构

- 核心代码位于 `transub/`（`cli.py`、`config.py`、`transcribe.py`、`translate.py`、`subtitles.py` 等）。
- 新增功能请在模块旁添加 `test_*.py` 单元测试（例如 `transub/test_subtitles.py`）。
- 命令行输出请复用 Rich 控制台工具与 `transub.logger.setup_logging`。

## 目录结构

```
transub/
├── audio.py           # ffmpeg 音频提取
├── auth.py            # Provider 凭据管理
├── batch.py           # 批量处理
├── cache.py           # API 响应缓存
├── cli.py             # Typer 命令入口
├── concurrent_translate.py  # 并发翻译
├── config.py          # Pydantic 配置模型
├── free_translate.py  # 免费翻译后端 (Bing/Google)
├── logger.py          # 日志配置
├── optimize.py        # LLM 字幕优化
├── segmentation.py    # NLP 智能断句
├── smart_retry.py     # 智能重试逻辑
├── state.py           # 流水线状态持久化
├── subtitles.py       # 字幕结构与排版策略
├── transcribe.py      # faster-whisper 转写
├── translate.py       # LLM 翻译批处理
└── test_*.py          # 单元测试
```

## 许可协议

项目主要用于个人学习与研究，目前不接受外部贡献；如需自定义请自行 fork。  
Transub 基于 [MIT License](LICENSE) 开源发布。
