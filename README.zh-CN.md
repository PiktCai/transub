# Transub 使用指南

[English README](https://github.com/PiktCai/transub/blob/main/README.md)

Transub 通过 Typer 命令行，**提取**视频字幕并加以**翻译**：使用 `ffmpeg` 抽取音频，借助 Whisper 完成转写，并由 LLM 进行翻译，生成可直接使用的字幕文件。

## 目录

- [概览](#概览)
- [功能亮点](#功能亮点)
- [安装](#安装)
  - [1. 基础依赖](#1-基础依赖)
  - [2. 安装 Transub](#2-安装-transub)
  - [3. 安装 Whisper 后端](#3-安装-whisper-后端)
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
2. 通过 Whisper（本地、mlx、whisper.cpp 或 API）生成语音转写。
3. 将字幕分批发送给 LLM，使用 JSON 约束确保输出稳定。
4. 输出 `.srt` / `.vtt` 文件，控制行长、断句和时间轴偏移。

所有中间状态都会写入工作目录，意外中断后可以就地恢复。

## 功能亮点

- **一键处理**：`transub run <视频文件>` 即可完成提取 → 转写 → 翻译 → 导出。
- **多种 Whisper 后端**：支持本地 Whisper、`mlx-whisper`、`whisper.cpp` 以及兼容 OpenAI 的 API。
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

### 3. 安装 Whisper 后端（可选）

`transub` 支持多种 Whisper 实现，根据需要选择：

- **云端 API（推荐快速开始）：**
  - 使用 OpenAI Whisper API 或兼容端点
  - 无需本地安装任何依赖
  - 设置 `OPENAI_API_KEY` 环境变量
  - 配置时选择 `backend = "api"`

- **本地后端（离线使用或自定义模型）：**
  - **常规使用（本地 CPU/GPU）**
    ```bash
    pip install openai-whisper
    ```
  
  - **Apple Silicon（macOS）**
    ```bash
    pip install mlx-whisper
    ```
  
  - **`whisper.cpp`**
    参考 [whisper.cpp 官方安装说明](https://github.com/ggerganov/whisper.cpp)，编译 `main` 可执行文件并加入 `PATH`。

### 4. 初始化 Transub

运行交互式向导生成配置文件：

```bash
transub init
```

向导会引导你选择 Whisper 后端、模型以及翻译所需的 LLM 提供方。

**API Key 说明：** 如果同时使用 OpenAI 的转录服务（Whisper API）和翻译服务（GPT 模型），默认共用同一个 `OPENAI_API_KEY` 环境变量。如需为不同服务使用不同的 API key，可在配置文件中分别自定义各服务的 `api_key_env` 字段。

### 5. 运行流水线

```bash
transub run /path/to/video.mp4
```

生成的字幕默认保存在原视频所在目录，支持 `.srt` 与 `.vtt`。如仅需原始转写，可在命令中追加 `--transcribe-only`。

> [!TIP]
> 更换视频或切换 Whisper 配置前，可清理默认缓存目录 `~/.cache/transub`，或直接通过 `--work-dir` 指向临时位置以避免旧缓存干扰。
如需调整导出目录，可在配置中设置 `pipeline.output_dir`；缓存位置可通过 `--work-dir` 指定。

## 配置总览

运行时配置存放于 `transub.conf`（TOML），主要包含：

- `[whisper]`：后端类型、模型、设备及额外参数。
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
transub show_config
transub init --config ./transub.conf   # 重新运行初始化向导
transub configure                      # 编辑配置（0 保存，Q 放弃）
transub run demo.mp4 --transcribe-only # 仅输出原始转写结果
transub run demo.mp4 -T               # 使用短参数启用仅转写
transub --version                     # 查看当前安装的版本号
```

默认缓存目录为 `~/.cache/transub`，其中存放音频、分段 JSON、翻译进度与流水线状态；如执行中断，重新运行即可继续。需要时可使用 `--work-dir` 指定自定义缓存路径。

## 开发者指南

如果希望参与贡献，可按以下步骤搭建本地环境。

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
   pip install -e ".[dev]"
   ```
4. **安装测试用 Whisper 后端**
   ```bash
   pip install openai-whisper
   ```

### 运行测试

```bash
python -m unittest
```

### 代码结构

- 核心代码位于 `transub/`（`cli.py`、`config.py`、`transcribe.py`、`translate.py`、`subtitles.py` 等）。
- 新增功能请在模块旁添加 `test_*.py` 单元测试（例如 `transub/test_subtitles.py`）。
- 命令行输出请复用 Rich 控制台工具与 `transub.logger.setup_logging`。

## 目录结构

```
transub/
├── audio.py           # ffmpeg 音频提取
├── cli.py             # Typer 命令入口
├── config.py          # Pydantic 配置模型
├── subtitles.py       # 字幕结构与排版策略
├── transcribe.py      # Whisper 后端适配
├── translate.py       # LLM 翻译批处理
└── test_subtitles.py  # 单元测试
```

## 许可协议

项目主要用于个人学习与研究，目前不接受外部贡献；如需自定义请自行 fork。  
Transub 基于 [MIT License](LICENSE) 开源发布。
