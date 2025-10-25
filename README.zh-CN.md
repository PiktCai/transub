# Transub 使用指南

[English README](README.md)

Transub 通过 Typer 命令行，**提取**视频字幕并加以**翻译**：使用 `ffmpeg` 抽取音频，借助 Whisper 完成转写，并由 LLM 进行翻译，生成可直接使用的字幕文件。

## 目录

- [概览](#概览)
- [功能亮点](#功能亮点)
- [快速开始](#快速开始)
  - [准备工作](#准备工作)
  - [Windows（PowerShell，本地 Whisper）](#windowspowershell本地-whisper)
  - [macOS（Apple Silicon，mlx-whisper）](#macosapple-siliconmlx-whisper)
  - [Linux（Bash，本地 Whisper）](#linuxbash本地-whisper)
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
- **断点续跑**：缓存目录 `.transub/` 可保存音频、分段和翻译进度，避免重复计算。

## 快速开始

### 准备工作

- 安装 **Python 3.10+**。
- 确保系统路径中可运行 **`ffmpeg`**。
- 预先准备 **LLM API 凭证**（默认配置对接 OpenAI 兼容接口，可在 `transub.conf` 中指定 `api_key_env` 环境变量）。
- 预留足够磁盘空间，用于 `./.transub/` 下的临时音频与转写缓存。

以下步骤针对不同平台提供推荐的 Whisper 后端与安装指引。

> 提示：如果只需要原始转写，可在运行命令时追加 `--transcribe-only` 跳过翻译阶段。
> 提示：仓库示例延续了英↔中的出发点，但只要 Whisper / LLM 覆盖对应语言，就能自由组合。
> 提示：执行 `transub init` 时，可输入 `back`（或“上一步”）返回上一题重新填写。

### Windows（PowerShell，本地 Whisper）

1. **安装依赖**
   - [Python 3.10+](https://www.python.org/downloads/windows/)
   - `ffmpeg`：`winget install Gyan.FFmpeg` 或 `choco install ffmpeg`
2. **克隆仓库并创建虚拟环境**
   ```powershell
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .
   pip install openai-whisper
   ```
   如需使用 GPU，请额外安装匹配的 PyTorch CUDA 轮子。
3. **初始化配置**
   ```powershell
   transub init
   ```
   在 `[whisper]` 区块保持 `backend = "local"`，并选取 `small`、`medium` 等模型。
4. **执行流水线**
   ```powershell
   transub run .\video.mp4 --work-dir .\.transub
   ```
   生成字幕输出到 `.\output\`，文件名会附加语言后缀（例如 `video.en.srt`、`video.ja.srt`）。

### macOS（Apple Silicon，mlx-whisper）

1. **安装依赖**
   - Python 3.10+（可通过 `brew install python@3.11`）
   - `ffmpeg`：`brew install ffmpeg`
2. **克隆并创建虚拟环境**
   ```bash
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install mlx-whisper
   ```
   `mlx-whisper` 基于 MLX，可在 Apple Silicon GPU/NPU 上高效运行。
3. **初始化配置**
   ```bash
   transub init
   ```
   将 `backend` 设置为 `"mlx"`，选择模型（如 `mlx-community/whisper-small.en-mlx`），必要时填写 `mlx_model_dir`。
4. **执行流水线**
   ```bash
   transub run ./video.mp4 --work-dir ./.transub
   ```
   字幕文件默认保存在 `./output/`，格式为 `.srt` 或 `.vtt`。

### Linux（Bash，本地 Whisper）

1. **安装依赖**
   ```bash
   sudo apt update && sudo apt install ffmpeg python3 python3-venv  # Debian/Ubuntu
   # Arch 用户：sudo pacman -S ffmpeg python python-virtualenv
   ```
2. **克隆仓库并创建虚拟环境**
   ```bash
   git clone https://github.com/PiktCai/transub.git
   cd transub
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install openai-whisper
   ```
   如需 GPU 加速，可参考 [PyTorch 官网](https://pytorch.org/) 安装适配的 `torch`、`torchvision`、`torchaudio`。
3. **初始化配置**
   ```bash
   transub init
   ```
   向导中保持 `backend = "local"`，选择符合硬件条件的 Whisper 模型。
4. **执行流水线**
   ```bash
   transub run ./video.mp4 --work-dir ./.transub
   ```
   生成字幕位于 `./output/` 目录。

> 提示：若更换视频或 Whisper 配置，为避免旧缓存干扰，可在重新运行前清理 `.transub/` 目录。

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
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub
transub show_config
transub init --config ./transub.conf   # 重新运行初始化向导
transub configure                      # 编辑配置（0 保存，Q 放弃）
transub run demo.mp4 --transcribe-only # 仅输出原始转写结果
transub run demo.mp4 -T               # 使用短参数启用仅转写
```

缓存目录 `.transub/` 会保存音频、分段 JSON、翻译进度与流水线状态；如执行中断，重新运行即可继续。

## 开发者指南

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest
```

- 核心代码位于 `transub/`（`cli.py`、`config.py`、`transcribe.py`、`translate.py`、`subtitles.py` 等）。
- 新增功能请在相关目录旁添加 `test_*.py` 单元测试（如 `transub/test_subtitles.py`）。
- 统一使用 Rich 控制台与 `transub.logger.setup_logging` 输出日志。

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
