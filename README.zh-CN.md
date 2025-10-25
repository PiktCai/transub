# Transub 使用指南

[English README](README.md)

Transub 是一个基于 Typer 的命令行工具，可将英文视频自动转换成高质量的中文字幕。它集成了音频提取、Whisper 转写和 LLM 翻译，最终生成经过排版优化的字幕文件。

## 功能亮点

- **一键处理流程**：执行 `transub run <视频文件>` 即可完成音频提取 → 转写 → 翻译 → 导出。
- **多种后端支持**：本地 Whisper、`mlx-whisper`、`whisper.cpp` 以及兼容 OpenAI 的 API。
- **稳定的翻译输出**：批量 JSON 约束、自动重试、可调节的译文分段长度。
- **字幕排版优化**：中文断句、最小/最大行长限制、时间轴偏移、自动在中英文字符之间补空格。
- **断点续跑**：缓存目录 `.transub/` 保存执行状态，支持在失败或中断后继续。

## 快速开始

### 环境依赖

- Python 3.10+
- 安装 `ffmpeg`
- 具备翻译模型 API（默认 OpenRouter / DeepSeek）
- 视 Whisper 后端选择安装 `mlx-whisper`、`openai-whisper` 或 `whisper.cpp`

### 安装

```bash
git clone https://github.com/PiktCai/transub.git
cd transub
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 首次运行

```bash
transub init           # 交互式生成 transub.conf
transub run video.mp4  # 自动生成中/英文字幕
```

生成的文件默认位于 `./output/`，包括：

- `video.zh_cn.srt`：中文翻译字幕，自动处理中英文间距。
- `video.en.srt`：英文转写字幕（可配置关闭）。

## 配置说明

`transub.conf` 使用 TOML 格式：

- `[whisper]`：选择后端（`local` / `mlx` / `cpp` / `api`）、模型、设备、额外参数。
- `[llm]`：翻译模型、批大小、温度、重试策略等。
- `[pipeline]`：输出格式、最大/最小行长、时间轴修正、标点剔除、CJK 空格处理等。

示例：

```toml
[pipeline]
output_format = "srt"
translation_max_chars_per_line = 26
translation_min_chars_per_line = 16
normalize_cjk_spacing = true
timing_offset_seconds = 0.05
```

执行 `transub configure` 可在终端中交互式修改配置项。

## 常用命令

```bash
transub run demo.mp4 --config ~/transub.conf --work-dir /tmp/transub
transub show_config
transub init --config ./transub.conf     # 重新初始化
transub configure                        # 配置编辑器
```

缓存目录 `.transub/` 存储音频、分段 JSON 以及翻译进度；流程成功后会自动清理，如需保留可在中断时选择“否”。

## 开发者指南

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest
```

- 核心模块位于 `transub/`，对应功能详见代码注释。
- 修复/新增功能时请补充单元测试（参考 `transub/test_subtitles.py`）。
- CLI 输出统一使用 Rich；日志通过 `transub.logger` 记录。

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

本项目主要用于个人学习参考，目前不接受外部贡献；如需修改请自行 fork。  
项目基于 [MIT License](LICENSE) 开源发布。
