# 组员 Quick Start

这份说明给第一次拿到项目的组员使用。目标是：先把项目跑通，再开始分工开发。

## 1. 拉取项目

```bash
git clone <仓库地址>
cd <仓库目录>/overlap_asr_llm
```

如果已经 clone 过：

```bash
git pull
```

## 2. 确认 Python 版本

建议使用 Python 3.10 或更高版本：

```bash
python3 --version
```

本项目的 mock 模式只依赖 Python 标准库，不需要先安装大型模型。

如果使用 Anaconda，建议先创建独立环境：

```bash
conda create -n overlap-asr-llm python=3.10 -y
conda activate overlap-asr-llm
pip install -e .
pip install pytest
```

## 3. 先跑 mock 版本

mock 模式用于检查项目结构、代码流程、输出文件是否正常。

```bash
PYTHONPATH=src python3 -m overlap_asr_llm.cli run --config configs/experiment.json --mock
```

运行成功后，终端会看到类似输出：

```text
Wrote results to .../outputs
```

生成结果在：

```text
outputs/results.json
outputs/results.csv
outputs/run_summary.md
```

## 4. 跑 smoke test

```bash
python3 scripts/smoke_test.py
```

这个脚本用于快速确认项目最基本功能是否能跑通。

## 5. 项目结构怎么看

```text
configs/experiment.json         实验配置：样本、模型、输出路径
docs/                           项目文档、实验设计、讲解稿
src/overlap_asr_llm/            核心代码
tests/                          测试代码
scripts/                        辅助脚本
outputs/                        运行生成的结果
README.md                       项目总体说明
CONTRIBUTIONS.md                组员贡献记录
```

核心代码阅读顺序：

```text
config.py       读取配置
providers.py    模型封装，包括 ASR、分离、说话人标注、LLM 修正
pipelines.py    四条实验流水线
metrics.py      CER/WER 指标
io.py           写出 JSON、CSV、Markdown 结果
cli.py          命令行入口
```

更详细的代码讲解见：

```text
docs/CORE_CODE_LINE_BY_LINE_CN.md
```

## 6. 分工开发流程

不要直接在 `main` 分支上改代码。每个人先创建自己的分支：

```bash
git checkout -b feature/<你的名字或任务名>
```

例如：

```bash
git checkout -b feature/asr-baseline
git checkout -b feature/experiment-docs
git checkout -b feature/video-script
```

完成修改后：

```bash
git status
git add <你修改的文件>
git commit -m "Describe your change"
git push -u origin feature/<你的名字或任务名>
```

然后在 GitHub 上开 Pull Request，合并到 `main`。

## 7. 提交前检查

提交前建议至少跑：

```bash
PYTHONPATH=src python3 -m overlap_asr_llm.cli run --config configs/experiment.json --mock
python3 scripts/smoke_test.py
```

如果安装了 pytest，也可以跑：

```bash
PYTHONPATH=src pytest
```

## 8. 不要提交哪些文件

一般不要提交：

```text
__pycache__/
.pytest_cache/
.venv/
venv/
outputs/
*.zip
```

这些通常是本地缓存、运行结果、虚拟环境或最终打包文件，不适合放进协作仓库。

## 9. 真模型模式

mock 跑通之后，如果要使用真实模型，需要在有合适环境的机器上安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

然后修改：

```text
configs/experiment.json
```

可以配置的 provider 包括：

```text
ASR:        mock, whisper, whisper:<model-name>, funasr
Diarizer:   mock, speechbrain
Separator:  mock, sepformer, sepformer:<huggingface-model-id>
LLM:        mock
```

真实模型可能需要 GPU、较大下载量和额外依赖。如果只是协作写代码或文档，优先使用 mock 模式。

当前配置已经接入 5 个真实混合音频样本。人工听写参考文本见：

```text
docs/REFERENCE_TRANSCRIPTION_CN.md
```

## 10. 每个人要更新贡献记录

每位组员完成自己的部分后，记得更新：

```text
CONTRIBUTIONS.md
```

里面需要写清楚：

- 成员姓名
- 主要负责内容
- 证据，例如代码文件、文档、实验结果、视频部分
- 贡献比例

## 11. 常见问题

### 找不到 `overlap_asr_llm` 模块

如果看到类似：

```text
ModuleNotFoundError: No module named 'overlap_asr_llm'
```

说明没有设置 `PYTHONPATH`。请使用：

```bash
PYTHONPATH=src python3 -m overlap_asr_llm.cli run --config configs/experiment.json --mock
```

或者安装为可编辑包：

```bash
pip install -e .
```

### 真实模型导入失败

如果没有安装 Whisper、FunASR、SpeechBrain 等依赖，真实模型会失败。先用 `--mock` 跑通项目流程。

### 输出结果和别人不一样

先确认是否都使用了同一个配置文件：

```text
configs/experiment.json
```

如果有人改了模型、样本或 reference，结果会不一样。

## 最短运行命令

只想确认项目能跑时，用这一条：

```bash
PYTHONPATH=src python3 -m overlap_asr_llm.cli run --config configs/experiment.json --mock
```
