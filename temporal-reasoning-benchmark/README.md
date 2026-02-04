# Temporal Reasoning Benchmark (M1: DeepSeek-only)

本仓库用于系统评测 LLM 的时间推理能力，并支持可复现、可审计的端到端实验流程。当前阶段聚焦 M1：DeepSeek-only 闭环跑通。

## 3 分钟跑通（Quick Start）

1) 安装依赖

```bash
cd temporal-reasoning-benchmark
pip install -r requirements.txt
```

2) 配置密钥

- 复制 .env.example 为 .env，并填入你的 DeepSeek 密钥

```bash
cp .env.example .env   # Windows 可手动复制
# 编辑 .env 并填入：
# DEEPSEEK_API_KEY="your_deepseek_key"
```

3) 运行基线（50 条样本，SP-0shot）

```bash
python -m src.runner --config configs/baseline.yaml
```

4) 查看输出

- 预测与日志：`outputs/runs/<timestamp>_baseline_sp0_0shot_small/predictions.jsonl`
- 汇总表：`outputs/runs/<timestamp>_baseline_sp0_0shot_small/summary.csv`
- 审计信息：
  - `outputs/runs/<timestamp>_.../config_snapshot.yaml`（本次运行配置快照）
  - `outputs/runs/<timestamp>_.../run_metadata.json`（run_config_hash、git_commit 等）

若出现 401/404 等 API 调用异常，请确认 .env 的 DEEPSEEK_API_KEY 是否正确，且网络可访问 https://api.deepseek.com/v1。

## 配置与可复现

- 主配置：`configs/baseline.yaml`
  - 支持修改 prompt_mode: sp|cot，n_shots: 0|5
  - 支持 subset_size、seed、params（temperature/top_p/max_tokens/timeout）
  - 模型在 `configs/models.yaml` 中定义（当前仅启用 deepseek_chat）
- Prompt 模板：`configs/prompts.yaml` 指向 `prompts/*.txt`
- 数据：统一 schema 的 JSONL（见 `data/raw/sample.jsonl` 示例）

## 下一步（M2 预告）

- 通过 `scripts/prepare_tram.py` 将 TRAM 原始数据转换为统一 JSONL（M2-1：Arithmetic SAQ/MCQ）：`data/raw/tram_arithmetic_mcq.jsonl`
- 新增配置：`configs/pilot_tram_sp0.yaml`、`configs/pilot_tram_cot0.yaml` 并运行首轮 pilot（subset_size=100）

## TRAM Arithmetic SAQ/MCQ 转换（可复制命令）

> 目标：从 TRAM-Benchmark 的 `arithmetic_saq.csv` / `arithmetic_mcq.csv` 生成统一 schema JSONL：`data/raw/tram_arithmetic_mcq.jsonl`

### 1) Dry-run（探测列名与样例，不写文件）

```powershell
cd temporal-reasoning-benchmark
python scripts\prepare_tram.py --dry_run
```

或显式指定输入：

```powershell
python scripts\prepare_tram.py --input "..\TRAM-Benchmark\datasets\arithmetic\arithmetic_saq.csv" --dry_run
```

### 2) 正式转换（写 JSONL + audit）

```powershell
python scripts\prepare_tram.py --input "..\TRAM-Benchmark\datasets\arithmetic\arithmetic_saq.csv" --output data\raw\tram_arithmetic_subset.jsonl
```

### 3) 验证输出（schema 校验）

```powershell
python scripts\prepare_tram.py --input "..\TRAM-Benchmark\datasets\arithmetic\arithmetic_saq.csv" --output data\raw\tram_arithmetic_subset.jsonl --validate_out
```

若列名无法自动推断，可手动覆盖：

```powershell
python scripts\prepare_tram.py --input "..\TRAM-Benchmark\datasets\arithmetic\arithmetic_saq.csv" --question_col Question --answer_col Answer --dry_run
```

## 目录结构

```
configs/
  baseline.yaml
  models.yaml
  prompts.yaml
prompts/
  sp_0shot.txt
  sp_5shot.txt
  cot_0shot.txt
  cot_5shot.txt
  exemplars.jsonl
src/
  adapters/
  analysis.py
  io_utils.py
  normalize.py
  prompt_builder.py
  runner.py
scripts/
  prepare_tram.py
outputs/
  runs/
  tables/
```





