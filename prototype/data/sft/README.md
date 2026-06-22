# 智维师 SFT 数据

这个目录准备的是本地 4B 主模型的监督微调（SFT）数据，而不是把维保知识从 RAG 中搬进模型参数。

## 训练目标

模型要学习的是稳定、保守地完成三种结构化任务：

1. `fault_diagnosis`：依据已检索片段生成排障卡。
2. `training_support`：把已确认的排障结论转成带教材料。
3. `case_review`：把人工填写的处置记录整理成待审核的案例复盘。

设备事实、阈值、工单和案例仍必须由运行时 RAG 提供。模型不得把训练行中的事实当成脱离证据也能直接回答的“常识”。

## 文件说明

- `runtime_v3_refined/`：当前唯一可训练入口。它固定复用线上 prompt、输入 JSON 和检索片段渲染格式；v0.4 已将无检索时的行业常识策略与线上 prompt 对齐，并通过精确配额和防泄漏终检。
- `runtime_v3/`：历史版本基线数据，留作审计。
- `runtime_v2/`：格式同构的基线审计集，不得混入 v3 训练导出文件。
- `seed_examples.jsonl`：带样本 ID、任务、证据文件和审核状态的主数据集。
- `schema.md`：与当前应用严格对齐的输出契约和标注规则。
- `source_manifest.json`：种子样本使用的本地证据文件清单。
- `../../scripts/validate_sft_dataset.py`：检查 JSON、字段、角色顺序、来源和重复 ID。
- `../../scripts/export_sft_messages.py`：导出大多数训练框架可直接读取的纯 `messages` JSONL。
- `../../scripts/generate_teacher_drafts.py`：调用教师模型生成独立的待审核草稿队列；绝不直接写入正式训练集。
- `../../scripts/generate_runtime_sft.py`：v2 基线生成器，仅保留审计用途。
- `../../scripts/generate_runtime_sft_refined.py`：精细场景生成器，支持 DMXAPI 并发和 `--retry-rejected` 原 scenario 重试机制。
- `../../scripts/refresh_runtime_sft_prompts.py`：运行时 prompt 变更后的样本重渲染与定向重标队列生成器。
- `../../scripts/evaluate_fault_diagnosis.py`：冻结 test 中辅助排查任务的 base / finetuned / DeepSeek-V4-Flash / DeepSeek-V4-Pro 四模型评测与 Gemini judge 链路。
- `../../scripts/generate_runtime_sft_v3.py`：按 `runtime_v3/scenario_matrix.json` 构造证据约束数据；仅作为历史审计保留。
- `synthetic/`：第一代自由合成方案的审计留档，不得用于本轮训练。

## 当前状态

当前 9 条种子样本都标为 `evidence_checked_pending_sme`：内容已逐项追溯到仓库内的说明书、SOP、日志、维修记录或案例卡，但尚未经过真实设备负责人/维保工程师签字确认。它们可用于验证训练流程、格式稳定性和基线微调实验；要进入正式生产训练集，状态必须升为 `sme_approved`。

种子集故意覆盖两类高价值反例：

1. 资料充分时可以给出有条件的排查假设和步骤。
2. 只有说明书/SOP、没有现场确证时，必须说明不确定性，不能臆造具体故障件。

种子集没有“前台案例回写（人工确认后归档）”的真实持久化样本，因此其 `matched_writeback_case_note` 均为空；它只保留为证据追溯和小样本回归资料。该缺口已由 `runtime_v2/` 的完整回写模板、命中/未命中/多案例/冲突情景覆盖，但这些仍是合成训练样本，不能替代后续真实现场评测。

## 当前训练入口：runtime_v3_refined (v0.4 精选数据)

根据项目最新确认，训练数据应统一使用 `runtime_v3_refined` 以满足精确配额、证据边界和无检索行业常识策略。

训练工具应读取：

```text
data/sft/runtime_v3_refined/export/train.jsonl
data/sft/runtime_v3_refined/export/development.jsonl
```

评测与终检应读取：

```text
data/sft/runtime_v3_refined/export/test.jsonl
```

如需重试/重建，请参考 `HANDOFF_v03.md`，使用 `validate_runtime_sft_refined.py` 进行终检：

```powershell
# 终检全量样本
uv run python scripts/validate_runtime_sft_refined.py `
  --input data/sft/runtime_v3_refined/v04/train_complete.jsonl `
  --input data/sft/runtime_v3_refined/v04/development.jsonl `
  --input data/sft/runtime_v3_refined/v04/test.jsonl

# 重新导出消息到 export/ 目录
uv run python scripts/export_sft_messages.py `
  --input data/sft/runtime_v3_refined/v04/train_complete.jsonl `
  --input data/sft/runtime_v3_refined/v04/development.jsonl `
  --input data/sft/runtime_v3_refined/v04/test.jsonl `
  --status teacher_generated `
  --output-dir data/sft/runtime_v3_refined/export
```

## 使用顺序

```powershell
cd prototype
uv run python scripts/validate_sft_dataset.py
uv run python scripts/export_sft_messages.py --status evidence_checked_pending_sme
uv run python scripts/generate_teacher_drafts.py --max-seeds 3 --variants-per-seed 2
```

导出文件位于 `data/sft/export/`，按 `train` 与 `validation` 划分。现有验证集与训练集仍共享当前两类设备资料，因此它只能检查格式、保守性和任务泛化，**不能**宣称为未见设备上的真实泛化评测。后续应使用新的设备、SOP 和工单建立独立测试集，且绝不参与 SFT。

教师草稿默认只预览。要实际调用 DeepSeek API，必须显式执行：

```powershell
uv run python scripts/generate_teacher_drafts.py --execute --confirm-external-transfer --model deepseek-v4-pro --max-seeds 1 --variants-per-seed 1
```

此命令会将当前选中的证据片段发送到外部教师 API；产物只会写入 `data/sft/teacher_drafts/`，不会进入 `seed_examples.jsonl`。

教师脚本默认模型 ID 为 `deepseek-v4-pro`。它使用 JSON Output，并显式关闭思考模式；思考内容不会保存为 SFT 语料。当前项目 `.env` 里的应用默认模型仍可能是旧的 `deepseek-chat`，教师脚本通过 `--model deepseek-v4-pro` 独立指定，不会改动原型线上推理配置。

## v0.3 教师提供方：DeepSeek 与 DMXAPI Gemini

`scripts/generate_runtime_sft_refined.py` 保留原有 DeepSeek 链路，并新增独立的 DMXAPI OpenAI-compatible 链路。它只用于构造 v0.3 SFT 数据，不会改变原型运行时的 `deepseek_*` 配置。

在 `.env` 中配置（密钥不要提交）：

```text
dmx_api_key=你的DMXAPI密钥
# 可选覆盖；默认即为下列值
dmx_base_url=https://www.dmxapi.cn/v1
dmx_model=gemini-3-flash-preview
```

DMXAPI 默认以 1,000 并发运行，HTTP 连接池也同步放大到该数量；需要收敛时可显式传 `--workers`。先用每个情景族一条的真实教师试标，再审核生成质量：

```powershell
uv run python scripts/generate_runtime_sft_refined.py `
  --teacher-provider dmxapi `
  --split train `
  --output data/sft/runtime_v3_refined/gemini_pilot.jsonl `
  --one-per-family `
  --execute
```

该链路使用 DMXAPI 已验证的 `https://www.dmxapi.cn/v1/chat/completions`、`gemini-3-flash-preview` 与 JSON Output；虽允许教师模型内部推理，训练文件只保存最终 assistant JSON。若要切回原链路，省略 `--teacher-provider dmxapi` 即可，DeepSeek 仍默认 `deepseek-v4-pro`、并发 1。

如果目标是训练模型把 RAG 文本转换成结构化输出，而非复述当前知识库事实，使用合成蒸馏生成器：

```powershell
uv run python scripts/generate_synthetic_sft.py --execute --model deepseek-v4-pro --task all --samples-per-task 3
```

该命令只发送任务与 schema，不发送 `data/documents/` 内的本地文本；但其自由生成输入外壳与覆盖范围不足，仅作为历史审计保留，不能替代 `runtime_v2/`。

## 人工审核最小要求

每个样本入正式训练集前，审核人至少确认：

1. 每个具体阈值、故障征兆和动作都能在 `source_files` 中找到依据。
2. “可能原因”没有被写成现场已确认的根因。
3. 停机、复机、拆检和上报边界符合实际安全制度。
4. 未命中“前台案例回写（人工确认后归档）”时，`matched_writeback_case_note` 必须为空字符串。
5. `uncertainty_note` 不得为空，也不得用无意义套话掩盖证据缺口。
