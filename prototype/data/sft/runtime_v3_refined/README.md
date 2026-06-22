# SFT 数据调整方案 v0.4：精选证据边界集

本目录是对 `runtime_v3/` 的精选重建版：总量收敛到 1,000 条，同时保留线上格式同构、教师 thinking 标注和严格 JSON schema。v0.4 修复了无检索 prompt 与行业常识建议之间的矛盾。

## 切分与任务配比

| split | 数量 | 用途 |
|---|---:|---|
| train | 850 | SFT 训练 |
| development | 75 | 调参和回归观察 |
| test | 75 | 冻结评测，不参与训练或调参 |

总任务配比：故障排查 650 条、案例复盘 180 条、培训模式 170 条。

## 无检索规则

无检索样本都必须先声明未命中内部 SOP、维护记录或历史案例；随后允许给出显式标注为行业常识/通用经验的低风险建议。行业常识不是知识库证据，不能作为当前根因结论，也不能包含拆检、更换、调参等高风险动作。

为保留原始配额的审计可比性，矩阵中仍保留 `fault_no_retrieval_strict` 名称；它现在表示**更保守的行业常识建议**（仅记录、目视核对和状态确认），不再表示禁止行业常识。

## 训练外元数据

每条记录包含但不导出给模型的：

```json
{
  "scenario_type": "no_retrieval_with_common_sense",
  "evidence_level": 0,
  "retrieval_quality": "none",
  "writeback_match_type": "none",
  "should_use_common_sense": true,
  "should_abstain_from_root_cause": true
}
```

## 当前训练入口

训练仅使用：

```text
export/train.jsonl
export/development.jsonl
```

`export/test.jsonl` 只能用于最终评测。

`v04/` 保存本次迁移的带审计原始 JSONL；根目录 `export/` 镜像 v0.4 的最终纯 messages 训练入口。`runtime_v3/` 保留为 1,500 条证据门禁基线；本目录是面向 4B 小模型的精选训练集。

## 生成与验收顺序

先预览 24 个情景族；不带 `--execute` 时不会调用教师 API，也不会写出数据：

```powershell
uv run python scripts/generate_runtime_sft_refined.py `
  --split train `
  --output data/sft/runtime_v3_refined/pilot.jsonl `
  --one-per-family
```

教师试标通过人工语义复核后，再分别生成三份完整原始集。若某些样本四次修复后仍未通过门禁，生成器会把**原始 scenario**写入同名的 `_rejected.jsonl`；重试时必须使用该文件，不能重新抽样或缩减 hard negative 配额：

默认教师提供方仍是 DeepSeek。若改用已在 `.env` 中配置的 DMXAPI Gemini，则加 `--teacher-provider dmxapi`；该路径默认 `gemini-3-flash-preview` 与 1,000 并发，并把 HTTP 连接池同步设为 1,000。先以每个情景族一条的实际试标检查质量：

```powershell
uv run python scripts/generate_runtime_sft_refined.py `
  --teacher-provider dmxapi `
  --split train `
  --output data/sft/runtime_v3_refined/gemini_pilot.jsonl `
  --one-per-family `
  --execute
```

使用 `--workers 100` 等参数可主动降并发；要回到 DeepSeek，只需省略 `--teacher-provider dmxapi`。

```powershell
uv run python scripts/generate_runtime_sft_refined.py `
  --split train `
  --retry-rejected data/sft/runtime_v3_refined/train_rejected.jsonl `
  --merge-with data/sft/runtime_v3_refined/train.jsonl `
  --output data/sft/runtime_v3_refined/train_recovered.jsonl `
  --execute --overwrite
```

全量完成（或重试合并完成）后，必须对三个原始 JSONL 一起终检。该命令会检查运行时 prompt 同构、schema、日期 cutoff、无 `□/☐`、精确场景/任务配额，以及跨 split 的设备和案例编号泄漏：

```powershell
uv run python scripts/validate_runtime_sft_refined.py `
  --input data/sft/runtime_v3_refined/train.jsonl `
  --input data/sft/runtime_v3_refined/development.jsonl `
  --input data/sft/runtime_v3_refined/test.jsonl
```

如使用了上面的重试示例，终检和导出时应把第一个 `train.jsonl` 替换为 `train_recovered.jsonl`。

只有终检通过的三份原始集才能交给 `scripts/export_sft_messages.py` 导出。训练只读 `export/train.jsonl` 与 `export/development.jsonl`；`export/test.jsonl` 始终冻结。

## v0.4 无检索策略迁移

修改 `src/llm_client.py` 的运行时 system/user 模板后，不能只重写无检索行：所有样本的 system prompt 都需要同步重渲染，否则 SFT 会与线上模板漂移。迁移脚本会保留其余样本的 assistant JSON，只把 80 条无检索样本按原 scenario 送回教师模型重标：

```powershell
uv run python scripts/refresh_runtime_sft_prompts.py `
  --input data/sft/runtime_v3_refined/train_complete.jsonl `
  --input data/sft/runtime_v3_refined/development_complete.jsonl `
  --input data/sft/runtime_v3_refined/test_complete.jsonl `
  --output-dir data/sft/runtime_v3_refined/v04_migration
```

迁移结果必须用 `scripts/validate_runtime_sft_refined.py` 对 v0.4 三个原始集联合终检；通过后再导出到本目录 `export/`。
