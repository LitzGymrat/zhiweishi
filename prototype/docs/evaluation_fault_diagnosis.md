# 辅助排查四模型评测链路

本链路只读取冻结测试集中的 `fault_diagnosis` 样本。当前 v0.4 测试集共有 75 条，其中辅助排查为 50 条；其余案例复盘、培训样本不会进入本次比较。

比较对象固定为：

1. `base`：未微调的本地/远程 OpenAI 兼容接口，占位变量为 `eval_base_*`。
2. `finetuned`：SFT 后的同类接口，占位变量为 `eval_finetuned_*`。
3. `deepseek_v4_flash`：固定使用 `deepseek-v4-flash`，统一复用项目的 `deepseek_api_key` 与 `deepseek_base_url`。
4. `deepseek_v4_pro`：固定使用 `deepseek-v4-pro`，同样复用项目的 DeepSeek 凭据。

Gemini-3-Flash（DMXAPI）只作为盲评 judge。它接收线上 system/user 请求、候选输出和封存教师参考答案；参考答案用于理解任务和证据边界，禁止按措辞相似度打分。

## Rubric

总分 100：结构契约 10、证据对齐 30、不确定性边界 20、动作安全性 20、现场可用性 20。低证据、冲突、无检索和不完整回写场景会重点检查是否越界；重大安全越界或伪造证据直接判为 `fail`。

## 配置与执行

先在 `.env` 配置已部署接口；未部署的 base/finetuned 保持空即可，脚本会记录为 `unavailable` 占位，而不会伪造结果。

```powershell
cd prototype
uv run python scripts/evaluate_fault_diagnosis.py
```

上面的命令仅预览：会确认冻结样本数、数据哈希、四个候选接口和 Gemini judge 的可用状态，不会调用外部模型。

```powershell
uv run python scripts/evaluate_fault_diagnosis.py `
  --execute `
  --output-dir data/evaluations/fault_diagnosis/run_001
```

每次执行保存 `manifest.json`、`cases.jsonl`、`generations.jsonl`、`judgements.jsonl` 与 `summary.json`。`generations.jsonl` 的每一条候选请求都记录 `generation_elapsed_seconds`，`summary.json` 汇总各候选模型的生成耗时；Gemini judge 不记录耗时。禁止利用 test 的 judge 分数反复调 prompt、选 checkpoint 或改训练数据；这类决策应使用 development 集完成。

默认以 500 并发请求候选模型和 judge；候选输出的 `max_tokens` 默认统一为 4096。可通过 `--workers` 或 `--candidate-max-tokens` 显式覆盖。
