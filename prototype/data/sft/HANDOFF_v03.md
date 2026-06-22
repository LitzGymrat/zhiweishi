# SFT v0.3 接手说明

## 用户当前目标（不可改动）

构建一套约 1,000 条的精选 SFT 数据；总量和任务配比必须严格为：

| 任务 | 总数 | 占比 |
|---|---:|---:|
| `fault_diagnosis` 故障排查卡 | 650 | 65% |
| `case_review` 案例复盘与沉淀 | 180 | 18% |
| `training_support` 培训模式 | 170 | 17% |
| 合计 | 1000 | 100% |

切分目标：`train=850`、`development=75`、`test=75`。`test` 不得参与训练或调参。

无检索规则已经由用户最新确认：**允许**给低风险行业通用建议，但必须先声明没有内部检索命中；明确行业常识不是知识库证据、不是当前根因结论，且不得给拆检、更换、调参等高风险操作。

## 已完成、可复用的资产

1. `data/sft/runtime_v3/`：上一版 1,500 条（1200/150/150）基线数据，已生成并通过格式、证据、日期和切分泄漏检查。它是审计基线，**不要和 v0.3 混训**。
2. `src/llm_client.py`：线上 system/user 构造已经抽成 `build_runtime_system_prompt` 与 `build_runtime_user_message`。任何新 SFT 生成器必须调用这两个函数，不能手写近似 prompt。
3. `src/llm_client.py` 已支持完整/不完整回写的运行时说明：完整回写需有来源标记、案例编号、处理人、回写时间、实际处理步骤和最终结论；缺字段只能作为疑似/不完整参考。
4. `scripts/validate_sft_dataset.py` 已支持 `train`、`development`、`test` split，以及 `synthetic_runtime_v3` source mode。
5. `scripts/export_sft_messages.py` 已可导出 `train`、`development`、`test`，不再只写 train/validation。
6. `data/sft/runtime_v3_refined/README.md`：v0.3 数据规范草案。
7. `data/sft/runtime_v3_refined/scenario_matrix.json`：v0.3 的精确场景配额矩阵。
8. `scripts/generate_runtime_sft_refined.py`：v0.3 生成器草稿。它尚未调用教师 API、尚未完成试标，不应视为已验收。

## v0.3 配额核对

`runtime_v3_refined/scenario_matrix.json` 设计为：

| split | fault | case_review | training | total |
|---|---:|---:|---:|---:|
| train | 550 | 153 | 147 | 850 |
| development | 50 | 14 | 11 | 75 |
| test | 50 | 13 | 12 | 75 |
| 合计 | 650 | 180 | 170 | 1000 |

接手后第一步必须用脚本重新求和校验以上数字；任何偏差都不要启动大规模教师生成。

## v0.3 场景原则

### 故障排查（650）

- 完整证据 + 人工回写：150
- 仅 SOP：70
- 无检索：80，其中 strict 20、行业常识低风险建议 60
- 单回写：90
- 多回写一致：50
- 多回写冲突：50
- 截断/字段缺失：50
- 错设备：50
- 同设备错症状/不完全适用：30
- 过期 SOP/新旧文档冲突：30

### 案例复盘（180）

完整回写 50；缺实际步骤 25；缺最终结论 25；经验总结不可复用 20；步骤/结论冲突 20；历史案例冲突 20；需更新 SOP 20。

### 培训（170）

基于 SOP 40；历史案例 35；冲突案例 25；截断材料 20；新员工基础培训 25；复盘后专项培训 25。

## 生成器接手前必须检查的事项

1. 编译：`uv run python -m compileall -q src scripts`
2. 预览矩阵，不调用 API：

```powershell
uv run python scripts/generate_runtime_sft_refined.py `
  --split train `
  --output data/sft/runtime_v3_refined/pilot.jsonl `
  --one-per-family
```

3. 先做每个情景族 1 条的教师试标，启用 thinking；必须通过自定义语义门禁后，才允许全量。
4. 为失败样本实现“按原 scenario 重试/合并”的修复路径；不能因为门禁失败就缩减某个 hard negative 的配额。
5. 全量后必须校验：JSON schema、线上 system/user 字符串完全一致、日期不晚于 cutoff、无 `□/☐`、train/dev/test 的设备与案例编号无交集、三类任务总量精确为 650/180/170。

## 已知风险与建议

- 当前 refined 生成器是新草稿，未试标；要先检查它的场景构造和质量门禁是否与本文件一致。
- 案例复盘的“字段缺失”场景可以训练 `missing_information`，但当前前端保存流程会先校验必填字段。因此这类 SFT 样本对应的是待审核/导入前复盘能力；若希望线上入口可触发，需后续单独设计“保存前预审”产品流程。
- 所有教师调用应开启 thinking，但训练数据中只能保存最终 assistant JSON，不能保存 reasoning。
- 不要删除 `runtime_v3/`、`runtime_v2/` 或 `synthetic/`；它们是审计和回归基线。

## 推荐最终训练入口

完成 v0.3 后，训练只读：

```text
data/sft/runtime_v3_refined/export/train.jsonl
data/sft/runtime_v3_refined/export/development.jsonl
```

冻结评测只读：

```text
data/sft/runtime_v3_refined/export/test.jsonl
```
