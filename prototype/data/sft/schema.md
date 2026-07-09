# 结构化 SFT 契约（v1）

## 为什么暂时保持 v1

当前 Streamlit 页面和 `src/llm_client.py` 已经围绕三套扁平 JSON schema 渲染与兜底。首轮 SFT 应先把模型训练到稳定遵守这份运行时契约，而不是同时改模型、训练数据和前端。

运行时会从真实检索片段生成 `evidence_sources`，因此它不是模型输出字段；训练样本的 `source_files` 只用于人工审核与离线评测，不会喂给模型作为答案。

下一版可增加“每个断言对应哪些证据片段”的可视化字段，但必须先设计前端展示和人工复核方式，不能把嵌套引用结构临时塞进首轮数据。

## `fault_diagnosis`

输出键必须且只能为：

```json
{
  "summary": "字符串",
  "matched_writeback_case_note": "字符串；未命中时为空字符串",
  "evidence_observations": ["最多4条"],
  "possible_causes": ["最多4条候选原因"],
  "troubleshooting_steps": ["最多5条，按先易后难、先低风险后高风险排序"],
  "risk_notes": ["最多4条"],
  "escalation_advice": "字符串",
  "training_points": ["最多4条"],
  "uncertainty_note": "字符串，必须保留证据边界"
}
```

标注规则：

- `summary` 只能描述当前证据支持的初步判断；除非检索片段明确记录“检查/结果”，不得把候选原因写成已确认根因。
- `possible_causes` 是候选项。缺少现场验证时应使用“需排查”“可能”等措辞。
- 若只有手册/SOP，允许给出手册明确列出的检查方向，但必须在 `uncertainty_note` 说明没有现场测量或工单确证。
- 仅当片段来源是“前台案例回写（人工确认后归档）”时，才填写 `matched_writeback_case_note`；当前两个基础案例卡不属于前台回写，必须留空。

## `training_support`

```json
{
  "training_goal": "字符串",
  "key_points": ["最多5条"],
  "common_mistakes": ["最多4条"],
  "quiz_questions": ["最多5条"],
  "assessment_checklist": ["最多5条"],
  "coach_tip": "字符串",
  "uncertainty_note": "字符串"
}
```

输入里的 `fault_summary`、`possible_causes` 和 `troubleshooting_steps` 是待转译材料，不自动成为事实；仍需与检索片段一致。

## `case_review`

```json
{
  "case_title": "字符串",
  "root_cause_summary": "字符串",
  "reusable_lessons": ["最多5条"],
  "sop_update_suggestions": ["最多4条"],
  "archive_tags": ["最多5条"],
  "missing_information": ["最多4条"],
  "review_note": "字符串"
}
```

案例复盘只能基于人工填写的处置记录和检索资料。即使处理结果看起来明确，也要保留缺失的测量值、复测记录或审核人等归档缺口。
