from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from openai import OpenAI

from src.config import AppConfig


FAULT_SCHEMA_EXAMPLE = {
    "summary": "一句话结论",
    "evidence_observations": ["证据观察1", "证据观察2"],
    "possible_causes": ["原因1", "原因2"],
    "troubleshooting_steps": ["步骤1", "步骤2"],
    "risk_notes": ["风险1"],
    "escalation_advice": "一句简洁建议",
    "training_points": ["培训要点1", "培训要点2"],
    "uncertainty_note": "一句保守说明",
}

TRAINING_SCHEMA_EXAMPLE = {
    "training_goal": "一句话培训目标",
    "key_points": ["要点1", "要点2"],
    "common_mistakes": ["易错点1", "易错点2"],
    "quiz_questions": ["问题1", "问题2"],
    "assessment_checklist": ["检查项1", "检查项2"],
    "coach_tip": "一句带教提示",
    "uncertainty_note": "一句保守说明",
}

CASE_REVIEW_SCHEMA_EXAMPLE = {
    "case_title": "案例标题",
    "root_cause_summary": "一句话根因概括",
    "reusable_lessons": ["经验1", "经验2"],
    "sop_update_suggestions": ["建议1", "建议2"],
    "archive_tags": ["标签1", "标签2"],
    "missing_information": ["缺失项1", "缺失项2"],
    "review_note": "一句归档建议",
}


@dataclass(frozen=True)
class PromptTaskSpec:
    label: str
    reasoning_framework: tuple[str, ...]
    style_rules: tuple[str, ...]
    output_schema: dict[str, Any]


TASK_PROMPTS: dict[str, PromptTaskSpec] = {
    "fault_diagnosis": PromptTaskSpec(
        label="故障排查卡",
        reasoning_framework=(
            "问题界定：先确认设备、现象和当前运行边界。",
            "证据抽取：只提炼检索片段里直接出现的观测和限制。",
            "原因假设：基于证据给出最多 4 条候选原因，并按可信度排序。",
            "动作规划：按先易后难、先低风险后高风险给出排查步骤。",
            "边界控制：单列停机条件、禁止动作和需人工复核事项。",
            "不确定性声明：证据不足时明确写出，不得用想象补齐。",
        ),
        style_rules=(
            "用现场维保口吻，短句、直接、可执行。",
            "先给结论，再给依据，不写空泛分析。",
            "避免宣传语、聊天语气和无依据推测。",
        ),
        output_schema=FAULT_SCHEMA_EXAMPLE,
    ),
    "training_support": PromptTaskSpec(
        label="培训模式",
        reasoning_framework=(
            "培训目标：先明确本次培训希望新人掌握什么。",
            "证据转译：把检索依据改写成岗位操作要点。",
            "易错点归纳：提炼新人最容易犯的误判和遗漏。",
            "演练设计：给出可直接提问的训练题。",
            "考核清单：给出可观察、可打勾的上岗检查项。",
            "不确定性声明：超出证据范围的内容必须保守表达。",
        ),
        style_rules=(
            "用带教口吻，但保持简洁和规整。",
            "突出先做什么、为什么、做到什么程度。",
            "问题和清单都要适合班组现场直接使用。",
        ),
        output_schema=TRAINING_SCHEMA_EXAMPLE,
    ),
    "case_review": PromptTaskSpec(
        label="案例复盘与沉淀",
        reasoning_framework=(
            "事件概要：只概括本次故障和处理结果。",
            "根因提炼：用现有输入和依据文档总结根因。",
            "经验复用：归纳可推广到同类设备的经验。",
            "制度更新：指出 SOP、点检项或培训材料该补什么。",
            "归档检查：识别当前案例还缺哪些关键信息。",
            "审核边界：缺证据时只能提出审核建议，不能替代结论。",
        ),
        style_rules=(
            "用归档复盘口吻，客观、克制、标准化。",
            "优先输出可沉淀、可复用、可纳入制度的内容。",
            "不夸大处理效果，不制造不存在的根因。",
        ),
        output_schema=CASE_REVIEW_SCHEMA_EXAMPLE,
    ),
}


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("未能从模型输出中提取JSON。")


def _normalize_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _normalize_list(value: Any, default: list[str], limit: int) -> list[str]:
    if not isinstance(value, list):
        return default[:limit]

    normalized_items: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip().strip("- ").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized_items.append(text)
        if len(normalized_items) >= limit:
            break
    return normalized_items or default[:limit]


class DeepSeekFaultReasoner:
    def __init__(self, config: AppConfig) -> None:
        if not config.deepseek_api_key:
            raise ValueError("deepseek_api_key 未配置，无法调用 DeepSeek。")
        self.client = OpenAI(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url)
        self.model = config.deepseek_model

    def _format_context(self, context_items: list[dict]) -> str:
        if not context_items:
            return "无检索片段。只能使用输入字段中的事实，必须保守作答。"

        return "\n\n".join(
            [
                (
                    f"[片段{index}] 来源：{item.get('source_label', '未知来源')}"
                    f" | 类别：{item.get('doc_category', '未分类文档')}"
                    f" | 设备：{item.get('device_name', '未分类设备')}\n"
                    f"内容：{item.get('content', '')}"
                )
                for index, item in enumerate(context_items, start=1)
            ]
        )

    def _build_system_prompt(self, task_name: str) -> str:
        spec = TASK_PROMPTS[task_name]
        framework_text = "\n".join(
            f"{index}. {step}" for index, step in enumerate(spec.reasoning_framework, start=1)
        )
        style_text = "\n".join(f"- {rule}" for rule in spec.style_rules)
        return (
            "你是工业设备维保场景的知识增强助手。"
            "你必须先做基于证据的半形式化推理，再输出结构化结果。\n"
            "硬性约束：\n"
            "1. 只允许使用输入字段和检索片段中的事实。\n"
            "2. 先抽取证据，再形成判断，再给出动作建议，不得跳步。\n"
            "3. 对证据不足部分必须写入不确定性说明，不得脑补。\n"
            "4. 禁止输出 markdown、代码块、解释文本或 schema 之外的字段。\n"
            "5. 输出必须是严格 JSON，字段名和层级必须与要求完全一致。\n\n"
            f"任务场景：{spec.label}\n"
            f"半形式化推理框架：\n{framework_text}\n\n"
            f"回复风格要求：\n{style_text}"
        )

    def _normalize_fault_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": _normalize_text(
                result.get("summary"),
                "当前证据不足，建议先核对基础状态和关键连接，再继续排查。",
            ),
            "evidence_observations": _normalize_list(
                result.get("evidence_observations"),
                ["检索片段已返回与当前故障相关的维保依据。"],
                limit=4,
            ),
            "possible_causes": _normalize_list(
                result.get("possible_causes"),
                ["建议优先排查供给状态、连接状态和易损件状态。"],
                limit=4,
            ),
            "troubleshooting_steps": _normalize_list(
                result.get("troubleshooting_steps"),
                ["先核对说明书和SOP，再按日志与维修记录逐项排查。"],
                limit=5,
            ),
            "risk_notes": _normalize_list(
                result.get("risk_notes"),
                ["证据不足时不建议继续高风险运行。"],
                limit=4,
            ),
            "escalation_advice": _normalize_text(
                result.get("escalation_advice"),
                "若基础排查后仍无法定位，请停止高风险运行并转人工复核。",
            ),
            "training_points": _normalize_list(
                result.get("training_points"),
                ["先说依据，再说判断，最后说动作。"],
                limit=4,
            ),
            "uncertainty_note": _normalize_text(
                result.get("uncertainty_note"),
                "当前输出仅限于已检索到的依据片段，未覆盖内容需人工确认。",
            ),
        }

    def _normalize_training_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "training_goal": _normalize_text(
                result.get("training_goal"),
                "让新人掌握该故障的首检顺序、风险边界和依据文档。",
            ),
            "key_points": _normalize_list(
                result.get("key_points"),
                ["按先易后难的顺序执行基础排查。"],
                limit=5,
            ),
            "common_mistakes": _normalize_list(
                result.get("common_mistakes"),
                ["未核对依据文档就直接拆检。"],
                limit=4,
            ),
            "quiz_questions": _normalize_list(
                result.get("quiz_questions"),
                ["遇到当前故障时，第一步应该核对什么？"],
                limit=5,
            ),
            "assessment_checklist": _normalize_list(
                result.get("assessment_checklist"),
                ["能说清首检顺序、风险点和依据文档。"],
                limit=5,
            ),
            "coach_tip": _normalize_text(
                result.get("coach_tip"),
                "带教时要求先说明依据，再说明处理动作。",
            ),
            "uncertainty_note": _normalize_text(
                result.get("uncertainty_note"),
                "培训内容仅覆盖当前案例相关证据，超出部分需另行确认。",
            ),
        }

    def _normalize_case_review_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "case_title": _normalize_text(result.get("case_title"), "设备案例复盘"),
            "root_cause_summary": _normalize_text(
                result.get("root_cause_summary"),
                "当前材料已形成处理结论，但仍需结合归档信息做人工复核。",
            ),
            "reusable_lessons": _normalize_list(
                result.get("reusable_lessons"),
                ["同类故障应先核对基础状态，再决定是否停机。"],
                limit=5,
            ),
            "sop_update_suggestions": _normalize_list(
                result.get("sop_update_suggestions"),
                ["建议将本次高频排查顺序补充到对应 SOP。"],
                limit=4,
            ),
            "archive_tags": _normalize_list(
                result.get("archive_tags"),
                ["案例回写"],
                limit=5,
            ),
            "missing_information": _normalize_list(
                result.get("missing_information"),
                ["若要升级为标准案例，建议再做一次人工审核。"],
                limit=4,
            ),
            "review_note": _normalize_text(
                result.get("review_note"),
                "建议按归档模板复核后发布，避免把临时判断直接固化为标准结论。",
            ),
        }

    def _run_task(self, task_name: str, payload: dict[str, Any], context_items: list[dict]) -> dict[str, Any]:
        spec = TASK_PROMPTS[task_name]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            max_tokens=1400,
            messages=[
                {"role": "system", "content": self._build_system_prompt(task_name)},
                {
                    "role": "user",
                    "content": (
                        f"输入字段：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
                        f"检索片段：\n{self._format_context(context_items)}\n\n"
                        f"输出 JSON，字段必须为：{json.dumps(spec.output_schema, ensure_ascii=False)}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        raw_result = extract_json_object(content)
        if task_name == "fault_diagnosis":
            return self._normalize_fault_result(raw_result)
        if task_name == "training_support":
            return self._normalize_training_result(raw_result)
        return self._normalize_case_review_result(raw_result)

    def build_fault_card(self, device: str, symptom: str, context_items: list[dict]) -> dict[str, Any]:
        return self._run_task(
            "fault_diagnosis",
            payload={"device": device, "symptom": symptom},
            context_items=context_items,
        )

    def build_training_card(self, fault_result: dict[str, Any], context_items: list[dict]) -> dict[str, Any]:
        payload = {
            "device": fault_result.get("device", ""),
            "symptom": fault_result.get("symptom", ""),
            "fault_summary": fault_result.get("summary", ""),
            "possible_causes": fault_result.get("possible_causes", []),
            "troubleshooting_steps": fault_result.get("troubleshooting_steps", []),
            "risk_notes": fault_result.get("risk_notes", []),
            "training_points": fault_result.get("training_points", []),
        }
        return self._run_task("training_support", payload=payload, context_items=context_items)

    def build_case_review(self, case_payload: dict[str, Any], context_items: list[dict]) -> dict[str, Any]:
        return self._run_task("case_review", payload=case_payload, context_items=context_items)
