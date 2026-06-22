from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_runtime_sft_v3 import (
    DATA_CUTOFF,
    PROFILES,
    SPLIT_OFFSETS,
    TERM_ALTERNATIVES,
    case_review_payload,
    case_reference,
    contains_illegal_strong_term,
    ctx,
    log_context,
    manual_context,
    reviewed_writeback_context,
    same_device_wrong_symptom_context,
    sop_context,
    training_payload,
    unconfirmed_case_context,
    unit_name,
    wrong_device_context,
)
from scripts.validate_sft_dataset import TASK_SCHEMAS, validate_assistant_output, validate_messages, validate_runtime_semantics
from src.config import AppConfig
from src.llm_client import build_runtime_system_prompt, build_runtime_user_message


REFINED_DIR = ROOT / "data" / "sft" / "runtime_v3_refined"
DEFAULT_MATRIX = REFINED_DIR / "scenario_matrix.json"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DMXAPI_DEFAULT_MODEL = "gemini-3-flash-preview"
DMXAPI_DEFAULT_WORKERS = 1000
DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
CASE_MISSING_ALTERNATIVES = {
    "实际处理步骤": ("实际处理步骤", "处理步骤", "实际处置步骤"),
    "最终结论": ("最终结论", "最终处理结论", "最终处理结果", "最终结果", "final_result"),
    "经验总结": ("经验总结", "经验归纳", "经验教训"),
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def resolve_teacher_provider(
    config: AppConfig,
    provider: str,
    model_override: str | None,
    workers_override: int | None,
) -> tuple[str, str, str, int]:
    """Return credentials and concurrency without changing the runtime LLM path."""
    if provider == "deepseek":
        api_key = config.deepseek_api_key
        base_url = config.deepseek_base_url
        model = model_override or DEEPSEEK_DEFAULT_MODEL
        workers = workers_override or 1
        key_name = "deepseek_api_key"
    elif provider == "dmxapi":
        api_key = config.dmx_api_key
        base_url = config.dmx_base_url
        model = model_override or config.dmx_model or DMXAPI_DEFAULT_MODEL
        workers = workers_override or DMXAPI_DEFAULT_WORKERS
        key_name = "dmx_api_key"
    else:  # pragma: no cover - argparse guards choices.
        raise ValueError(f"未知教师提供方：{provider}")
    if not api_key.strip():
        raise ValueError(f"未配置 {key_name}。")
    if workers <= 0:
        raise ValueError("workers 必须为正数。")
    return api_key, base_url.rstrip("/"), model, workers


def create_teacher_client(api_key: str, base_url: str, workers: int) -> OpenAI:
    """Create an OpenAI-compatible client sized for the selected worker count.

    The default OpenAI client pool is intentionally conservative.  DMXAPI runs
    can use 1,000 worker threads, so size the HTTP connection pool explicitly
    instead of silently throttling that path below its configured concurrency.
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=httpx.Timeout(120.0, connect=30.0),
        http_client=httpx.Client(
            limits=httpx.Limits(
                max_connections=workers,
                max_keepalive_connections=workers,
            ),
            timeout=httpx.Timeout(120.0, connect=30.0),
        ),
    )


def load_retry_scenarios(path: Path, split: str) -> list[dict[str, Any]]:
    """Load rejected rows without rebuilding or reshuffling their scenarios."""
    scenarios: list[dict[str, Any]] = []
    scenario_ids: set[str] = set()
    for line_number, rejected in enumerate(load_jsonl(path), start=1):
        scenario = rejected.get("scenario")
        if not isinstance(scenario, dict):
            raise ValueError(f"{path} 第 {line_number} 行缺少可重试的 scenario。")
        scenario_id = str(scenario.get("id", "")).strip()
        if not scenario_id:
            raise ValueError(f"{path} 第 {line_number} 行的 scenario 缺少 id。")
        if scenario.get("split") != split:
            raise ValueError(f"{path} 第 {line_number} 行属于 split={scenario.get('split')!r}，不是 {split!r}。")
        if scenario_id in scenario_ids:
            raise ValueError(f"{path} 含重复 scenario id：{scenario_id}。")
        scenario_ids.add(scenario_id)
        scenarios.append(scenario)
    return scenarios


def merge_records(existing: list[dict[str, Any]], recovered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine original accepted rows with recovered rows, rejecting ambiguous IDs."""
    merged: dict[str, dict[str, Any]] = {}
    for label, rows in (("已有接受样本", existing), ("重试接受样本", recovered)):
        for row in rows:
            example_id = str(row.get("id", "")).strip()
            if not example_id:
                raise ValueError(f"{label}中存在缺少 id 的样本。")
            if example_id in merged:
                raise ValueError(f"不能合并：样本 id 重复 {example_id}。")
            merged[example_id] = row
    return [merged[key] for key in sorted(merged)]


def task_for_family(family: str) -> str:
    if family.startswith("fault_"):
        return "fault_diagnosis"
    if family.startswith("case_review_"):
        return "case_review"
    if family.startswith("training_"):
        return "training_support"
    raise ValueError(f"未知情景族：{family}")


def material_id(sequence: int, split: str) -> int:
    return SPLIT_OFFSETS[split] + sequence + 50_000


def metadata(
    scenario_type: str,
    evidence_level: int,
    retrieval_quality: str,
    writeback_match_type: str,
    *,
    use_common_sense: bool = False,
    abstain_root_cause: bool = False,
) -> dict[str, Any]:
    return {
        "scenario_type": scenario_type,
        "evidence_level": evidence_level,
        "retrieval_quality": retrieval_quality,
        "writeback_match_type": writeback_match_type,
        "should_use_common_sense": use_common_sense,
        "should_abstain_from_root_cause": abstain_root_cause,
        "data_cutoff": DATA_CUTOFF.strftime("%Y-%m-%d"),
    }


def fault_payload(profile: Any, unit: str) -> dict[str, Any]:
    return {"device": unit, "symptom": profile.symptom}


def incomplete_case_payload(profile: Any, unit: str, sequence: int, kind: str) -> dict[str, Any]:
    payload = case_review_payload(profile, unit, sequence, verified=False)
    if kind == "missing_actual":
        payload["actual_steps"] = ""
    elif kind == "missing_final":
        payload["final_result"] = ""
    elif kind == "short_experience":
        payload["experience_summary"] = "以后注意。"
    elif kind == "steps_result_conflict":
        payload["actual_steps"] = "更换滤芯后症状暂时减轻，未记录其他检查。"
        payload["final_result"] = "最终结论为温度传感器故障。"
    return payload


def make_scenario(family: str, sequence: int, split: str) -> dict[str, Any]:
    task = task_for_family(family)
    synthetic_id = material_id(sequence, split)
    profile = PROFILES[synthetic_id % len(PROFILES)]
    unit = unit_name(profile, synthetic_id)
    contexts: list[dict[str, str]] = []
    contract: dict[str, Any] = {"writeback_policy": "none", "required_terms": []}

    if task == "fault_diagnosis":
        payload = fault_payload(profile, unit)
        if family == "fault_full_evidence_writeback":
            reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id), reviewed]
            meta = metadata(family, 4, "sufficient", "exact", abstain_root_cause=True)
            contract = {"writeback_policy": "exact", "references": [reference], "required_terms": ["现场复核"]}
        elif family == "fault_only_sop":
            contexts = [sop_context(profile, unit, synthetic_id)]
            meta = metadata(family, 1, "weak", "none", abstain_root_cause=True)
        elif family == "fault_no_retrieval_strict":
            meta = metadata(family, 0, "none", "none", use_common_sense=True, abstain_root_cause=True)
            contract = {"writeback_policy": "none", "required_terms": [], "common_sense_no_retrieval": True, "conservative_common_sense": True}
        elif family == "fault_no_retrieval_with_common_sense":
            meta = metadata(family, 0, "none", "none", use_common_sense=True, abstain_root_cause=True)
            contract = {"writeback_policy": "none", "required_terms": [], "common_sense_no_retrieval": True}
        elif family == "fault_single_writeback":
            reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id)
            contexts = [reviewed]
            meta = metadata(family, 4, "partial", "exact", abstain_root_cause=True)
            contract = {"writeback_policy": "exact", "references": [reference], "required_terms": ["现场复核"]}
        elif family == "fault_multi_writeback_consistent":
            first, first_ref = reviewed_writeback_context(profile, unit, synthetic_id, offset=0)
            second, second_ref = reviewed_writeback_context(profile, unit, synthetic_id, offset=1)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id), first, second]
            meta = metadata(family, 4, "sufficient", "multiple_consistent", abstain_root_cause=True)
            contract = {"writeback_policy": "multiple_consistent", "references": [first_ref, second_ref], "required_terms": ["一致", "现场复核"]}
        elif family == "fault_multi_writeback_conflicting":
            first, first_ref = reviewed_writeback_context(profile, unit, synthetic_id, offset=0)
            second, second_ref = reviewed_writeback_context(profile, unit, synthetic_id, offset=1, conflict=True)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="conflict"), first, second]
            meta = metadata(family, 4, "conflicting", "multiple_conflicting", abstain_root_cause=True)
            contract = {"writeback_policy": "multiple_conflicting", "references": [first_ref, second_ref], "required_terms": ["冲突", "不能直接"]}
        elif family == "fault_truncated_or_missing":
            variant = synthetic_id % 4
            if variant == 0:
                reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id, missing={"final_result", "experience_summary"}, truncate=True)
                required = ["截断"]
            elif variant == 1:
                reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id, missing={"operator", "written_at"})
                required = ["缺少"]
            elif variant == 2:
                reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id, missing={"actual_steps"})
                required = ["实际处理步骤"]
            else:
                reviewed, reference = reviewed_writeback_context(profile, unit, synthetic_id, missing={"final_result"})
                required = ["最终结论"]
            contexts = [manual_context(profile, unit, synthetic_id), reviewed]
            meta = metadata(family, 3, "partial", "partial", abstain_root_cause=True)
            contract = {"writeback_policy": "partial", "references": [reference], "required_terms": required}
        elif family == "fault_wrong_device":
            contexts = [wrong_device_context(profile, synthetic_id)]
            meta = metadata(family, 1, "noisy", "none", abstain_root_cause=True)
            contract = {"writeback_policy": "none", "required_terms": ["设备不一致"], "strict_no_retrieval": True}
        elif family == "fault_same_device_wrong_symptom":
            contexts = [same_device_wrong_symptom_context(profile, unit, synthetic_id)]
            meta = metadata(family, 1, "noisy", "none", abstain_root_cause=True)
            contract = {"writeback_policy": "none", "required_terms": ["症状不一致"], "strict_no_retrieval": True}
        elif family == "fault_stale_sop_or_new_conflict":
            if synthetic_id % 2:
                contexts = [sop_context(profile, unit, synthetic_id, stale=True)]
                required = ["不适用"]
            else:
                contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="conflict")]
                required = ["冲突"]
            meta = metadata(family, 2, "conflicting", "none", abstain_root_cause=True)
            contract = {"writeback_policy": "none", "required_terms": required}
        else:  # pragma: no cover
            raise ValueError(f"未实现故障情景：{family}")
    elif task == "case_review":
        if family == "case_review_complete":
            payload = case_review_payload(profile, unit, synthetic_id, verified=True)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="verified")]
            meta = metadata(family, 5, "sufficient", "not_applicable")
        elif family == "case_review_missing_actual_steps":
            payload = incomplete_case_payload(profile, unit, synthetic_id, "missing_actual")
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id)]
            meta = metadata(family, 2, "partial", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "case_missing": "实际处理步骤", "required_terms": ["实际处理步骤"]}
        elif family == "case_review_missing_final_conclusion":
            payload = incomplete_case_payload(profile, unit, synthetic_id, "missing_final")
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id)]
            meta = metadata(family, 2, "partial", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "case_missing": "最终结论", "required_terms": ["最终结论"]}
        elif family == "case_review_short_experience":
            payload = incomplete_case_payload(profile, unit, synthetic_id, "short_experience")
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id)]
            meta = metadata(family, 2, "partial", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "case_missing": "经验总结", "required_terms": ["经验总结"]}
        elif family == "case_review_steps_result_conflict":
            payload = incomplete_case_payload(profile, unit, synthetic_id, "steps_result_conflict")
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="conflict")]
            meta = metadata(family, 2, "conflicting", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "case_conflict": True, "required_terms": ["不一致"]}
        elif family == "case_review_history_conflict":
            payload = case_review_payload(profile, unit, synthetic_id, verified=False)
            first, _ = reviewed_writeback_context(profile, unit, synthetic_id, offset=0)
            second, _ = reviewed_writeback_context(profile, unit, synthetic_id, offset=1, conflict=True)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="conflict"), first, second]
            meta = metadata(family, 2, "conflicting", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "case_conflict": True, "required_terms": ["冲突"]}
        elif family == "case_review_sop_update":
            payload = case_review_payload(profile, unit, synthetic_id, verified=True)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="verified"), sop_context(profile, unit, synthetic_id)]
            meta = metadata(family, 5, "sufficient", "not_applicable")
            contract = {"writeback_policy": "not_applicable", "requires_sop_update": True, "required_terms": ["SOP"]}
        else:  # pragma: no cover
            raise ValueError(f"未实现复盘情景：{family}")
    else:
        if family == "training_sop":
            payload = training_payload(profile, unit, 1)
            contexts = [sop_context(profile, unit, synthetic_id)]
            meta = metadata(family, 1, "weak", "not_applicable", abstain_root_cause=True)
        elif family == "training_history":
            payload = training_payload(profile, unit, 2)
            contexts = [manual_context(profile, unit, synthetic_id), unconfirmed_case_context(profile, unit, synthetic_id)]
            meta = metadata(family, 3, "partial", "not_applicable", abstain_root_cause=True)
        elif family == "training_conflict":
            payload = training_payload(profile, unit, 2)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="conflict")]
            meta = metadata(family, 2, "conflicting", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "required_terms": ["冲突"]}
        elif family == "training_truncated":
            payload = training_payload(profile, unit, 2)
            reviewed, _ = reviewed_writeback_context(profile, unit, synthetic_id, missing={"final_result"}, truncate=True)
            contexts = [manual_context(profile, unit, synthetic_id), reviewed]
            meta = metadata(family, 3, "partial", "not_applicable", abstain_root_cause=True)
            contract = {"writeback_policy": "not_applicable", "required_terms": ["不完整"]}
        elif family == "training_new_employee":
            payload = training_payload(profile, unit, 1)
            contexts = [sop_context(profile, unit, synthetic_id), manual_context(profile, unit, synthetic_id)]
            meta = metadata(family, 1, "sufficient", "not_applicable", abstain_root_cause=True)
        elif family == "training_post_review":
            payload = training_payload(profile, unit, 2)
            contexts = [manual_context(profile, unit, synthetic_id), log_context(profile, unit, synthetic_id, status="verified")]
            meta = metadata(family, 5, "sufficient", "not_applicable")
        else:  # pragma: no cover
            raise ValueError(f"未实现培训情景：{family}")
        if not contract:
            contract = {"writeback_policy": "not_applicable", "required_terms": []}

    return {
        "id": f"RT3R-{split}-{family}-{sequence:04d}",
        "task": task,
        "split": split,
        "payload": payload,
        "context_items": contexts,
        "metadata": meta,
        "contract": contract,
    }


def build_scenarios(matrix: dict[str, Any], split: str, *, one_per_family: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    counts = matrix.get(split)
    if not isinstance(counts, dict):
        raise ValueError(f"矩阵缺少 split={split}")
    if one_per_family:
        return [make_scenario(family, index, split) for index, family in enumerate(counts)]
    rows: list[dict[str, Any]] = []
    sequence = 0
    for family, count in counts.items():
        for _ in range(int(count)):
            rows.append(make_scenario(family, sequence, split))
            sequence += 1
            if limit and len(rows) >= limit:
                return rows
    return rows


def instruction(scenario: dict[str, Any]) -> str:
    meta = scenario["metadata"]
    contract = scenario["contract"]
    lines = [
        f"证据等级={meta['evidence_level']}，检索质量={meta['retrieval_quality']}。",
        "证据观察只写输入或检索中直接存在的事实；候选原因必须标明来源与不确定性。",
        "低于证据等级 5 时，summary、possible_causes 不得包含“根因是、确认当前、导致、引起、异常消失”等当前确定或直接因果表达；统一改写为“可能相关/候选方向/待现场复核”。",
        "严格遵守列表上限：evidence_observations 最多 4 条、possible_causes 最多 4 条、troubleshooting_steps 最多 5 条、risk_notes 最多 4 条、training_points 最多 4 条。",
        "assessment_checklist 是字符串数组，禁止 □、☐ 等视觉符号。",
    ]
    if contract.get("strict_no_retrieval"):
        lines.append("当前没有可用内部证据。summary 必须先说明未检索/未命中；possible_causes 必须仅为“证据不足，暂不列出具体原因。”。排查步骤只能是补证、复核等低风险动作，不得建议拆检、更换、调参等高风险操作。")
    if contract.get("common_sense_no_retrieval"):
        lines.append("当前无内部检索命中。summary 必须先说明未检索/未命中；可在 possible_causes 和低风险步骤中给行业通用建议，但原因只能写为“可能参与/待核对”，不得写“导致/引起”的直接因果。uncertainty_note 必须原样包含“行业常识不是知识库证据，不能确认当前根因。”。不得把行业常识写入 evidence_observations，不得建议拆检、更换、调参等高风险操作。")
    if contract.get("conservative_common_sense"):
        lines.append("这是保守型无检索样本：行业常识建议只保留记录、目视核对、状态确认等低风险补证方向，不给出拆检、替换或参数调整方案。")
    policy = contract.get("writeback_policy")
    refs = contract.get("references", [])
    if policy in {"exact", "multiple_consistent", "multiple_conflicting"}:
        items = "；".join(f"{item['case_id']} / {item['operator']} / {item['written_at']}" for item in refs)
        lines.append(f"完整回写命中必须逐项引用：{items}；历史案例只能作为参考，必须保留本次现场复核边界。")
    if policy == "partial":
        lines.append("回写片段不完整或截断，matched_writeback_case_note 必须明确疑似/不完整及缺失字段，不能当完整命中。")
    if contract.get("case_missing"):
        lines.append(f"案例复盘输入缺少“{contract['case_missing']}”；missing_information 和 review_note 必须指出该缺口，不能补造。")
    if contract.get("case_conflict"):
        lines.append("案例输入或历史资料存在冲突；root_cause_summary、missing_information、review_note 必须说明冲突，不能强行合并。")
    if contract.get("requires_sop_update"):
        lines.append("该复盘必须在 sop_update_suggestions 中提出可执行的 SOP 更新建议。")
    if contract.get("required_terms"):
        lines.append(f"输出必须明确体现：{'、'.join(contract['required_terms'])}。")
    return "\n".join(lines)


def teacher_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    task = scenario["task"]
    runtime_system = build_runtime_system_prompt(task)
    runtime_user = build_runtime_user_message(task, scenario["payload"], scenario["context_items"])
    return [
        {"role": "system", "content": "你是工业维保 RAG 金标准标注教师。可以内部思考，但最终只输出 assistant_output JSON，不得泄露思考过程。"},
        {"role": "user", "content": f"根据线上请求生成目标 JSON。\n\n[线上 system]\n{runtime_system}\n\n[线上 user]\n{runtime_user}\n\n[验收规则]\n{instruction(scenario)}\n\n返回：{{\"assistant_output\": {{...}}}}"},
    ]


def generate_target(client: OpenAI, scenario: dict[str, Any], model: str, max_tokens: int, *, previous: dict[str, Any] | None = None, errors: list[str] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            messages = teacher_messages(scenario)
            if previous is not None and errors:
                repair_notes = list(errors)
                if any("低证据等级出现当前强结论" in error for error in errors):
                    repair_notes.append("删除 summary 和 possible_causes 中所有“导致/引起/根因是/确认当前”；例如把“油位偏低导致吸空”改为“油位偏低（候选方向，待现场复核）”。")
                if any("超出 4 条上限" in error for error in errors):
                    repair_notes.append("将对应列表压缩为最多 4 条，保留最直接的证据或候选项。")
                messages[-1]["content"] += "\n\n[上一版输出]\n" + json.dumps(previous, ensure_ascii=False) + "\n[必须修复]\n- " + "\n- ".join(repair_notes) + "\n只返回修正后的 assistant_output。"
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled"}, "user_id": "zhiweishi_sft_refined_v03"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            if isinstance(payload, dict) and isinstance(payload.get("assistant_output"), dict):
                return payload["assistant_output"]
            if isinstance(payload, dict) and set(payload) == set(TASK_SCHEMAS[scenario["task"]]):
                return payload
            raise ValueError("教师响应未提供 assistant_output。")
        except Exception as error:
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"教师调用连续失败：{last_error}")


def full_answer_text(answer: dict[str, Any]) -> str:
    values: list[str] = []
    for value in answer.values():
        values.extend(str(item) for item in value) if isinstance(value, list) else values.append(str(value))
    return "\n".join(values)


def contains_illegal_current_claim(text: str) -> bool:
    """Keep the baseline strong-claim gate without rejecting an explicit negation."""
    negative_claims = ("不能确认当前", "尚不能确认当前", "无法确认当前", "不代表当前根因", "不是当前根因结论", "非当前根因结论")
    cleaned = text
    for phrase in negative_claims:
        cleaned = cleaned.replace(phrase, "")
    return contains_illegal_strong_term(cleaned)


def contains_high_risk_action(text: str) -> bool:
    """Reject action recommendations while allowing low-risk review of maintenance records."""
    cleaned = text
    for phrase in ("更换日期", "更换记录", "最近更换", "更换历史", "更换周期"):
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"(?:核对|查阅|记录|确认)[^。；\n]*更换[^。；\n]*(?:记录|日期|历史|周期)", "", cleaned)
    return any(token in cleaned for token in ("拆检", "调整参数", "直接更换", "立即拆", "更换"))


def validate_refined_contract(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    example_id = str(record.get("id", "unknown"))
    scenario = record.get("scenario")
    messages = record.get("messages")
    task = record.get("task")
    if not isinstance(scenario, dict) or not isinstance(messages, list) or len(messages) != 3 or task not in TASK_SCHEMAS:
        return [f"{example_id}: 缺少 refined scenario。"]
    payload, contexts, meta, contract = (scenario.get("payload"), scenario.get("context_items"), scenario.get("metadata"), scenario.get("contract"))
    required_meta = {"scenario_type", "evidence_level", "retrieval_quality", "writeback_match_type", "should_use_common_sense", "should_abstain_from_root_cause", "data_cutoff"}
    if not isinstance(payload, dict) or not isinstance(contexts, list) or not isinstance(meta, dict) or not isinstance(contract, dict):
        return [f"{example_id}: refined 元数据不完整。"]
    if set(meta) != required_meta:
        errors.append(f"{example_id}: refined 元数据键不一致。")
    if messages[0].get("content") != build_runtime_system_prompt(task) or messages[1].get("content") != build_runtime_user_message(task, payload, contexts):
        errors.append(f"{example_id}: 未复用线上 system/user 格式。")
    try:
        answer = json.loads(str(messages[2].get("content", "")))
    except json.JSONDecodeError:
        return errors
    output_text = full_answer_text(answer)
    if "□" in output_text or "☐" in output_text:
        errors.append(f"{example_id}: 输出含未统一 checklist 符号。")
    for text in DATE_PATTERN.findall(messages[1].get("content", "") + "\n" + output_text):
        if datetime.strptime(text, "%Y-%m-%d") > DATA_CUTOFF:
            errors.append(f"{example_id}: 存在 cutoff 后日期 {text}。")
    level = int(meta["evidence_level"])
    if task == "fault_diagnosis":
        note = str(answer.get("matched_writeback_case_note", "")).strip()
        policy = contract.get("writeback_policy")
        if policy == "none" and note:
            errors.append(f"{example_id}: 无回写命中时 note 必须为空。")
        if policy in {"exact", "multiple_consistent", "multiple_conflicting"}:
            for ref in contract.get("references", []):
                for key in ("case_id", "operator", "written_at"):
                    if str(ref[key]) not in note:
                        errors.append(f"{example_id}: 回写说明缺少 {key}。")
        if policy == "partial" and (not note or not any(word in note for word in ("疑似", "不完整", "缺少", "截断"))):
            errors.append(f"{example_id}: 不完整回写未降级说明。")
        causes = answer.get("possible_causes", [])
        if contract.get("strict_no_retrieval"):
            summary = str(answer.get("summary", ""))
            steps = "\n".join(str(item) for item in answer.get("troubleshooting_steps", []))
            if causes != ["证据不足，暂不列出具体原因。"]:
                errors.append(f"{example_id}: 严格无检索不得展开原因。")
            if not any(token in summary for token in ("未检索", "无检索", "未命中")):
                errors.append(f"{example_id}: 严格无检索样本未声明检索状态。")
            if contains_high_risk_action(steps):
                errors.append(f"{example_id}: 严格无检索包含高风险操作。")
        if contract.get("common_sense_no_retrieval"):
            summary = str(answer.get("summary", ""))
            uncertainty = str(answer.get("uncertainty_note", ""))
            joined_causes = "\n".join(str(item) for item in causes)
            steps = "\n".join(str(item) for item in answer.get("troubleshooting_steps", []))
            evidence = "\n".join(str(item) for item in answer.get("evidence_observations", []))
            boundary_text = "\n".join((summary, joined_causes, uncertainty))
            if not any(token in summary for token in ("未检索", "无检索", "未命中")):
                errors.append(f"{example_id}: 无检索行业常识样本未声明检索状态。")
            if not any(token in joined_causes for token in ("行业常识", "通用经验", "通用常识", "行业通用逻辑", "行业通用建议")) or "知识库证据" not in boundary_text:
                errors.append(f"{example_id}: 行业常识原因未标注来源和知识库证据边界。")
            if not any(token in uncertainty for token in ("行业常识", "通用经验", "通用常识")) or not any(token in uncertainty for token in ("不能", "不等同", "不替代")):
                errors.append(f"{example_id}: 行业常识不确定性边界不足。")
            if not any(token in boundary_text for token in ("不能确认当前根因", "不代表当前根因", "不是当前根因结论", "非当前根因结论", "不代表对本次故障根因的最终判定")):
                errors.append(f"{example_id}: 行业常识未声明不是当前根因结论。")
            if contains_high_risk_action(steps):
                errors.append(f"{example_id}: 无检索行业常识包含高风险操作。")
            if any(token in evidence for token in ("堵塞", "磨损", "漂移", "接触不良")):
                errors.append(f"{example_id}: 无检索 evidence_observations 混入未观测故障事实。")
        if level < 5:
            current_claim = "\n".join([str(answer.get("summary", "")), *[str(item) for item in causes]])
            if contains_illegal_current_claim(current_claim):
                errors.append(f"{example_id}: 低证据等级出现当前强结论。")
            uncertainty = str(answer.get("uncertainty_note", ""))
            if not any(token in uncertainty for token in ("尚不能确认", "需", "证据不足", "待", "不确定", "无法", "不能", "当前")):
                errors.append(f"{example_id}: 低证据等级未说明不确定性。")
    if task == "case_review":
        if level < 5:
            if contains_illegal_current_claim(str(answer.get("root_cause_summary", ""))):
                errors.append(f"{example_id}: 候选复盘出现确定性根因。")
            if "候选" not in str(answer.get("review_note", "")):
                errors.append(f"{example_id}: 候选复盘未标记候选状态。")
        if contract.get("case_missing"):
            missing_name = str(contract["case_missing"])
            missing_text = "\n".join(answer.get("missing_information", []))
            aliases = CASE_MISSING_ALTERNATIVES.get(missing_name, (missing_name,))
            if not any(alias in missing_text for alias in aliases):
                errors.append(f"{example_id}: 未识别缺少的 {missing_name}。")
        if contract.get("case_conflict") and not any(word in output_text for word in ("冲突", "不一致", "矛盾")):
            errors.append(f"{example_id}: 未识别案例冲突。")
        if contract.get("requires_sop_update") and not any("SOP" in item for item in answer.get("sop_update_suggestions", [])):
            errors.append(f"{example_id}: 缺少 SOP 更新建议。")
    missing_terms: list[str] = []
    for term in contract.get("required_terms", []):
        choices = TERM_ALTERNATIVES.get(term, (term,))
        if not any(choice in output_text for choice in choices):
            missing_terms.append(term)
    if missing_terms:
        errors.append(f"{example_id}: 未体现要求判断：{missing_terms}。")
    return errors


def build_record(
    scenario: dict[str, Any],
    assistant_output: dict[str, Any],
    model: str,
    provider: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    task = scenario["task"]
    messages = [
        {"role": "system", "content": build_runtime_system_prompt(task)},
        {"role": "user", "content": build_runtime_user_message(task, scenario["payload"], scenario["context_items"])},
        {"role": "assistant", "content": json.dumps(assistant_output, ensure_ascii=False)},
    ]
    record = {
        "id": scenario["id"], "task": task, "split": scenario["split"], "review_status": "teacher_generated", "source_mode": "synthetic_runtime_v3",
        "messages": messages,
        "scenario": {"payload": scenario["payload"], "context_items": scenario["context_items"], "metadata": scenario["metadata"], "contract": scenario["contract"]},
        "teacher": {
            "provider": provider,
            "model": model,
            "thinking": "enabled",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    errors = validate_messages(messages, record["id"])
    errors.extend(validate_assistant_output(task, messages[2]["content"], record["id"]))
    errors.extend(validate_runtime_semantics(task, messages, record["id"]))
    errors.extend(validate_refined_contract(record))
    return (record if not errors else None), errors


def generate_record(
    client: OpenAI,
    scenario: dict[str, Any],
    model: str,
    max_tokens: int,
    provider: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
    previous: dict[str, Any] | None = None
    errors: list[str] = []
    for _ in range(4):
        output = generate_target(client, scenario, model, max_tokens, previous=previous, errors=errors or None)
        record, errors = build_record(scenario, output, model, provider)
        if record:
            return record, [], output
        previous = output
    return None, errors, previous


def coverage(rows: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    def counter(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))
    metadata_rows = [row["scenario"]["metadata"] for row in rows]
    return {
        "planned": counter([scenario["metadata"]["scenario_type"] for scenario in scenarios]),
        "accepted": {
            "total": len(rows),
            "by_task": counter([row["task"] for row in rows]),
            "by_scenario": counter([item["scenario_type"] for item in metadata_rows]),
            "common_sense_no_retrieval": sum(bool(item["should_use_common_sense"]) for item in metadata_rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构建精选、证据边界清晰的 refined SFT 数据。")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--split", choices=tuple(SPLIT_OFFSETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-provider",
        choices=("deepseek", "dmxapi"),
        default="deepseek",
        help="教师模型提供方。默认保留原有 DeepSeek 链路；dmxapi 使用 Gemini 3 Flash Preview。",
    )
    parser.add_argument(
        "--model",
        help="覆盖该提供方的默认模型；dmxapi 默认读取 dmx_model（gemini-3-flash-preview）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="覆盖并发请求数；dmxapi 默认 1000，deepseek 默认 1。",
    )
    parser.add_argument("--max-tokens", type=int, default=3200)
    parser.add_argument("--one-per-family", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--retry-rejected",
        type=Path,
        help="读取本生成器写出的 *_rejected.jsonl，并严格按其中的原 scenario 重试。",
    )
    parser.add_argument(
        "--merge-with",
        type=Path,
        help="将本次重试成功样本与已有接受样本合并后写入 --output；不会静默覆盖重复 id。",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers is not None and args.workers <= 0:
        parser.error("workers 必须为正数。")
    if args.one_per_family and args.limit is not None:
        parser.error("--one-per-family 与 --limit 不能并用。")
    if args.retry_rejected and (args.one_per_family or args.limit is not None):
        parser.error("--retry-rejected 不能与 --one-per-family 或 --limit 并用。")
    if args.merge_with and not args.retry_rejected:
        parser.error("--merge-with 只能与 --retry-rejected 一起使用。")
    matrix = load_json(args.matrix.resolve())
    planned_scenarios = build_scenarios(matrix, args.split)
    scenarios = (
        load_retry_scenarios(args.retry_rejected.resolve(), args.split)
        if args.retry_rejected
        else build_scenarios(matrix, args.split, one_per_family=args.one_per_family, limit=args.limit)
    )
    mode = "原 scenario 重试" if args.retry_rejected else "常规生成"
    if args.teacher_provider == "dmxapi":
        preview_model = args.model or "dmx_model（默认 gemini-3-flash-preview）"
        preview_workers = args.workers or DMXAPI_DEFAULT_WORKERS
    else:
        preview_model = args.model or DEEPSEEK_DEFAULT_MODEL
        preview_workers = args.workers or 1
    print(
        f"refined runtime split={args.split}；模式={mode}；教师提供方={args.teacher_provider}；"
        f"模型={preview_model}；情景数={len(scenarios)}；并发={preview_workers}；"
        f"分布={coverage([], scenarios)['planned']}"
    )
    if not args.execute:
        print("预览模式：未调用教师 API。")
        return 0
    config = AppConfig.from_env()
    try:
        api_key, base_url, model, workers = resolve_teacher_provider(
            config,
            args.teacher_provider,
            args.model,
            args.workers,
        )
    except ValueError as error:
        print(error)
        return 1
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        print(f"输出已存在，拒绝覆盖：{output}")
        return 1
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with create_teacher_client(api_key, base_url, workers) as client:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(generate_record, client, scenario, model, args.max_tokens, args.teacher_provider): scenario
                for scenario in scenarios
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                scenario = futures[future]
                try:
                    record, errors, raw = future.result()
                    if record:
                        accepted.append(record)
                    else:
                        rejected.append({"scenario": scenario, "reason": errors, "raw": raw})
                except Exception as error:
                    rejected.append({"scenario": scenario, "reason": [f"教师调用失败：{error}"]})
                if completed % 20 == 0 or completed == len(scenarios):
                    print(f"进度 {completed}/{len(scenarios)}；接受 {len(accepted)}；拒绝 {len(rejected)}。")
    accepted.sort(key=lambda item: item["id"])
    output_rows = accepted
    coverage_scenarios = scenarios
    if args.merge_with:
        existing = load_jsonl(args.merge_with.resolve())
        output_rows = merge_records(existing, accepted)
        coverage_scenarios = planned_scenarios
    write_jsonl(output, output_rows, overwrite=args.overwrite)
    if rejected:
        write_jsonl(output.with_name(f"{output.stem}_rejected.jsonl"), rejected, overwrite=args.overwrite)
    output.with_name(f"{output.stem}_coverage_report.json").write_text(json.dumps(coverage(output_rows, coverage_scenarios), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成：本次接受 {len(accepted)}；拒绝 {len(rejected)}；写出 {len(output_rows)}；输出={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
