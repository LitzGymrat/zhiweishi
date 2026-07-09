from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import argparse


ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "sft"
DATASET_PATH = SFT_DIR / "seed_examples.jsonl"
MANIFEST_PATH = SFT_DIR / "source_manifest.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_SCHEMAS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "fault_diagnosis": {
        "summary": str,
        "matched_writeback_case_note": str,
        "evidence_observations": list,
        "possible_causes": list,
        "troubleshooting_steps": list,
        "risk_notes": list,
        "escalation_advice": str,
        "training_points": list,
        "uncertainty_note": str,
    },
    "training_support": {
        "training_goal": str,
        "key_points": list,
        "common_mistakes": list,
        "quiz_questions": list,
        "assessment_checklist": list,
        "coach_tip": str,
        "uncertainty_note": str,
    },
    "case_review": {
        "case_title": str,
        "root_cause_summary": str,
        "reusable_lessons": list,
        "sop_update_suggestions": list,
        "archive_tags": list,
        "missing_information": list,
        "review_note": str,
    },
}


def runtime_schema_fields() -> dict[str, set[str]]:
    """Read the production schema so dataset drift is caught before training."""
    from src.llm_client import CASE_REVIEW_SCHEMA_EXAMPLE, FAULT_SCHEMA_EXAMPLE, TRAINING_SCHEMA_EXAMPLE

    return {
        "fault_diagnosis": set(FAULT_SCHEMA_EXAMPLE),
        "training_support": set(TRAINING_SCHEMA_EXAMPLE),
        "case_review": set(CASE_REVIEW_SCHEMA_EXAMPLE),
    }

LIST_LIMITS = {
    "evidence_observations": 4,
    "possible_causes": 4,
    "troubleshooting_steps": 5,
    "risk_notes": 4,
    "training_points": 4,
    "key_points": 5,
    "common_mistakes": 4,
    "quiz_questions": 5,
    "assessment_checklist": 5,
    "reusable_lessons": 5,
    "sop_update_suggestions": 4,
    "archive_tags": 5,
    "missing_information": 4,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"第 {line_number} 行不是合法 JSON：{error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象。")
        record["_line_number"] = line_number
        records.append(record)
    return records


def validate_messages(messages: Any, example_id: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(messages, list) or len(messages) != 3:
        return [f"{example_id}: messages 必须恰好包含 system、user、assistant 三条消息。"]

    expected_roles = ["system", "user", "assistant"]
    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if roles != expected_roles:
        errors.append(f"{example_id}: messages 角色顺序必须是 {expected_roles}，实际为 {roles}。")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"{example_id}: 每条消息都必须有非空字符串 content。")
    return errors


def validate_assistant_output(task: str, content: str, example_id: str) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        return [f"{example_id}: assistant 输出不是合法 JSON：{error}"]

    if not isinstance(payload, dict):
        return [f"{example_id}: assistant 输出必须是 JSON 对象。"]

    schema = TASK_SCHEMAS[task]
    if set(payload) != set(schema):
        errors.append(
            f"{example_id}: 输出键不匹配。缺少={sorted(set(schema) - set(payload))}，多出={sorted(set(payload) - set(schema))}。"
        )

    for field_name, expected_type in schema.items():
        value = payload.get(field_name)
        if not isinstance(value, expected_type):
            errors.append(f"{example_id}: {field_name} 应为 {expected_type.__name__}。")
            continue
        if isinstance(value, str) and field_name != "matched_writeback_case_note" and not value.strip():
            errors.append(f"{example_id}: {field_name} 不得为空。")
        if isinstance(value, list):
            if not all(isinstance(item, str) and item.strip() for item in value):
                errors.append(f"{example_id}: {field_name} 必须是非空字符串列表。")
            if len(value) > LIST_LIMITS[field_name]:
                errors.append(f"{example_id}: {field_name} 超出 {LIST_LIMITS[field_name]} 条上限。")

    if task == "fault_diagnosis" and not str(payload.get("uncertainty_note", "")).strip():
        errors.append(f"{example_id}: 排障卡必须保留 uncertainty_note。")
    return errors


def validate_runtime_semantics(task: str, messages: Any, example_id: str) -> list[str]:
    """Check a small number of semantics that the current UI relies on."""
    if not isinstance(messages, list) or len(messages) != 3:
        return []
    try:
        payload = json.loads(str(messages[2].get("content", "")))
    except (AttributeError, json.JSONDecodeError):
        return []
    if task != "fault_diagnosis":
        return []
    case_note = str(payload.get("matched_writeback_case_note", "")).strip()
    user_content = str(messages[1].get("content", ""))
    if case_note and "前台案例回写（人工确认后归档）" not in user_content:
        return [
            f"{example_id}: matched_writeback_case_note 非空时，检索片段必须明确包含“前台案例回写（人工确认后归档）”。"
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="校验智维师 SFT JSONL 数据集。")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="待校验的 JSONL；默认校验 data/sft/seed_examples.jsonl。",
    )
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        print(f"SFT 数据集不存在：{dataset_path}")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    known_sources = {item["path"] for item in manifest.get("sources", [])}
    records = load_jsonl(dataset_path)
    errors: list[str] = []
    example_ids: set[str] = set()

    try:
        production_fields = runtime_schema_fields()
    except Exception as error:  # pragma: no cover - import errors should be visible in CLI output
        print(f"无法读取运行时 schema：{error}")
        return 1
    for task, schema in TASK_SCHEMAS.items():
        if set(schema) != production_fields[task]:
            errors.append(
                f"{task}: SFT schema 与 src/llm_client.py 不一致。"
                f"数据集={sorted(schema)}，运行时={sorted(production_fields[task])}。"
            )

    for record in records:
        line_number = record.pop("_line_number")
        example_id = str(record.get("id", "")).strip() or f"line-{line_number}"
        if example_id in example_ids:
            errors.append(f"{example_id}: 样本 ID 重复。")
        example_ids.add(example_id)

        task = record.get("task")
        if task not in TASK_SCHEMAS:
            errors.append(f"{example_id}: 未知 task={task!r}。")
            continue
        if record.get("split") not in {"train", "validation", "development", "test"}:
            errors.append(f"{example_id}: split 必须为 train、validation、development 或 test。")
        if record.get("review_status") not in {
            "draft",
            "evidence_checked_pending_sme",
            "sme_approved",
            "teacher_generated",
        }:
            errors.append(f"{example_id}: review_status 不合法。")

        source_mode = record.get("source_mode", "retrieved_corpus")
        if source_mode not in {"retrieved_corpus", "synthetic_teacher", "synthetic_runtime_template", "synthetic_runtime_v3"}:
            errors.append(f"{example_id}: source_mode 不合法：{source_mode!r}。")
        source_files = record.get("source_files")
        if source_mode == "retrieved_corpus":
            if not isinstance(source_files, list) or not source_files:
                errors.append(f"{example_id}: retrieved_corpus 的 source_files 必须为非空列表。")
                source_files = []
            unknown_sources = [source for source in source_files if source not in known_sources]
            if unknown_sources:
                errors.append(f"{example_id}: source_files 含未登记文件：{unknown_sources}。")
        elif source_files not in (None, []):
            errors.append(f"{example_id}: 合成样本不应填写 source_files。")

        messages = record.get("messages")
        errors.extend(validate_messages(messages, example_id))
        if isinstance(messages, list) and len(messages) == 3 and isinstance(messages[2], dict):
            errors.extend(validate_assistant_output(task, str(messages[2].get("content", "")), example_id))
            errors.extend(validate_runtime_semantics(task, messages, example_id))

    if errors:
        print("SFT 数据校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    split_counts = {
        split: sum(record.get("split") == split for record in records)
        for split in ("train", "validation", "development", "test")
        if any(record.get("split") == split for record in records)
    }
    task_counts = {task: sum(record.get("task") == task for record in records) for task in TASK_SCHEMAS}
    print(f"SFT 数据校验通过：{dataset_path.name}，{len(records)} 条样本，split={split_counts}，task={task_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
