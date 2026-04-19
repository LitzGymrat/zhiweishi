from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CASE_OUTPUT_DIR = BASE_DIR / "data" / "case_output"
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
DEFAULT_DEVICE_NAME = "未分类设备"
LEGACY_AUTO_WRITEBACK_SOURCE = "来源：前台案例回写自动归档"
REVIEWED_WRITEBACK_SOURCE = "来源：前台案例回写（人工确认后归档）"
AUTO_WRITEBACK_FILE_PATTERN = re.compile(r"^case_\d{8}_\d{6}\.md$", re.IGNORECASE)
PLACEHOLDER_VALUES = {"待补充", "未填写", "无", "n/a", "na"}
MIN_CASE_TEXT_LENGTHS = {
    "actual_steps": 8,
    "final_result": 6,
    "experience_summary": 6,
}

CATEGORY_LABELS = {
    "manuals": "设备说明书",
    "sop": "SOP",
    "inspection_logs": "点检日志",
    "maintenance_logs": "维保日志",
    "repair_records": "维修记录",
    "case_cards": "案例卡",
    "training_cards": "培训资料",
    "uploaded": "上传文档",
}

CATEGORY_ORDER = [
    "manuals",
    "sop",
    "inspection_logs",
    "maintenance_logs",
    "repair_records",
    "case_cards",
    "training_cards",
    "uploaded",
]


class CaseWritebackValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def normalize_path_component(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "未命名"


def _normalize_single_line_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_multiline_text(value: Any) -> str:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    return "\n".join(lines)


def _normalize_list(values: Any) -> list[str]:
    items = values if isinstance(values, list) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _normalize_single_line_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _is_placeholder_text(value: str, minimum_length: int = 1) -> bool:
    normalized = _normalize_single_line_text(value)
    if not normalized:
        return True
    if normalized.lower() in PLACEHOLDER_VALUES:
        return True
    if re.fullmatch(r"[0-9一二三四五六七八九十]+[.、]?", normalized):
        return True
    return len(normalized) < minimum_length


def is_auto_writeback_source_label(source_label: str) -> bool:
    return bool(AUTO_WRITEBACK_FILE_PATTERN.match(Path(str(source_label or "")).name))


def filter_case_evidence_sources(evidence_sources: Any) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for item in evidence_sources if isinstance(evidence_sources, list) else []:
        source_label = Path(str(item or "")).name
        if not source_label or is_auto_writeback_source_label(source_label):
            continue
        if source_label in seen:
            continue
        seen.add(source_label)
        filtered.append(source_label)
    return filtered


def is_legacy_auto_writeback_document(file_path: Path) -> bool:
    if file_path.suffix.lower() not in {".md", ".txt"}:
        return False
    if not AUTO_WRITEBACK_FILE_PATTERN.match(file_path.name):
        return False
    context = infer_document_context(file_path)
    if context["category_key"] != "case_cards":
        return False
    try:
        preview = file_path.read_text(encoding="utf-8", errors="ignore")[:512]
    except OSError:
        return False
    return LEGACY_AUTO_WRITEBACK_SOURCE in preview


def prepare_case_writeback_payload(payload: dict[str, Any]) -> dict[str, Any]:
    prepared_payload = {
        **payload,
        "device": _normalize_single_line_text(payload.get("device", DEFAULT_DEVICE_NAME)) or DEFAULT_DEVICE_NAME,
        "symptom": _normalize_single_line_text(payload.get("symptom", "")),
        "actual_steps": _normalize_multiline_text(payload.get("actual_steps", "")),
        "final_result": _normalize_multiline_text(payload.get("final_result", "")),
        "experience_summary": _normalize_multiline_text(payload.get("experience_summary", "")),
        "operator": _normalize_single_line_text(payload.get("operator", "")),
        "possible_causes": _normalize_list(payload.get("possible_causes", [])),
        "recommended_steps": _normalize_list(payload.get("recommended_steps", [])),
        "evidence_sources": filter_case_evidence_sources(payload.get("evidence_sources", [])),
    }

    errors: list[str] = []
    for field_name, label in (
        ("device", "维修设备"),
        ("symptom", "触发症状"),
        ("operator", "责任操作人"),
    ):
        if _is_placeholder_text(str(prepared_payload.get(field_name, ""))):
            errors.append(f"{label}不能为空。")

    for field_name, label in (
        ("actual_steps", "实际确认与处理步骤"),
        ("final_result", "最终处理结论"),
        ("experience_summary", "经验总结"),
    ):
        minimum_length = MIN_CASE_TEXT_LENGTHS[field_name]
        if _is_placeholder_text(str(prepared_payload.get(field_name, "")), minimum_length=minimum_length):
            errors.append(f"{label}需要补充完整，不能留空、不能写“待补充”，也不能只填序号。")

    if errors:
        raise CaseWritebackValidationError(errors)

    return prepared_payload


def collect_corpus_documents(source_dir: Path | None = None) -> list[Path]:
    root_dir = source_dir or DOCUMENTS_DIR
    if not root_dir.exists():
        return []

    return sorted(
        [
            file_path
            for file_path in root_dir.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_SUFFIXES
            and not is_legacy_auto_writeback_document(file_path)
        ],
        key=lambda item: str(item).lower(),
    )


def infer_document_context(file_path: Path) -> dict[str, str]:
    try:
        relative_path = file_path.resolve().relative_to(DOCUMENTS_DIR.resolve())
        parts = relative_path.parts
    except ValueError:
        parts = file_path.parts

    category_key = parts[0] if parts else "uploaded"
    category_label = CATEGORY_LABELS.get(category_key, "未分类文档")
    device_name = parts[1] if len(parts) >= 3 else DEFAULT_DEVICE_NAME
    return {
        "category_key": category_key,
        "category_label": category_label,
        "device_name": device_name,
    }


def build_corpus_summary(document_paths: list[Path] | None = None) -> dict[str, Any]:
    paths = document_paths or collect_corpus_documents(DOCUMENTS_DIR)
    records: list[dict[str, str]] = []
    devices: set[str] = set()

    for file_path in paths:
        context = infer_document_context(file_path)
        devices.add(context["device_name"])
        records.append(
            {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "device_name": context["device_name"],
                "category_key": context["category_key"],
                "category_label": context["category_label"],
            }
        )

    ordered_categories = CATEGORY_ORDER + sorted(
        {record["category_key"] for record in records if record["category_key"] not in CATEGORY_ORDER}
    )

    categories: list[dict[str, Any]] = []
    for category_key in ordered_categories:
        category_records = [record for record in records if record["category_key"] == category_key]
        if not category_records:
            continue
        categories.append(
            {
                "key": category_key,
                "label": CATEGORY_LABELS.get(category_key, category_key),
                "count": len(category_records),
                "records": category_records,
            }
        )

    known_devices = {device for device in devices if device != DEFAULT_DEVICE_NAME}
    return {
        "total_documents": len(records),
        "device_count": len(known_devices),
        "category_count": len(categories),
        "categories": categories,
    }


def resolve_document_dir(category_key: str, device_name: str | None = None) -> Path:
    target_dir = DOCUMENTS_DIR / category_key
    if device_name:
        target_dir = target_dir / normalize_path_component(device_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def render_case_writeback_markdown(payload: dict[str, Any]) -> str:
    possible_causes = payload.get("possible_causes", [])
    recommended_steps = payload.get("recommended_steps", [])
    evidence_sources = payload.get("evidence_sources", [])
    actual_steps = str(payload.get("actual_steps", "")).strip()
    final_result = str(payload.get("final_result", "")).strip()
    experience_summary = str(payload.get("experience_summary", "")).strip()

    lines = [
        f"案例编号：AUTO-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        f"设备：{payload.get('device', DEFAULT_DEVICE_NAME)}",
        "文档类别：案例卡",
        REVIEWED_WRITEBACK_SOURCE,
        f"处理人：{payload.get('operator', '未填写') or '未填写'}",
        f"回写时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "故障现象：",
        str(payload.get("symptom", "未填写")),
        "",
        "系统建议可能原因：",
    ]

    if possible_causes:
        lines.extend([f"- {item}" for item in possible_causes])
    else:
        lines.append("- 无")

    lines.extend(["", "系统建议排查步骤："])
    if recommended_steps:
        lines.extend([f"{index}. {item}" for index, item in enumerate(recommended_steps, start=1)])
    else:
        lines.append("1. 无")

    lines.extend(
        [
            "",
            "实际处理步骤：",
            actual_steps,
            "",
            "最终结论：",
            final_result,
            "",
            "经验总结：",
            experience_summary,
            "",
            "依据文档：",
        ]
    )

    if evidence_sources:
        lines.extend([f"- {item}" for item in evidence_sources])
    else:
        lines.append("- 无")

    return "\n".join(lines).strip() + "\n"


def save_case_writeback(payload: dict[str, Any]) -> dict[str, Path]:
    prepared_payload = prepare_case_writeback_payload(payload)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = str(prepared_payload.get("device", DEFAULT_DEVICE_NAME) or DEFAULT_DEVICE_NAME)
    knowledge_dir = resolve_document_dir("case_cards", device_name)
    audit_dir = CASE_OUTPUT_DIR / normalize_path_component(device_name)
    audit_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"case_{timestamp}"
    audit_json_path = audit_dir / f"{base_name}.json"
    knowledge_doc_path = knowledge_dir / f"{base_name}.md"

    payload_with_timestamp = {
        **prepared_payload,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "knowledge_doc_path": str(knowledge_doc_path),
    }
    audit_json_path.write_text(
        json.dumps(payload_with_timestamp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    knowledge_doc_path.write_text(
        render_case_writeback_markdown(payload_with_timestamp),
        encoding="utf-8",
    )
    return {
        "audit_json_path": audit_json_path,
        "knowledge_doc_path": knowledge_doc_path,
    }