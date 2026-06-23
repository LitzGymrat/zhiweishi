from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import AppConfig


BASE_DIR = Path(__file__).resolve().parents[1]
CASE_IMAGE_DIR = BASE_DIR / "data" / "case_images"
ALLOWED_IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
}
_IMAGE_ID_PATTERN = re.compile(r"图片编号：\s*([A-Za-z0-9_-]+)")
_STORAGE_REF_PATTERN = re.compile(r"原图相对路径：\s*`?([^`\n]+?)`?\s*$", re.MULTILINE)


class ImageEvidenceValidationError(ValueError):
    """Raised when an uploaded image cannot be safely archived."""


@dataclass(frozen=True, slots=True)
class PreparedImageUpload:
    image_id: str
    original_filename: str
    extension: str
    media_type: str
    content: bytes
    sha256: str
    width: int
    height: int


def _clean_text(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:limit]


def _clean_list(value: Any, *, limit: int, item_limit: int) -> list[str]:
    values = value if isinstance(value, list) else []
    items: list[str] = []
    seen: set[str] = set()
    for raw_item in values:
        item = _clean_text(raw_item, limit=item_limit)
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def prepare_image_upload(
    filename: str,
    content: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> PreparedImageUpload:
    """Validate, orient and strip metadata from a JPEG/PNG upload.

    The original filename is retained only for audit. The stored filename is
    generated from image_id, so user-controlled names never become paths.
    """

    if not content:
        raise ImageEvidenceValidationError(f"{filename or '图片'}为空。")
    if len(content) > max_bytes:
        raise ImageEvidenceValidationError(
            f"{filename or '图片'}超过上传上限（{max_bytes // (1024 * 1024)} MB）。"
        )
    try:
        with Image.open(BytesIO(content)) as probe:
            probe.verify()
        with Image.open(BytesIO(content)) as image:
            image_format = str(image.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ImageEvidenceValidationError("仅支持 JPEG 或 PNG 图片。")
            normalized_image = ImageOps.exif_transpose(image)
            width, height = normalized_image.size
            if width < 16 or height < 16:
                raise ImageEvidenceValidationError("图片尺寸过小，至少需要 16×16 像素。")
            if width * height > max_pixels:
                raise ImageEvidenceValidationError("图片像素数超过上限，请压缩或裁剪后重试。")

            extension, media_type = ALLOWED_IMAGE_FORMATS[image_format]
            normalized = BytesIO()
            if image_format == "JPEG":
                normalized_image.convert("RGB").save(normalized, format="JPEG", quality=95, optimize=True)
            else:
                normalized_image.save(normalized, format="PNG", optimize=True)
    except ImageEvidenceValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageEvidenceValidationError(f"{filename or '图片'}不是可读取的 JPEG/PNG 图片。") from error

    normalized_content = normalized.getvalue()
    if len(normalized_content) > max_bytes:
        raise ImageEvidenceValidationError(
            f"{filename or '图片'}标准化后超过上传上限，请压缩后重试。"
        )
    return PreparedImageUpload(
        image_id=f"IMG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8].upper()}",
        original_filename=Path(filename or "未命名图片").name,
        extension=extension,
        media_type=media_type,
        content=normalized_content,
        sha256=sha256(normalized_content).hexdigest(),
        width=width,
        height=height,
    )


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


class ImageEvidenceService:
    """Optional OpenAI-compatible vision extractor for case evidence only."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def extract(self, image: PreparedImageUpload, storage_ref: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "image_id": image.image_id,
            "original_filename": image.original_filename,
            "sha256": image.sha256,
            "media_type": image.media_type,
            "width": image.width,
            "height": image.height,
            "storage_ref": storage_ref,
            "human_review_required": True,
            "image_summary": "",
            "ocr_text": "",
            "observable_findings": [],
            "unreadable_or_missing": [],
        }
        if not self.config.has_vision_endpoint:
            return {
                **record,
                "extraction_status": "not_configured",
                "extraction_note": "未配置视觉模型；原图已归档，等待人工补充或配置视觉抽取。",
            }

        data_url = f"data:{image.media_type};base64,{base64.b64encode(image.content).decode('ascii')}"
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是工业维保图片证据抽取器。只描述图中可见事实和可读取文字，"
                    "不得判断根因、确认故障件、给出维修动作或安全处置指令。"
                    "无法从图片确认的内容必须放入 unreadable_or_missing。只返回 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请按此 schema 返回："
                            '{"image_summary":"","ocr_text":"","observable_findings":[""],'
                            '"unreadable_or_missing":[""]}。'
                            "observable_findings 只写可见现象，并在不确定时明确角度/清晰度限制。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ]
        try:
            with OpenAI(
                api_key=self.config.vision_api_key or "EMPTY",
                base_url=self.config.vision_base_url.rstrip("/"),
                timeout=self.config.vision_timeout_seconds,
            ) as client:
                response = client.chat.completions.create(
                    model=self.config.vision_model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=self.config.vision_max_tokens,
                )
            parsed = _parse_json_object(response.choices[0].message.content or "")
            if parsed is None:
                raise ValueError("视觉模型没有返回可解析 JSON。")
            return {
                **record,
                "extraction_status": "complete",
                "extraction_note": "视觉模型派生内容，仅作可见事实，须人工复核。",
                "image_summary": _clean_text(parsed.get("image_summary"), limit=500),
                "ocr_text": _clean_text(parsed.get("ocr_text"), limit=2000),
                "observable_findings": _clean_list(parsed.get("observable_findings"), limit=6, item_limit=360),
                "unreadable_or_missing": _clean_list(parsed.get("unreadable_or_missing"), limit=6, item_limit=240),
                "extractor": {"model": self.config.vision_model, "base_url": self.config.vision_base_url},
            }
        except Exception as error:
            return {
                **record,
                "extraction_status": "error",
                "extraction_note": "视觉抽取失败；原图已归档，不能把该图片作为已解析证据。",
                "extraction_error": _clean_text(error, limit=300),
            }


def persist_case_image_evidence(
    *,
    case_basename: str,
    device_name: str,
    uploads: list[PreparedImageUpload],
    service: ImageEvidenceService,
) -> list[dict[str, Any]]:
    case_directory = CASE_IMAGE_DIR / re.sub(r'[<>:"/\\|?*]+', "_", device_name.strip() or "未分类设备") / case_basename
    case_directory.mkdir(parents=True, exist_ok=True)
    evidence_records: list[dict[str, Any]] = []
    for upload in uploads:
        image_path = case_directory / f"{upload.image_id}{upload.extension}"
        image_path.write_bytes(upload.content)
        storage_ref = image_path.relative_to(BASE_DIR).as_posix()
        evidence_records.append(service.extract(upload, storage_ref))
    return evidence_records


def render_image_evidence_markdown(evidence_records: list[dict[str, Any]]) -> list[str]:
    if not evidence_records:
        return []
    lines = ["", "图片派生证据（仅作可见事实，须人工复核）："]
    for record in evidence_records:
        lines.extend(
            [
                f"- 图片编号：{record.get('image_id', '未记录')}",
                f"  - 原图相对路径：{record.get('storage_ref', '未记录')}",
                f"  - SHA-256：{record.get('sha256', '未记录')}",
                f"  - 抽取状态：{record.get('extraction_status', 'unknown')}",
                f"  - 抽取说明：{record.get('extraction_note', '须人工复核')}",
            ]
        )
        if record.get("image_summary"):
            lines.append(f"  - 图像说明：{record['image_summary']}")
        if record.get("ocr_text"):
            lines.append(f"  - OCR：{record['ocr_text']}")
        for finding in record.get("observable_findings", []):
            lines.append(f"  - 可见观察：{finding}")
        for item in record.get("unreadable_or_missing", []):
            lines.append(f"  - 不可判读/缺失：{item}")
    return lines


def extract_image_references_from_text(content: str, source_label: str) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    positions = list(_IMAGE_ID_PATTERN.finditer(content))
    for index, match in enumerate(positions):
        segment_end = positions[index + 1].start() if index + 1 < len(positions) else len(content)
        segment = content[match.end() : segment_end]
        storage_match = _STORAGE_REF_PATTERN.search(segment)
        if not storage_match:
            continue
        references.append(
            {
                "image_id": match.group(1),
                "storage_ref": storage_match.group(1).strip(),
                "source_label": source_label,
            }
        )
    return references


def resolve_stored_image(storage_ref: str) -> Path | None:
    try:
        candidate = (BASE_DIR / storage_ref).resolve()
        candidate.relative_to(CASE_IMAGE_DIR.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None
