from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SFT_DIR = ROOT / "data" / "sft"
SOURCE_PATH = SFT_DIR / "seed_examples.jsonl"
EXPORT_DIR = SFT_DIR / "export"


def load_records(source_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            records.append(json.loads(raw_line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出纯 messages 格式的智维师 SFT 数据。")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="要导出的 SFT JSONL；可重复传入多个文件，默认是 data/sft/seed_examples.jsonl。",
    )
    parser.add_argument(
        "--status",
        default="evidence_checked_pending_sme",
        help="仅导出指定审核状态的样本。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPORT_DIR,
        help="导出目录；默认是 data/sft/export。",
    )
    args = parser.parse_args()

    input_paths = [path.resolve() for path in args.input] or [SOURCE_PATH]
    missing_paths = [path for path in input_paths if not path.exists()]
    if missing_paths:
        parser.error(f"输入数据集不存在：{', '.join(str(path) for path in missing_paths)}")
    selected = [
        record
        for source_path in input_paths
        for record in load_records(source_path)
        if record.get("review_status") == args.status
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    known_splits = ("train", "validation", "development", "test")
    selected_splits = [split for split in known_splits if any(record.get("split") == split for record in selected)]
    for split in selected_splits:
        messages = [{"messages": record["messages"]} for record in selected if record.get("split") == split]
        target_path = output_dir / f"{split}.jsonl"
        write_jsonl(target_path, messages)
        print(f"{target_path.relative_to(ROOT)}: {len(messages)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
