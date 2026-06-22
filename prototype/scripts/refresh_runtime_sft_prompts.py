from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_runtime_sft_refined import make_scenario, validate_refined_contract
from scripts.validate_sft_dataset import validate_assistant_output, validate_messages, validate_runtime_semantics
from src.llm_client import build_runtime_system_prompt, build_runtime_user_message


TARGET_FAMILIES = {"fault_no_retrieval_strict", "fault_no_retrieval_with_common_sense"}
EXPECTED_SPLITS = ("train", "development", "test")
ID_PATTERN = re.compile(r"^RT3R-(train|development|test)-(.+)-(\d{4})$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON：{error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象。")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def scenario_family(record: dict[str, Any]) -> str:
    scenario = record.get("scenario")
    metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
    family = metadata.get("scenario_type") if isinstance(metadata, dict) else None
    if not isinstance(family, str):
        raise ValueError(f"{record.get('id', 'unknown')}: 缺少 scenario.metadata.scenario_type。")
    return family


def rebuild_target_scenario(record: dict[str, Any]) -> dict[str, Any]:
    example_id = str(record.get("id", ""))
    match = ID_PATTERN.fullmatch(example_id)
    if not match:
        raise ValueError(f"{example_id}: 无法从样本 id 恢复原 scenario。")
    split, family, sequence = match.groups()
    rebuilt = make_scenario(family, int(sequence), split)
    if rebuilt["id"] != example_id:
        raise ValueError(f"{example_id}: 重建后的 scenario id 不一致。")
    return rebuilt


def rerender_record(record: dict[str, Any]) -> dict[str, Any]:
    refreshed = deepcopy(record)
    scenario = refreshed.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"{refreshed.get('id', 'unknown')}: 缺少 scenario。")
    task = str(refreshed.get("task", ""))
    payload = scenario.get("payload")
    contexts = scenario.get("context_items")
    if not isinstance(payload, dict) or not isinstance(contexts, list):
        raise ValueError(f"{refreshed.get('id', 'unknown')}: scenario 的 payload/context_items 不完整。")
    messages = refreshed.get("messages")
    if not isinstance(messages, list) or len(messages) != 3 or not isinstance(messages[2], dict):
        raise ValueError(f"{refreshed.get('id', 'unknown')}: 原 messages 不完整。")
    refreshed["messages"] = [
        {"role": "system", "content": build_runtime_system_prompt(task)},
        {"role": "user", "content": build_runtime_user_message(task, payload, contexts)},
        {"role": "assistant", "content": messages[2].get("content", "")},
    ]
    return refreshed


def validate_refreshed_record(record: dict[str, Any]) -> list[str]:
    example_id = str(record.get("id", "unknown"))
    task = str(record.get("task", ""))
    messages = record.get("messages")
    errors = validate_messages(messages, example_id)
    if isinstance(messages, list) and len(messages) == 3 and isinstance(messages[2], dict):
        errors.extend(validate_assistant_output(task, str(messages[2].get("content", "")), example_id))
        errors.extend(validate_runtime_semantics(task, messages, example_id))
    errors.extend(validate_refined_contract(record))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="按当前运行时模板重渲染 SFT prompt，并导出无检索重标队列。")
    parser.add_argument("--input", type=Path, action="append", required=True, help="现有 train/development/test 原始 JSONL；可重复传入。")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-relabel", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = [path.resolve() for path in args.input]
    missing = [path for path in paths if not path.exists()]
    if missing:
        parser.error(f"输入文件不存在：{', '.join(str(path) for path in missing)}")

    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in EXPECTED_SPLITS}
    for path in paths:
        for row in load_jsonl(path):
            split = row.get("split")
            if split not in by_split:
                raise ValueError(f"{row.get('id', path)}: 不支持的 split={split!r}。")
            by_split[split].append(row)
    missing_splits = [split for split, rows in by_split.items() if not rows]
    if missing_splits:
        parser.error(f"缺少 split：{', '.join(missing_splits)}")

    output_dir = args.output_dir.resolve()
    refreshed_by_split: dict[str, list[dict[str, Any]]] = {}
    requeue_by_split: dict[str, list[dict[str, Any]]] = {}
    validation_errors: list[str] = []
    for split, rows in by_split.items():
        refreshed: list[dict[str, Any]] = []
        requeue: list[dict[str, Any]] = []
        for row in rows:
            family = scenario_family(row)
            if family in TARGET_FAMILIES:
                scenario = rebuild_target_scenario(row)
                requeue.append(
                    {
                        "scenario": scenario,
                        "reason": ["运行时无检索策略已改为允许显式标注的行业常识，必须按原 scenario 重新教师标注。"],
                        "raw": None,
                    }
                )
                continue
            refreshed_record = rerender_record(row)
            record_errors = validate_refreshed_record(refreshed_record)
            validation_errors.extend(record_errors)
            refreshed.append(refreshed_record)
        refreshed_by_split[split] = sorted(refreshed, key=lambda item: item["id"])
        requeue_by_split[split] = sorted(requeue, key=lambda item: item["scenario"]["id"])

    requeue_total = sum(len(rows) for rows in requeue_by_split.values())
    if requeue_total != args.expected_relabel:
        validation_errors.append(f"无检索重标数不匹配：期望 {args.expected_relabel}，实际 {requeue_total}。")
    if validation_errors:
        print("prompt 刷新预检失败：")
        for error in validation_errors:
            print(f"- {error}")
        return 1

    manifest = {
        "policy": "runtime_no_retrieval_common_sense_v04",
        "relabel_total": requeue_total,
        "by_split": {
            split: {"rerendered": len(refreshed_by_split[split]), "requeue": len(requeue_by_split[split])}
            for split in EXPECTED_SPLITS
        },
    }
    for split in EXPECTED_SPLITS:
        write_jsonl(output_dir / f"{split}_base.jsonl", refreshed_by_split[split], overwrite=args.overwrite)
        write_jsonl(output_dir / f"{split}_no_retrieval_requeue.jsonl", requeue_by_split[split], overwrite=args.overwrite)
    manifest_path = output_dir / "prompt_refresh_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prompt 刷新准备完成：{json.dumps(manifest, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
