from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_runtime_sft_refined import (
    DEFAULT_MATRIX,
    build_scenarios,
    load_json,
    validate_refined_contract,
)
from scripts.validate_sft_dataset import TASK_SCHEMAS, validate_assistant_output, validate_messages, validate_runtime_semantics


CASE_ID_PATTERN = re.compile(r"\bAUTO-\d{8}-\d{6}\b")
EXPECTED_SPLITS = ("train", "development", "test")


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
        row["_source_path"] = str(path)
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def record_device(record: dict[str, Any]) -> str | None:
    scenario = record.get("scenario")
    payload = scenario.get("payload") if isinstance(scenario, dict) else None
    device = payload.get("device") if isinstance(payload, dict) else None
    return str(device).strip() if isinstance(device, str) and device.strip() else None


def record_case_ids(record: dict[str, Any]) -> set[str]:
    scenario = record.get("scenario")
    serialized = json.dumps(scenario, ensure_ascii=False) if isinstance(scenario, dict) else ""
    return set(CASE_ID_PATTERN.findall(serialized))


def append_cross_split_errors(values_by_split: dict[str, set[str]], label: str, errors: list[str]) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for split, values in values_by_split.items():
        for value in values:
            owners[value].add(split)
    for value, splits in sorted(owners.items()):
        if len(splits) > 1:
            errors.append(f"{label} 跨 split 重复：{value} 出现在 {sorted(splits)}。")


def expected_counts(matrix: dict[str, Any]) -> tuple[Counter[tuple[str, str]], Counter[tuple[str, str]], Counter[str]]:
    by_split_family: Counter[tuple[str, str]] = Counter()
    by_split_task: Counter[tuple[str, str]] = Counter()
    by_task: Counter[str] = Counter()
    for split, scenarios in matrix.items():
        if split not in EXPECTED_SPLITS or not isinstance(scenarios, dict):
            continue
        for family, count in scenarios.items():
            task = "fault_diagnosis" if family.startswith("fault_") else "case_review" if family.startswith("case_review_") else "training_support" if family.startswith("training_") else "unknown"
            by_split_family[(split, family)] += int(count)
            by_split_task[(split, task)] += int(count)
            by_task[task] += int(count)
    return by_split_family, by_split_task, by_task


def main() -> int:
    parser = argparse.ArgumentParser(description="对 refined runtime 成品执行全量终检。")
    parser.add_argument("--input", type=Path, action="append", required=True, help="一个或多个 refined 原始 JSONL；train/development/test 都必须提供。")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    args = parser.parse_args()

    input_paths = [path.resolve() for path in args.input]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        parser.error(f"输入文件不存在：{', '.join(str(path) for path in missing)}")

    try:
        matrix = load_json(args.matrix.resolve())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"无法读取场景矩阵：{error}")
    planned = {split: build_scenarios(matrix, split) for split in EXPECTED_SPLITS}
    expected_by_family, expected_by_split_task, expected_by_task = expected_counts(matrix)

    rows = [row for path in input_paths for row in load_jsonl(path)]
    errors: list[str] = []
    ids: set[str] = set()
    actual_by_family: Counter[tuple[str, str]] = Counter()
    actual_by_split_task: Counter[tuple[str, str]] = Counter()
    actual_by_task: Counter[str] = Counter()
    devices_by_split = {split: set() for split in EXPECTED_SPLITS}
    cases_by_split = {split: set() for split in EXPECTED_SPLITS}
    present_splits: Counter[str] = Counter()

    for row in rows:
        source = f"{row.pop('_source_path')}:{row.pop('_line_number')}"
        example_id = str(row.get("id", "")).strip() or source
        if example_id in ids:
            errors.append(f"{example_id}: 样本 id 重复。")
        ids.add(example_id)
        task = row.get("task")
        split = row.get("split")
        if task not in TASK_SCHEMAS:
            errors.append(f"{example_id}: 未知 task={task!r}。")
            continue
        if split not in EXPECTED_SPLITS:
            errors.append(f"{example_id}: refined split 必须为 train、development 或 test。")
            continue
        present_splits[split] += 1
        if row.get("review_status") != "teacher_generated":
            errors.append(f"{example_id}: review_status 必须为 teacher_generated。")
        if row.get("source_mode") != "synthetic_runtime_v3":
            errors.append(f"{example_id}: source_mode 必须为 synthetic_runtime_v3。")
        messages = row.get("messages")
        errors.extend(validate_messages(messages, example_id))
        if isinstance(messages, list) and len(messages) == 3 and isinstance(messages[2], dict):
            errors.extend(validate_assistant_output(task, str(messages[2].get("content", "")), example_id))
            errors.extend(validate_runtime_semantics(task, messages, example_id))
        errors.extend(validate_refined_contract(row))

        scenario = row.get("scenario")
        metadata = scenario.get("metadata") if isinstance(scenario, dict) else None
        family = metadata.get("scenario_type") if isinstance(metadata, dict) else None
        if not isinstance(family, str):
            errors.append(f"{example_id}: 缺少 scenario_type。")
        else:
            actual_by_family[(split, family)] += 1
        actual_by_split_task[(split, task)] += 1
        actual_by_task[task] += 1
        device = record_device(row)
        if device is None:
            errors.append(f"{example_id}: scenario.payload 缺少 device，无法进行切分泄漏检查。")
        else:
            devices_by_split[split].add(device)
        cases_by_split[split].update(record_case_ids(row))

    for split in EXPECTED_SPLITS:
        if not present_splits[split]:
            errors.append(f"缺少 split={split} 的输入样本。")
        if len(planned[split]) != sum(count for (actual_split, _), count in actual_by_family.items() if actual_split == split):
            errors.append(f"split={split} 总量不匹配：期望 {len(planned[split])}，实际 {sum(count for (actual_split, _), count in actual_by_family.items() if actual_split == split)}。")
    for key in sorted(set(expected_by_family) | set(actual_by_family)):
        if expected_by_family[key] != actual_by_family[key]:
            errors.append(f"场景配额不匹配：split={key[0]}，{key[1]} 期望 {expected_by_family[key]}，实际 {actual_by_family[key]}。")
    for key in sorted(set(expected_by_split_task) | set(actual_by_split_task)):
        if expected_by_split_task[key] != actual_by_split_task[key]:
            errors.append(f"split 任务配额不匹配：split={key[0]}，task={key[1]} 期望 {expected_by_split_task[key]}，实际 {actual_by_split_task[key]}。")
    for task in sorted(set(expected_by_task) | set(actual_by_task)):
        if expected_by_task[task] != actual_by_task[task]:
            errors.append(f"总任务配额不匹配：task={task} 期望 {expected_by_task[task]}，实际 {actual_by_task[task]}。")
    append_cross_split_errors(devices_by_split, "设备", errors)
    append_cross_split_errors(cases_by_split, "案例编号", errors)

    if errors:
        print("refined runtime 终检失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    split_counts = {split: present_splits[split] for split in EXPECTED_SPLITS}
    task_counts = dict(sorted(actual_by_task.items()))
    print(f"refined runtime 终检通过：{len(rows)} 条，split={split_counts}，task={task_counts}，跨 split 设备/案例编号无交集。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
