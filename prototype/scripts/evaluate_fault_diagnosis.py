from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig
from src.evaluation import (
    FaultCase,
    build_judge_messages,
    load_fault_diagnosis_cases,
    normalize_judgement,
    parse_json_object,
    sha256_file,
    summarize_judgements,
)
from src.openai_compatible import create_openai_compatible_client


DEFAULT_DATASET = ROOT / "data" / "sft" / "runtime_v3_refined" / "v04" / "test.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "evaluations" / "fault_diagnosis"
CANDIDATE_ORDER = ("base", "finetuned", "deepseek_v4_flash", "deepseek_v4_pro")
DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_V4_PRO_MODEL = "deepseek-v4-pro"


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def endpoint(label: str, *, model: str, base_url: str, api_key: str) -> dict[str, Any]:
    model, base_url, api_key = model.strip(), base_url.strip(), api_key.strip()
    if not model or not base_url:
        return {"id": label, "available": False, "reason": "未配置 model 或 base_url。", "model": model, "base_url": base_url}
    if not api_key:
        return {"id": label, "available": False, "reason": "未配置 api_key。", "model": model, "base_url": base_url}
    return {"id": label, "available": True, "model": model, "base_url": base_url.rstrip("/"), "api_key": api_key}


def candidate_endpoints(config: AppConfig) -> list[dict[str, Any]]:
    return [
        endpoint("base", model=config.eval_base_model, base_url=config.eval_base_base_url, api_key=config.eval_base_api_key),
        endpoint("finetuned", model=config.eval_finetuned_model, base_url=config.eval_finetuned_base_url, api_key=config.eval_finetuned_api_key),
        endpoint(
            "deepseek_v4_flash",
            model=DEEPSEEK_V4_FLASH_MODEL,
            base_url=config.deepseek_base_url,
            api_key=config.deepseek_api_key,
        ),
        endpoint(
            "deepseek_v4_pro",
            model=DEEPSEEK_V4_PRO_MODEL,
            base_url=config.deepseek_base_url,
            api_key=config.deepseek_api_key,
        ),
    ]


def judge_endpoint(config: AppConfig) -> dict[str, Any]:
    return endpoint("gemini_3_flash_judge", model=config.dmx_model, base_url=config.dmx_base_url, api_key=config.dmx_api_key)


def request_candidate(client: OpenAI, config: dict[str, Any], case: FaultCase, max_tokens: int) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=case.runtime_messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
    except Exception as first_error:
        try:
            response = client.chat.completions.create(
                model=config["model"], messages=case.runtime_messages, temperature=0, max_tokens=max_tokens
            )
            content = response.choices[0].message.content or ""
        except Exception as second_error:
            return {
                "status": "error",
                "error": f"JSON 模式失败：{first_error}；普通请求失败：{second_error}",
                "generation_elapsed_seconds": round(perf_counter() - started_at, 3),
            }
    parsed, parse_error = parse_json_object(content)
    return {
        "status": "ok",
        "response_text": content,
        "parsed_output": parsed,
        "parse_error": parse_error,
        "generation_elapsed_seconds": round(perf_counter() - started_at, 3),
    }


def summarize_generation_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_candidate: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        elapsed = row.get("generation_elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            by_candidate[str(row["candidate_id"])].append(float(elapsed))
    return {
        candidate_id: {
            "measured_requests": len(values),
            "total_seconds": round(sum(values), 3),
            "average_seconds": round(mean(values), 3),
            "min_seconds": round(min(values), 3),
            "max_seconds": round(max(values), 3),
        }
        for candidate_id, values in sorted(by_candidate.items())
        if values
    }


def request_judgement(client: OpenAI, config: dict[str, Any], case: FaultCase, generation: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    candidate_output = str(generation.get("response_text", ""))
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=build_judge_messages(case, candidate_output),
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "enabled"}, "user_id": "zhiweishi_fault_eval_judge"},
        )
        raw = response.choices[0].message.content or ""
    except Exception as error:
        return {"status": "error", "error": str(error)}
    payload, parse_error = parse_json_object(raw)
    if payload is None:
        return {"status": "error", "error": parse_error, "raw_judge_response": raw}
    judgement, validation_error = normalize_judgement(payload)
    if judgement is None:
        return {"status": "error", "error": validation_error, "raw_judge_response": raw}
    return {"status": "ok", "judgement": judgement}


def main() -> int:
    parser = argparse.ArgumentParser(description="对冻结 test 集的辅助排查任务执行四模型盲评。")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=500, help="候选生成与 judge 的统一并发数。")
    parser.add_argument("--candidate-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--execute", action="store_true", help="实际调用可用候选接口与 Gemini judge。")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers 必须为正数。")
    dataset = args.dataset.resolve()
    if not dataset.exists():
        parser.error(f"测试集不存在：{dataset}")
    cases = load_fault_diagnosis_cases(dataset)
    config = AppConfig.from_env()
    candidates = candidate_endpoints(config)
    judge = judge_endpoint(config)
    preview = {
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "task": "fault_diagnosis",
        "frozen_test_cases": len(cases),
        "candidates": [{key: value for key, value in item.items() if key != "api_key"} for item in candidates],
        "judge": {key: value for key, value in judge.items() if key != "api_key"},
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    if not args.execute:
        print("预览模式：未调用候选模型或 Gemini judge。传入 --execute 后仅评测冻结 test 的 50 条辅助排查样本。")
        return 0

    run_dir = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
        parser.error(f"输出目录已存在且非空：{run_dir}；如确认覆盖请传 --overwrite。")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**preview, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rubric": "Gemini 3 Flash blind judge; test set must not be used for training or prompt tuning."}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(run_dir / "cases.jsonl", [case.to_dict() for case in cases], overwrite=args.overwrite)

    available = {item["id"]: item for item in candidates if item["available"]}
    unavailable = {item["id"]: item for item in candidates if not item["available"]}
    clients = {
        candidate_id: create_openai_compatible_client(
            api_key=item["api_key"], base_url=item["base_url"], max_connections=args.workers
        )
        for candidate_id, item in available.items()
    }
    generations: list[dict[str, Any]] = []
    for candidate_id, item in unavailable.items():
        for case in cases:
            generations.append({"case_id": case.case_id, "candidate_id": candidate_id, "model": item.get("model", ""), "status": "unavailable", "error": item["reason"]})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(request_candidate, clients[candidate_id], item, case, args.candidate_max_tokens): (candidate_id, item, case)
                for candidate_id, item in available.items()
                for case in cases
            }
            for future in as_completed(futures):
                candidate_id, item, case = futures[future]
                result = future.result()
                generations.append({"case_id": case.case_id, "candidate_id": candidate_id, "model": item["model"], **result})
    finally:
        for client in clients.values():
            client.close()
    generations.sort(key=lambda item: (CANDIDATE_ORDER.index(item["candidate_id"]), item["case_id"]))
    write_jsonl(run_dir / "generations.jsonl", generations, overwrite=args.overwrite)

    judgements: list[dict[str, Any]] = []
    successful = [row for row in generations if row.get("status") == "ok"]
    case_map = {case.case_id: case for case in cases}
    if not judge["available"]:
        judgements = [{"case_id": row["case_id"], "candidate_id": row["candidate_id"], "status": "unavailable", "error": judge["reason"]} for row in successful]
    else:
        with create_openai_compatible_client(
            api_key=judge["api_key"], base_url=judge["base_url"], max_connections=args.workers
        ) as judge_client:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(request_judgement, judge_client, judge, case_map[row["case_id"]], row, args.judge_max_tokens): row
                    for row in successful
                }
                for future in as_completed(futures):
                    generation = futures[future]
                    result = future.result()
                    judgements.append({"case_id": generation["case_id"], "candidate_id": generation["candidate_id"], "judge_model": judge["model"], **result})
    judgements.sort(key=lambda item: (CANDIDATE_ORDER.index(item["candidate_id"]), item["case_id"]))
    write_jsonl(run_dir / "judgements.jsonl", judgements, overwrite=args.overwrite)
    summary = {
        **summarize_judgements(judgements),
        "generation_status_counts": {candidate_id: sum(row["candidate_id"] == candidate_id and row["status"] == "ok" for row in generations) for candidate_id in CANDIDATE_ORDER},
        "generation_timing_seconds": summarize_generation_timing(generations),
        "candidate_placeholders": [item["id"] for item in unavailable.values()],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"评测完成：输出={run_dir}；生成成功={len(successful)}；judge 成功={summary['scored_rows']}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
