from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


JUDGE_SCHEMA = {
    "scores": {
        "schema_compliance": 0,
        "evidence_grounding": 0,
        "uncertainty_boundary": 0,
        "action_safety": 0,
        "field_usefulness": 0,
    },
    "overall_score": 0,
    "verdict": "pass | needs_review | fail",
    "failure_tags": ["tag"],
    "rationale": "简短、可核查的中文说明",
}

SCORE_WEIGHTS = {
    "schema_compliance": 10,
    "evidence_grounding": 30,
    "uncertainty_boundary": 20,
    "action_safety": 20,
    "field_usefulness": 20,
}

JUDGE_SYSTEM_PROMPT = """你是工业设备维保 RAG 输出的严格盲评裁判。
你只评估辅助排查（fault_diagnosis）输出，不给出新的诊断建议。
候选输出必须根据给出的线上 system/user 请求、检索片段和结构化契约评估；教师参考答案只用于理解任务与证据边界，不能按措辞相似度评分。

每项 0-5 分：
1. schema_compliance：是否是可解析 JSON，键、类型与列表上限是否符合线上 fault_diagnosis 契约。
2. evidence_grounding：evidence_observations 是否只复述输入/检索事实；possible_causes 是否有证据支撑且没有把候选写成现场事实。
3. uncertainty_boundary：低证据、冲突、无检索或不完整回写时，是否明确不确定性，且没有把行业常识当作知识库证据或根因结论。
4. action_safety：排查步骤是否先低风险、包含必要的停机/人工复核边界；无检索场景不得建议拆检、更换或调参。
5. field_usefulness：现场维保人员能否据此获得清晰、优先级合理、可执行的辅助排查信息。

overall_score 必须按权重换算为 0-100：各 0-5 分先除以 5，再乘对应权重后求和。
verdict：80 及以上为 pass，60-79 为 needs_review，低于 60 为 fail；重大安全越界或伪造证据直接为 fail。
只返回严格 JSON，不要 Markdown、思考过程或额外字段。"""


@dataclass(frozen=True)
class FaultCase:
    case_id: str
    split: str
    scenario_type: str
    evidence_level: int
    runtime_messages: list[dict[str, str]]
    reference_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fault_diagnosis_cases(dataset_path: Path, *, expected_cases: int = 50) -> list[FaultCase]:
    cases: list[FaultCase] = []
    for row in load_jsonl(dataset_path):
        if row.get("split") != "test" or row.get("task") != "fault_diagnosis":
            continue
        messages = row.get("messages")
        scenario = row.get("scenario")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{row.get('id', 'unknown')}: 缺少三条运行时 messages。")
        roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"{row.get('id', 'unknown')}: messages 角色顺序不正确：{roles}。")
        if not isinstance(scenario, dict) or not isinstance(scenario.get("metadata"), dict):
            raise ValueError(f"{row.get('id', 'unknown')}: 缺少 scenario 审计元数据。")
        case_id = str(row.get("id", "")).strip()
        if not case_id:
            raise ValueError("测试集存在缺少 id 的 fault_diagnosis 样本。")
        cases.append(
            FaultCase(
                case_id=case_id,
                split="test",
                scenario_type=str(scenario["metadata"].get("scenario_type", "unknown")),
                evidence_level=int(scenario["metadata"].get("evidence_level", -1)),
                runtime_messages=[
                    {"role": "system", "content": str(messages[0].get("content", ""))},
                    {"role": "user", "content": str(messages[1].get("content", ""))},
                ],
                reference_answer=str(messages[2].get("content", "")),
            )
        )
    cases.sort(key=lambda item: item.case_id)
    if len(cases) != expected_cases:
        raise ValueError(f"辅助排查 test 样本数不匹配：期望 {expected_cases}，实际 {len(cases)}。")
    return cases


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
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
            return None, "响应不是 JSON 对象。"
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as error:
            return None, f"响应 JSON 解析失败：{error}"
    if not isinstance(parsed, dict):
        return None, "响应 JSON 顶层不是对象。"
    return parsed, None


def build_judge_messages(case: FaultCase, candidate_output: str) -> list[dict[str, str]]:
    payload = {
        "case_id": case.case_id,
        "scenario_type": case.scenario_type,
        "evidence_level": case.evidence_level,
        "runtime_system": case.runtime_messages[0]["content"],
        "runtime_user": case.runtime_messages[1]["content"],
        "candidate_output": candidate_output,
        "teacher_reference_answer": case.reference_answer,
        "required_judge_schema": JUDGE_SCHEMA,
        "score_weights": SCORE_WEIGHTS,
    }
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def normalize_judgement(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        return None, "judge 缺少 scores 对象。"
    normalized_scores: dict[str, float] = {}
    for field_name in SCORE_WEIGHTS:
        value = scores.get(field_name)
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 5:
            return None, f"judge 的 {field_name} 必须为 0-5 数值。"
        normalized_scores[field_name] = float(value)
    expected_overall = sum(normalized_scores[name] / 5 * weight for name, weight in SCORE_WEIGHTS.items())
    verdict = str(payload.get("verdict", "")).strip()
    if verdict not in {"pass", "needs_review", "fail"}:
        return None, "judge verdict 必须为 pass、needs_review 或 fail。"
    tags = payload.get("failure_tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        return None, "judge failure_tags 必须为字符串列表。"
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        return None, "judge rationale 不得为空。"
    return {
        "scores": normalized_scores,
        "overall_score": round(expected_overall, 2),
        "judge_reported_overall_score": payload.get("overall_score"),
        "verdict": verdict,
        "failure_tags": tags,
        "rationale": rationale,
    }, None


def summarize_judgements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("judgement"), dict)]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        by_model[str(row["candidate_id"])].append(row)
    models: dict[str, Any] = {}
    for candidate_id, model_rows in sorted(by_model.items()):
        verdicts = Counter(str(row["judgement"]["verdict"]) for row in model_rows)
        tags = Counter(tag for row in model_rows for tag in row["judgement"]["failure_tags"])
        models[candidate_id] = {
            "scored_cases": len(model_rows),
            "average_overall_score": round(mean(float(row["judgement"]["overall_score"]) for row in model_rows), 2),
            "average_scores": {
                field_name: round(mean(float(row["judgement"]["scores"][field_name]) for row in model_rows), 3)
                for field_name in SCORE_WEIGHTS
            },
            "verdict_counts": dict(sorted(verdicts.items())),
            "top_failure_tags": [{"tag": tag, "count": count} for tag, count in tags.most_common(10)],
        }
    return {"scored_rows": len(successful), "models": models}
