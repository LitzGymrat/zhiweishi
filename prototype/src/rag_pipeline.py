from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings

from src.config import AppConfig
from src.corpus_manager import collect_corpus_documents
from src.demo_data import build_training_questions, find_case
from src.document_store import IngestResult, MetadataStore
from src.embeddings import build_embedding_client
from src.hybrid_retriever import SimpleBM25, dump_sparse_records, load_sparse_records
from src.llm_client import DeepSeekFaultReasoner
from src.loaders import load_document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [segment.strip() for segment in text.split("\n") if segment.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current}\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + chunk_size, len(paragraph))
            chunks.append(paragraph[start:end])
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = end
            start = next_start
        current = ""

    if current:
        chunks.append(current)
    return chunks


def _deduplicate_items(items: list[str], limit: int) -> list[str]:
    unique_items: list[str] = []
    seen: set[str] = set()
    ignored_items = {
        "根因判断",
        "故障现象",
        "处理经过",
        "经验总结",
        "日常维护基线",
        "常见异常",
        "系统建议可能原因",
        "系统建议排查步骤",
        "依据文档",
    }
    for item in items:
        normalized = item.strip().strip("- ").strip()
        if len(normalized) < 4:
            continue
        if normalized in ignored_items or normalized.endswith("案例"):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(normalized)
        if len(unique_items) >= limit:
            break
    return unique_items


def _extract_lines_by_keywords(retrieved_chunks: list[dict], keywords: list[str], limit: int) -> list[str]:
    candidates: list[str] = []
    for chunk in retrieved_chunks:
        for raw_line in str(chunk.get("content", "")).splitlines():
            line = raw_line.strip().lstrip("#")
            line = line.lstrip("0123456789.、- ").strip()
            if any(keyword in line for keyword in keywords):
                candidates.append(line)
    return _deduplicate_items(candidates, limit=limit)


def _build_fault_query(device_name: str, symptom: str) -> str:
    return f"设备：{device_name}\n故障现象：{symptom}"


def _normalize_device_name(device_name: str) -> str:
    return "".join(str(device_name or "").strip().split()).lower()


class RagPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.metadata = MetadataStore(config.chroma_dir)
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl="src.chroma_telemetry.NoOpProductTelemetry",
            ),
        )
        self.collection = self.chroma_client.get_or_create_collection(name="zhiweishi_chunks")
        self.embedding_client = build_embedding_client(config)
        self.index_config = {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }
        if self.metadata.index_exists() and not self.metadata.matches_index_config(self.index_config):
            self.rebuild_default_corpus()
        self.reasoner = None
        if config.deepseek_api_key:
            try:
                self.reasoner = DeepSeekFaultReasoner(config)
            except Exception:
                self.reasoner = None
        self._bm25: SimpleBM25 | None = None
        self._chunk_cache: dict[str, dict[str, Any]] = {}
        persisted_chunk_records = load_sparse_records(config.bm25_path)
        if persisted_chunk_records:
            self._refresh_sparse_index(persisted_chunk_records)
        else:
            self._refresh_sparse_index()

    def _recreate_collection(self) -> None:
        self.collection = self.chroma_client.get_or_create_collection(name="zhiweishi_chunks")
        existing = self.collection.get()
        ids = existing.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)

    def _refresh_sparse_index(self, chunk_records: list[dict[str, Any]] | None = None) -> None:
        if chunk_records is None:
            documents = self.collection.get(include=["documents", "metadatas"])
            chunk_records = [
                {
                    "id": chunk_id,
                    "content": content,
                    "source_label": metadata.get("source_label", "未知来源"),
                    "path": metadata.get("path", ""),
                    "doc_category": metadata.get("doc_category", "未分类文档"),
                    "device_name": metadata.get("device_name", "未分类设备"),
                }
                for chunk_id, content, metadata in zip(
                    documents.get("ids", []),
                    documents.get("documents", []),
                    documents.get("metadatas", []),
                    strict=False,
                )
            ]

        doc_texts: dict[str, str] = {}
        self._chunk_cache = {}
        for record in chunk_records:
            chunk_id = str(record["id"])
            content = str(record["content"])
            doc_texts[chunk_id] = content
            self._chunk_cache[chunk_id] = {
                "id": chunk_id,
                "content": content,
                "source_label": str(record.get("source_label", "未知来源")),
                "path": str(record.get("path", "")),
                "doc_category": str(record.get("doc_category", "未分类文档")),
                "device_name": str(record.get("device_name", "未分类设备")),
            }
        self._bm25 = SimpleBM25(doc_texts) if doc_texts else None

    def rebuild_default_corpus(self, source_dir: Path | None = None) -> IngestResult:
        document_paths = collect_corpus_documents(source_dir)
        return self.ingest_documents(document_paths)

    def ingest_documents(self, document_paths: list[Path]) -> IngestResult:
        self._recreate_collection()

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        chunk_records: list[dict[str, str]] = []

        valid_documents = 0
        for document_path in document_paths:
            raw_document = load_document(document_path)
            chunks = chunk_text(
                raw_document.content,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            if not chunks:
                continue
            valid_documents += 1
            for index, chunk in enumerate(chunks, start=1):
                chunk_id = str(uuid4())
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source_label": raw_document.source_label,
                        "path": str(raw_document.path),
                        "chunk_index": index,
                        "doc_category": raw_document.doc_category,
                        "device_name": raw_document.device_name,
                    }
                )
                chunk_records.append(
                    {
                        "id": chunk_id,
                        "content": chunk,
                        "source_label": raw_document.source_label,
                        "path": str(raw_document.path),
                        "doc_category": raw_document.doc_category,
                        "device_name": raw_document.device_name,
                    }
                )

        if documents:
            embeddings = self.embedding_client.embed_texts(documents)
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        dump_sparse_records(self.config.bm25_path, chunk_records)
        self.metadata.write(
            document_count=valid_documents,
            chunk_count=len(documents),
            index_config=self.index_config,
        )
        self._refresh_sparse_index(chunk_records)
        return IngestResult(document_count=valid_documents, chunk_count=len(documents))

    def _dense_retrieve(self, query: str, top_k: int) -> dict[str, float]:
        chunk_count = self.metadata.chunk_count()
        if chunk_count == 0:
            return {}

        n_results = min(top_k, chunk_count)
        query_embedding = self.embedding_client.embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances"],
        )
        scores: dict[str, float] = {}
        for chunk_id, distance in zip(result.get("ids", [[]])[0], result.get("distances", [[]])[0], strict=False):
            scores[chunk_id] = 1.0 / (1.0 + float(distance))
        return scores

    def _sparse_retrieve(self, query: str, top_k: int) -> dict[str, float]:
        if not self._bm25:
            return {}
        hits = self._bm25.search(query, top_k=top_k)
        if not hits:
            return {}
        max_score = max(item.score for item in hits) or 1.0
        return {item.chunk_id: item.score / max_score for item in hits}

    def hybrid_retrieve(self, query: str, top_k: int) -> list[dict]:
        dense_scores = self._dense_retrieve(query, top_k=top_k)
        sparse_scores = self._sparse_retrieve(query, top_k=top_k)
        candidate_ids = set(dense_scores) | set(sparse_scores)

        results: list[dict] = []
        for chunk_id in candidate_ids:
            record = self._chunk_cache.get(chunk_id)
            if not record:
                continue
            hybrid_score = 0.65 * dense_scores.get(chunk_id, 0.0) + 0.35 * sparse_scores.get(chunk_id, 0.0)
            results.append(
                {
                    **record,
                    "dense_score": dense_scores.get(chunk_id, 0.0),
                    "sparse_score": sparse_scores.get(chunk_id, 0.0),
                    "hybrid_score": hybrid_score,
                }
            )
        results.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return results[:top_k]

    def _resolve_retrieved_chunks(
        self,
        device_name: str,
        symptom: str,
        retrieved_chunks: list[dict] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        if retrieved_chunks:
            return retrieved_chunks

        effective_top_k = top_k if top_k and top_k > 0 else self.config.top_k
        candidate_chunks = self.hybrid_retrieve(
            _build_fault_query(device_name, symptom),
            top_k=min(max(effective_top_k * 3, effective_top_k), max(self.config.max_context_chunks, effective_top_k * 3)),
        )
        normalized_device = _normalize_device_name(device_name)
        if not normalized_device:
            return candidate_chunks[:effective_top_k]

        exact_matches = [
            chunk
            for chunk in candidate_chunks
            if _normalize_device_name(chunk.get("device_name", "")) == normalized_device
        ]
        partial_matches = [
            chunk
            for chunk in candidate_chunks
            if chunk not in exact_matches and normalized_device in _normalize_device_name(chunk.get("device_name", ""))
        ]
        if exact_matches or partial_matches:
            prioritized = exact_matches + partial_matches
            return prioritized[:effective_top_k]
        return candidate_chunks[:effective_top_k]

    def _build_fallback_result(self, device_name: str, symptom: str, retrieved_chunks: list[dict]) -> dict:
        if retrieved_chunks:
            evidence_observations = _extract_lines_by_keywords(
                retrieved_chunks,
                keywords=["液位", "油位", "压差", "异响", "振动", "温升", "日志", "记录", "滤芯"],
                limit=4,
            )
            possible_causes = _extract_lines_by_keywords(
                retrieved_chunks,
                keywords=["可能", "根因", "导致", "进气", "堵塞", "磨损", "异常"],
                limit=4,
            )
            troubleshooting_steps = _extract_lines_by_keywords(
                retrieved_chunks,
                keywords=["检查", "更换", "紧固", "补充", "复核", "执行", "停机"],
                limit=5,
            )
            risk_notes = _extract_lines_by_keywords(
                retrieved_chunks,
                keywords=["不得", "不宜", "停机", "风险", "禁止", "超过"],
                limit=3,
            )
            training_points = _extract_lines_by_keywords(
                retrieved_chunks,
                keywords=["经验", "优先", "应先", "总结", "不得"],
                limit=3,
            )
            evidence_sources = _deduplicate_items(
                [str(item.get("source_label", "未知来源")) for item in retrieved_chunks],
                limit=5,
            )

            return {
                "device": device_name,
                "symptom": symptom,
                "summary": "已根据检索片段生成保守版排障结论，建议按先易后难顺序执行。",
                "evidence_observations": evidence_observations
                or ["检索片段覆盖了说明书、SOP、日志和案例卡中的相关信息。"],
                "possible_causes": possible_causes
                or ["根据检索片段，建议优先排查供给状态、连接状态和易损件状态。"],
                "troubleshooting_steps": troubleshooting_steps
                or [
                    "先按检索到的说明书和SOP核对基础状态。",
                    "再根据日志和维修记录按先易后难的顺序排查。",
                    "若无法定位原因，停止高风险运行并转人工复核。",
                ],
                "risk_notes": risk_notes
                or ["当前为无模型兜底模式，建议严格按检索依据执行并保留人工复核。"],
                "escalation_advice": "当前未启用 DeepSeek，已根据维保知识库检索结果生成兜底排障建议。",
                "training_points": training_points
                or ["优先复用案例卡、维修记录和SOP中的已沉淀经验。"],
                "evidence_sources": evidence_sources,
                "uncertainty_note": "当前为无模型兜底模式，未被片段覆盖的内容需人工确认。",
                "retrieved_chunks": retrieved_chunks,
            }

        demo_case = find_case(device_name, symptom)
        if demo_case is None:
            sources = [item["source_label"] for item in retrieved_chunks[:3]] or ["当前知识库检索结果"]
            return {
                "device": device_name,
                "symptom": symptom,
                "summary": "当前检索命中有限，建议按基础排查模板保守处理。",
                "evidence_observations": ["当前知识库中尚未找到足够直接对应的维保片段。"],
                "possible_causes": [
                    "当前检索结果提示需先检查相关部件的供给状态和连接状态",
                    "若存在持续异响或波动，应重点排查磨损、堵塞或松动问题",
                ],
                "troubleshooting_steps": [
                    "先核对对应SOP和点检规范，确认基础状态是否异常",
                    "根据检索到的手册与历史记录，按先易后难的顺序排查",
                    "若无法定位原因，停止高风险运行并转人工复核",
                ],
                "risk_notes": [
                    "证据不足时不建议继续高负荷运行",
                    "若异常伴随温升、振动或异响，应优先考虑保护性停机",
                ],
                "escalation_advice": "当前未启用 DeepSeek 或匹配案例不足，建议结合检索依据进行人工确认。",
                "training_points": [
                    "先看基础状态，再查关键连接和易损部件",
                    "排障时要记录文档依据和处理过程，便于后续沉淀",
                ],
                "evidence_sources": sources,
                "uncertainty_note": "当前案例匹配度有限，建议人工复核后再执行高风险操作。",
                "retrieved_chunks": retrieved_chunks,
            }

        return {
            **demo_case,
            "device": device_name,
            "symptom": symptom,
            "retrieved_chunks": retrieved_chunks,
        }

    def answer_fault_question(self, device_name: str, symptom: str, top_k: int | None = None) -> dict:
        retrieved_chunks = self._resolve_retrieved_chunks(device_name, symptom, top_k=top_k)
        if self.reasoner and retrieved_chunks:
            try:
                result = self.reasoner.build_fault_card(device_name, symptom, retrieved_chunks)
                result["device"] = device_name
                result["symptom"] = symptom
                result["evidence_sources"] = _deduplicate_items(
                    [item["source_label"] for item in retrieved_chunks],
                    limit=6,
                )
                result["retrieved_chunks"] = retrieved_chunks
                return result
            except Exception:
                pass
        return self._build_fallback_result(device_name, symptom, retrieved_chunks)

    def _build_training_fallback(self, fault_result: dict, retrieved_chunks: list[dict]) -> dict:
        device_name = str(fault_result.get("device", "当前设备"))
        symptom = str(fault_result.get("symptom", "当前故障"))
        evidence_sources = _deduplicate_items(
            [str(item.get("source_label", "未知来源")) for item in retrieved_chunks],
            limit=6,
        )
        return {
            "device": device_name,
            "symptom": symptom,
            "training_goal": f"让新人掌握“{symptom}”场景下的首检顺序和停机边界。",
            "key_points": fault_result.get("training_points", [])[:4]
            or ["先核对依据，再做判断，再执行动作。"],
            "common_mistakes": [
                "未查说明书和SOP就直接拆检。",
                "发现异响或波动后仍然带故障高负荷运行。",
                "排障结束后未沉淀处理依据和最终结论。",
            ],
            "quiz_questions": build_training_questions(
                {
                    "device": device_name,
                    "symptom": symptom,
                }
            ),
            "assessment_checklist": [
                "能说清首检顺序和主要风险边界。",
                "能指出至少两条依据文档或历史案例来源。",
                "能说明何时需要停机并转人工复核。",
            ],
            "coach_tip": "带教时要求新人先复述依据，再复述判断和动作，不接受无依据猜测。",
            "uncertainty_note": "当前培训卡为兜底模式，仍应以现场制度文件为准。",
            "evidence_sources": evidence_sources,
            "retrieved_chunks": retrieved_chunks,
        }

    def build_training_support(self, fault_result: dict) -> dict:
        device_name = str(fault_result.get("device", ""))
        symptom = str(fault_result.get("symptom", ""))
        retrieved_chunks = self._resolve_retrieved_chunks(
            device_name,
            symptom,
            retrieved_chunks=fault_result.get("retrieved_chunks"),
        )
        if self.reasoner:
            try:
                result = self.reasoner.build_training_card(fault_result, retrieved_chunks)
                result["device"] = device_name
                result["symptom"] = symptom
                result["evidence_sources"] = _deduplicate_items(
                    [item["source_label"] for item in retrieved_chunks],
                    limit=6,
                )
                result["retrieved_chunks"] = retrieved_chunks
                return result
            except Exception:
                pass
        return self._build_training_fallback(fault_result, retrieved_chunks)

    def _build_case_review_fallback(self, payload: dict, retrieved_chunks: list[dict]) -> dict:
        device_name = str(payload.get("device", "当前设备"))
        symptom = str(payload.get("symptom", "当前故障"))
        possible_causes = [str(item) for item in payload.get("possible_causes", []) if str(item).strip()]
        recommended_steps = [str(item) for item in payload.get("recommended_steps", []) if str(item).strip()]
        experience_summary = str(payload.get("experience_summary", "")).strip()
        missing_information: list[str] = []
        for field_name, label in (
            ("actual_steps", "实际处理步骤"),
            ("final_result", "最终结论"),
            ("experience_summary", "经验总结"),
            ("operator", "处理人"),
        ):
            if not str(payload.get(field_name, "")).strip():
                missing_information.append(f"缺少{label}")

        return {
            "device": device_name,
            "symptom": symptom,
            "case_title": f"{device_name}_{symptom}_处理复盘",
            "root_cause_summary": str(payload.get("final_result", "")).strip()
            or "当前已记录处理结果，建议结合依据文档做人工复核。",
            "reusable_lessons": _deduplicate_items(
                [experience_summary, *possible_causes[:2], *recommended_steps[:2]],
                limit=5,
            )
            or ["同类故障应先核对基础状态，再判断是否需要停机。"],
            "sop_update_suggestions": [
                "将本次首检顺序补充到对应 SOP 或点检卡。",
                "把高频异常信号和停机边界同步到培训材料。",
            ],
            "archive_tags": _deduplicate_items(
                [device_name, symptom, *possible_causes[:2], "案例回写"],
                limit=5,
            ),
            "missing_information": missing_information or ["建议增加人工审核记录后再升级为标准案例。"],
            "review_note": "当前复盘建议基于已有输入和知识库依据生成，正式发布前建议做一次班组长审核。",
            "evidence_sources": _deduplicate_items(
                [str(item.get("source_label", "未知来源")) for item in retrieved_chunks],
                limit=6,
            ),
            "retrieved_chunks": retrieved_chunks,
        }

    def review_case_writeback(self, payload: dict, retrieved_chunks: list[dict] | None = None) -> dict:
        device_name = str(payload.get("device", ""))
        symptom = str(payload.get("symptom", ""))
        context_chunks = self._resolve_retrieved_chunks(
            device_name,
            symptom,
            retrieved_chunks=retrieved_chunks,
        )
        if self.reasoner:
            try:
                result = self.reasoner.build_case_review(payload, context_chunks)
                result["device"] = device_name
                result["symptom"] = symptom
                result["evidence_sources"] = _deduplicate_items(
                    [item["source_label"] for item in context_chunks],
                    limit=6,
                )
                result["retrieved_chunks"] = context_chunks
                return result
            except Exception:
                pass
        return self._build_case_review_fallback(payload, context_chunks)
