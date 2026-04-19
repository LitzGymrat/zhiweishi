from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{1}|[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(slots=True)
class SparseHit:
    chunk_id: str
    score: float


class SimpleBM25:
    def __init__(self, chunk_texts: dict[str, str]) -> None:
        self.chunk_ids = list(chunk_texts.keys())
        self.documents = {chunk_id: tokenize(text) for chunk_id, text in chunk_texts.items()}
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.term_freq: dict[str, Counter[str]] = {}
        self.doc_len: dict[str, int] = {}
        self.avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        total_length = 0
        for chunk_id, tokens in self.documents.items():
            frequencies = Counter(tokens)
            self.term_freq[chunk_id] = frequencies
            self.doc_len[chunk_id] = len(tokens)
            total_length += len(tokens)
            for term in frequencies:
                self.doc_freq[term] += 1
        self.avg_doc_len = total_length / max(len(self.documents), 1)

    def search(self, query: str, top_k: int = 6, k1: float = 1.5, b: float = 0.75) -> list[SparseHit]:
        query_terms = tokenize(query)
        results: list[SparseHit] = []
        doc_count = max(len(self.documents), 1)

        for chunk_id in self.chunk_ids:
            score = 0.0
            frequencies = self.term_freq.get(chunk_id, Counter())
            doc_len = self.doc_len.get(chunk_id, 0)
            for term in query_terms:
                if term not in frequencies:
                    continue
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / max(self.avg_doc_len, 1e-6))
                score += idf * numerator / denominator
            if score > 0:
                results.append(SparseHit(chunk_id=chunk_id, score=score))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]


def dump_sparse_records(file_path: Path, chunk_records: list[dict[str, str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunks": chunk_records}
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_sparse_records(file_path: Path) -> list[dict[str, str]]:
    if not file_path.exists():
        return []

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    records = payload.get("chunks", [])
    return [
        record
        for record in records
        if isinstance(record, dict) and record.get("id") and record.get("content")
    ]
