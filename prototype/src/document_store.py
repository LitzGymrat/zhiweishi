from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IngestResult:
    document_count: int
    chunk_count: int


class MetadataStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.base_dir / "ingest_metadata.json"

    def read(self) -> dict:
        if not self.file_path.exists():
            return {"documents": 0, "chunks": 0, "index_config": {}}
        return json.loads(self.file_path.read_text(encoding="utf-8"))

    def write(self, document_count: int, chunk_count: int, index_config: dict | None = None) -> None:
        payload = {
            "documents": document_count,
            "chunks": chunk_count,
            "index_config": index_config or {},
        }
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def index_exists(self) -> bool:
        data = self.read()
        return data.get("documents", 0) > 0 and data.get("chunks", 0) > 0

    def document_count(self) -> int:
        return int(self.read().get("documents", 0))

    def chunk_count(self) -> int:
        return int(self.read().get("chunks", 0))

    def index_config(self) -> dict:
        value = self.read().get("index_config", {})
        return value if isinstance(value, dict) else {}

    def matches_index_config(self, expected_config: dict) -> bool:
        return self.index_config() == expected_config
