from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.corpus_manager import build_corpus_summary, collect_corpus_documents
from src.rag_pipeline import RagPipeline


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local knowledge base for the 智维师 prototype.")
    parser.add_argument(
        "--source-dir",
        default="./data/documents",
        help="Directory containing source documents for ingestion.",
    )
    args = parser.parse_args()

    source_dir = (BASE_DIR / args.source_dir).resolve()
    if not source_dir.exists():
        raise SystemExit(f"文档目录不存在: {source_dir}")

    documents = collect_corpus_documents(source_dir)
    if not documents:
        raise SystemExit(f"未找到可导入文档: {source_dir}")

    pipeline = RagPipeline(AppConfig.from_env())
    result = pipeline.ingest_documents(documents)
    summary = build_corpus_summary(documents)

    print(f"文档目录: {source_dir}")
    print(f"导入文档数: {result.document_count}")
    print(f"生成文本块数: {result.chunk_count}")
    print(f"Chroma索引目录: {pipeline.config.chroma_dir}")
    print(f"BM25索引文件: {pipeline.config.bm25_path}")
    print("分类统计:")
    for category in summary["categories"]:
        print(f"- {category['label']}: {category['count']} 份文档")


if __name__ == "__main__":
    main()