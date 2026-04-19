#!/usr/bin/env sh
set -eu

cd /app

if [ ! -d "./data/chroma_db" ] || [ ! -f "./data/bm25_index.json" ]; then
  echo "[entrypoint] no local index found, building knowledge base first..."
  python build_knowledge_base.py
else
  echo "[entrypoint] existing index detected, skipping rebuild."
fi

exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-8506}" \
  --server.headless true \
  --browser.gatherUsageStats false