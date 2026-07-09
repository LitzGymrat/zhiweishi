from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from openai import OpenAI

from src.config import AppConfig


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{1}|[a-zA-Z0-9_]+")


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class QwenEmbeddingClient:
    def __init__(self, config: AppConfig) -> None:
        if not config.qwen_api_key:
            raise ValueError("qwen_api_key 未配置，无法使用 Qwen embedding。")
        self.client = OpenAI(api_key=config.qwen_api_key, base_url=config.qwen_base_url)
        self.model = config.embedding_model
        self.batch_size = 10

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


class LocalOpenAIEmbeddingClient:
    """Embedding client for a locally served OpenAI-compatible embedding model."""

    def __init__(self, config: AppConfig) -> None:
        if not config.local_embedding_base_url.strip():
            raise ValueError("local_embedding_base_url 未配置，无法调用本地 embedding 服务。")
        if not config.local_embedding_name.strip():
            raise ValueError("local_embedding_name 未配置，无法调用本地 embedding 服务。")
        self.client = OpenAI(
            api_key=config.local_embedding_api_key or "EMPTY",
            base_url=config.local_embedding_base_url,
        )
        self.model = config.local_embedding_name
        self.batch_size = 10

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


class LocalHashEmbeddingClient:
    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dims
            sign = -1.0 if int(digest[8:10], 16) % 2 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]


def build_embedding_client(config: AppConfig) -> EmbeddingClient:
    if config.embedding_provider == "qwen":
        return QwenEmbeddingClient(config)
    if config.embedding_provider == "local":
        return LocalOpenAIEmbeddingClient(config)
    if config.embedding_provider == "hash":
        return LocalHashEmbeddingClient()
    raise ValueError(f"不支持的 embedding_provider：{config.embedding_provider}")
