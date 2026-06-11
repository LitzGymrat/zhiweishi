from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"


def _resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _normalize_embedding_provider(raw_value: str) -> str:
    value = raw_value.strip().lower()
    if value in {"dashscope", "qwen"}:
        return "qwen"
    if value == "":
        return "qwen"
    return value


def _normalize_app_mode(raw_value: str) -> str:
    value = str(raw_value or "full").strip().lower()
    if value in {"poc1", "poc1_readonly", "readonly", "read_only"}:
        return "poc1_readonly"
    return "full"


def _parse_bool(raw_value: str) -> bool:
    return str(raw_value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("deepseek_api_key", "DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("deepseek_base_url", "DEEPSEEK_BASE_URL"),
    )
    model_name: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("model_name", "DEEPSEEK_MODEL"),
    )
    embedding_provider: str = Field(
        default="qwen",
        validation_alias=AliasChoices("embedding_provider", "ZHIWEISHI_EMBEDDING_PROVIDER"),
    )
    local_embedding_name: str = Field(
        default="all-MiniLM-L6-v2",
        validation_alias=AliasChoices("local_embedding_name", "LOCAL_EMBEDDING_NAME"),
    )
    qwen_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("qwen_api_key", "QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    )
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("qwen_base_url", "QWEN_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    qwen_embedding_name: str = Field(
        default="text-embedding-v4",
        validation_alias=AliasChoices(
            "qwen_embedding_name",
            "QWEN_EMBEDDING_NAME",
            "ZHIWEISHI_EMBEDDING_MODEL",
        ),
    )
    Chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        validation_alias=AliasChoices(
            "Chroma_persist_dir",
            "chroma_persist_dir",
            "ZHIWEISHI_CHROMA_DIR",
        ),
    )
    bm25_persist_dir: str = Field(
        default="./data/bm25_index.json",
        validation_alias=AliasChoices("bm25_persist_dir", "BM25_PERSIST_DIR"),
    )
    chunk_size: int = Field(
        default=500,
        validation_alias=AliasChoices("chunk_size", "ZHIWEISHI_CHUNK_SIZE"),
    )
    chunk_overlap: int = Field(
        default=80,
        validation_alias=AliasChoices("chunk_overlap", "ZHIWEISHI_CHUNK_OVERLAP"),
    )
    top_k: int = Field(
        default=6,
        validation_alias=AliasChoices("top_k", "ZHIWEISHI_MAX_CONTEXT_CHUNKS"),
    )
    app_mode: str = Field(
        default="full",
        validation_alias=AliasChoices("app_mode", "APP_MODE", "ZHIWEISHI_APP_MODE"),
    )
    access_password: str = Field(
        default="",
        validation_alias=AliasChoices("access_password", "ACCESS_PASSWORD", "ZHIWEISHI_ACCESS_PASSWORD"),
    )
    demo_reset_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("demo_reset_enabled", "DEMO_RESET_ENABLED", "ZHIWEISHI_DEMO_RESET_ENABLED"),
    )

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        return _normalize_embedding_provider(str(value or "qwen"))

    @field_validator("app_mode", mode="before")
    @classmethod
    def normalize_app_mode(cls, value: str) -> str:
        return _normalize_app_mode(str(value or "full"))

    @field_validator("demo_reset_enabled", mode="before")
    @classmethod
    def normalize_demo_reset_enabled(cls, value: str | bool) -> bool:
        if isinstance(value, bool):
            return value
        return _parse_bool(str(value or ""))

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls()

    @property
    def deepseek_model(self) -> str:
        return self.model_name

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider == "local":
            return self.local_embedding_name
        return self.qwen_embedding_name

    @property
    def chroma_dir(self) -> Path:
        return _resolve_project_path(self.Chroma_persist_dir)

    @property
    def bm25_path(self) -> Path:
        return _resolve_project_path(self.bm25_persist_dir)

    @property
    def max_context_chunks(self) -> int:
        return self.top_k

    @property
    def is_poc1_readonly(self) -> bool:
        return self.app_mode == "poc1_readonly"

    @property
    def has_access_password(self) -> bool:
        return bool(self.access_password.strip())


def get_env_help_text() -> str:
    return "\n".join(
        [
            "deepseek_api_key=你的DeepSeek API Key",
            "deepseek_base_url=https://api.deepseek.com",
            "model_name=deepseek-chat",
            "qwen_api_key=你的Qwen/DashScope API Key",
            "qwen_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_embedding_name=text-embedding-v4",
            "embedding_provider=qwen 或 local",
            "local_embedding_name=all-MiniLM-L6-v2",
            "Chroma_persist_dir=./data/chroma_db",
            "bm25_persist_dir=./data/bm25_index.json",
            "chunk_size=500",
            "chunk_overlap=80",
            "top_k=6",
            "app_mode=full 或 poc1_readonly",
            "access_password=可选；对外共享时建议设置访问口令",
            "demo_reset_enabled=true 时显示演示重置按钮",
            "兼容旧变量：DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ZHIWEISHI_*",
        ]
    )
