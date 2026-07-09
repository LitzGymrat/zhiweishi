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
    if value in {"local", "vllm", "local_vllm"}:
        return "local"
    if value in {"hash", "local_hash"}:
        return "hash"
    if value == "":
        return "qwen"
    return value


def _normalize_app_mode(raw_value: str) -> str:
    value = str(raw_value or "full").strip().lower()
    if value in {"poc1", "poc1_readonly", "readonly", "read_only"}:
        return "poc1_readonly"
    return "full"


def _normalize_runtime_provider(raw_value: str) -> str:
    value = str(raw_value or "online").strip().lower()
    if value in {"local", "on_premise", "on-premise"}:
        return "local"
    return "online"


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
    dmx_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("dmx_api_key", "DMX_API_KEY"),
    )
    dmx_base_url: str = Field(
        default="https://www.dmxapi.cn/v1",
        validation_alias=AliasChoices("dmx_base_url", "DMX_BASE_URL"),
    )
    dmx_model: str = Field(
        default="gemini-3-flash-preview",
        validation_alias=AliasChoices("dmx_model", "DMX_MODEL"),
    )
    vision_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("vision_api_key", "VISION_API_KEY"),
    )
    vision_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("vision_base_url", "VISION_BASE_URL"),
    )
    vision_model: str = Field(
        default="",
        validation_alias=AliasChoices("vision_model", "VISION_MODEL"),
    )
    vision_max_tokens: int = Field(
        default=1024,
        validation_alias=AliasChoices("vision_max_tokens", "VISION_MAX_TOKENS"),
    )
    vision_timeout_seconds: float = Field(
        default=60.0,
        validation_alias=AliasChoices("vision_timeout_seconds", "VISION_TIMEOUT_SECONDS"),
    )
    image_max_upload_mb: int = Field(
        default=10,
        validation_alias=AliasChoices("image_max_upload_mb", "IMAGE_MAX_UPLOAD_MB"),
    )
    image_max_pixels: int = Field(
        default=20_000_000,
        validation_alias=AliasChoices("image_max_pixels", "IMAGE_MAX_PIXELS"),
    )
    eval_base_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("eval_base_api_key", "EVAL_BASE_API_KEY"),
    )
    eval_base_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("eval_base_base_url", "EVAL_BASE_BASE_URL"),
    )
    eval_base_model: str = Field(
        default="",
        validation_alias=AliasChoices("eval_base_model", "EVAL_BASE_MODEL"),
    )
    eval_finetuned_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("eval_finetuned_api_key", "EVAL_FINETUNED_API_KEY"),
    )
    eval_finetuned_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("eval_finetuned_base_url", "EVAL_FINETUNED_BASE_URL"),
    )
    eval_finetuned_model: str = Field(
        default="",
        validation_alias=AliasChoices("eval_finetuned_model", "EVAL_FINETUNED_MODEL"),
    )
    model_name: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("model_name", "DEEPSEEK_MODEL"),
    )
    runtime_provider: str = Field(
        default="online",
        validation_alias=AliasChoices("runtime_provider", "ZHIWEISHI_RUNTIME_PROVIDER"),
    )
    local_llm_base_url: str = Field(
        default="http://127.0.0.1:8001/v1",
        validation_alias=AliasChoices("local_llm_base_url", "LOCAL_LLM_BASE_URL"),
    )
    local_llm_api_key: str = Field(
        default="EMPTY",
        validation_alias=AliasChoices("local_llm_api_key", "LOCAL_LLM_API_KEY"),
    )
    local_llm_model: str = Field(
        default="finetuned",
        validation_alias=AliasChoices("local_llm_model", "LOCAL_LLM_MODEL"),
    )
    embedding_provider: str = Field(
        default="qwen",
        validation_alias=AliasChoices("embedding_provider", "ZHIWEISHI_EMBEDDING_PROVIDER"),
    )
    local_embedding_name: str = Field(
        default="qwen3-embedding-0.6b",
        validation_alias=AliasChoices("local_embedding_name", "LOCAL_EMBEDDING_NAME"),
    )
    local_embedding_base_url: str = Field(
        default="http://127.0.0.1:8002/v1",
        validation_alias=AliasChoices("local_embedding_base_url", "LOCAL_EMBEDDING_BASE_URL"),
    )
    local_embedding_api_key: str = Field(
        default="EMPTY",
        validation_alias=AliasChoices("local_embedding_api_key", "LOCAL_EMBEDDING_API_KEY"),
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

    @field_validator("runtime_provider", mode="before")
    @classmethod
    def normalize_runtime_provider(cls, value: str) -> str:
        return _normalize_runtime_provider(str(value or "online"))

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
    def generation_base_url(self) -> str:
        return self.local_llm_base_url if self.runtime_provider == "local" else self.deepseek_base_url

    @property
    def generation_api_key(self) -> str:
        return self.local_llm_api_key if self.runtime_provider == "local" else self.deepseek_api_key

    @property
    def generation_model(self) -> str:
        return self.local_llm_model if self.runtime_provider == "local" else self.deepseek_model

    @property
    def generation_provider_label(self) -> str:
        return "本地 LoRA 模型" if self.runtime_provider == "local" else "DeepSeek 在线模型"

    @property
    def embedding_model(self) -> str:
        if self.embedding_provider == "local":
            return self.local_embedding_name
        if self.embedding_provider == "hash":
            return "local-hash-256"
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

    @property
    def has_vision_endpoint(self) -> bool:
        return bool(self.vision_base_url.strip() and self.vision_model.strip())

    @property
    def image_max_upload_bytes(self) -> int:
        return max(self.image_max_upload_mb, 1) * 1024 * 1024


def get_env_help_text() -> str:
    return "\n".join(
        [
            "deepseek_api_key=你的DeepSeek API Key",
            "deepseek_base_url=https://api.deepseek.com",
            "model_name=deepseek-chat",
            "runtime_provider=online 或 local",
            "local_llm_base_url=http://127.0.0.1:8001/v1（Docker 中改为 http://host.docker.internal:8001/v1）",
            "local_llm_model=finetuned",
            "local_llm_api_key=EMPTY",
            "dmx_api_key=你的DMXAPI Key（仅用于 SFT 教师模型）",
            "dmx_base_url=https://www.dmxapi.cn/v1",
            "dmx_model=gemini-3-flash-preview",
            "vision_api_key=可选；视觉模型接口密钥（本地接口可填 EMPTY）",
            "vision_base_url=可选；OpenAI-compatible 视觉接口地址",
            "vision_model=可选；视觉模型 ID。留空时图片仅归档、不发送到外部接口",
            "vision_max_tokens=1024",
            "vision_timeout_seconds=60",
            "image_max_upload_mb=10",
            "image_max_pixels=20000000",
            "eval_base_base_url=http://127.0.0.1:8000/v1（未部署时留空）",
            "eval_base_model=基础模型服务的模型 ID",
            "eval_base_api_key=本地兼容接口可填 EMPTY",
            "eval_finetuned_base_url=http://127.0.0.1:8001/v1（未部署时留空）",
            "eval_finetuned_model=微调模型服务的模型 ID",
            "eval_finetuned_api_key=本地兼容接口可填 EMPTY",
            "DeepSeek 评测候选统一复用 deepseek_api_key 与 deepseek_base_url",
            "qwen_api_key=你的Qwen/DashScope API Key",
            "qwen_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1",
            "qwen_embedding_name=text-embedding-v4",
            "embedding_provider=qwen、local 或 hash（仅开发回退）",
            "local_embedding_base_url=http://127.0.0.1:8002/v1（Docker 中改为 http://host.docker.internal:8002/v1）",
            "local_embedding_name=qwen3-embedding-0.6b",
            "local_embedding_api_key=EMPTY",
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
