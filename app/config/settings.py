"""Application settings loaded from environment variables for FINO."""

from __future__ import annotations

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Typed settings for FINO runtime configuration."""

    env: str = Field("local", env="ENV")
    debug: bool = Field(False, env="DEBUG")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.3, env="OPENAI_TEMPERATURE")
    openai_timeout: float = Field(30.0, env="OPENAI_TIMEOUT")
    rag_backend: str = Field("stub", env="RAG_BACKEND")
    rag_top_k: int = Field(3, env="RAG_TOP_K")
    rag_score_threshold: float = Field(0.45, env="RAG_SCORE_THRESHOLD")
    law_scope_dsl_path: str = Field("./data/law_scope_dsl.json", env="LAW_SCOPE_DSL_PATH")
    guide_index_path: str = Field("./data/guides/", env="GUIDE_INDEX_PATH")
    mcp_enabled: bool = Field(False, env="MCP_ENABLED")
    mcp_endpoint: str = Field("http://localhost:9000", env="MCP_ENDPOINT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    backbone_enabled: bool = Field(False, env="BACKBONE_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
