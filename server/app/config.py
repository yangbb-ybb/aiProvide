"""全局配置。

优先级：环境变量 > server/.env > 代码默认值。
所有可调参数集中在这里，参数取值依据见 docs/项目逻辑规划.md §4。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# server/ 目录（app/ 的上一级）
SERVER_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """服务配置。字段名对应环境变量名（大小写不敏感）。"""

    model_config = SettingsConfigDict(
        env_file=SERVER_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Ollama ----------
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3:4b"
    ollama_keep_alive: str = "5m"

    # ---------- 向量化 / 重排 ----------
    embed_model: str = "BAAI/bge-m3"
    vector_dim: int = 1024
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True

    # ---------- 检索 ----------
    retrieve_top_n: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 5
    context_chunk_chars: int = 800

    # ---------- 分块 ----------
    chunk_target_chars: int = 500
    chunk_overlap_chars: int = 80

    # ---------- 其它 ----------
    data_dir: str = "./data"
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        """允许跨域的前端地址列表。"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def data_path(self) -> Path:
        """运行时数据目录。相对路径按 server/ 解析，保证从任何 cwd 启动都一致。"""
        raw = Path(self.data_dir).expanduser()
        if not raw.is_absolute():
            raw = SERVER_DIR / raw
        return raw.resolve()

    @property
    def uploads_path(self) -> Path:
        """上传的原始文档目录。"""
        return self.data_path / "uploads"

    @property
    def lancedb_path(self) -> Path:
        """LanceDB 向量仓库目录。"""
        return self.data_path / "lancedb"

    @property
    def bm25_path(self) -> Path:
        """BM25 关键词索引目录。"""
        return self.data_path / "bm25"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例（FastAPI 依赖注入用）。"""
    return Settings()
