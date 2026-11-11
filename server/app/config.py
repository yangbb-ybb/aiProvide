"""全局配置。

优先级：环境变量 > server/.env > 代码默认值。
所有可调参数集中在这里，参数取值依据见 docs/项目逻辑规划.md §4。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .hf_env import configure_fastembed_cache, configure_hf_endpoint

# server/ 目录（app/ 的上一级）
SERVER_DIR = Path(__file__).resolve().parent.parent

# 导入本模块时就先把模型下载的运行时环境设好，保证任何入口（API / 评测脚本）
# 在 import fastembed 之前都拿到正确的镜像地址和缓存目录
configure_hf_endpoint()
configure_fastembed_cache()


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
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 512
    ollama_think: bool = False

    # ---------- 向量化 / 重排 ----------
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B-Q"
    vector_dim: int = 1024
    # Qwen3-Embedding 的查询指令前缀。留空则不加前缀（换别的模型时用）。
    embed_query_instruction: str = (
        "Given a question, retrieve relevant passages from a knowledge base "
        "that answer the question"
    )
    rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    rerank_enabled: bool = True

    # ---------- 检索 ----------
    retrieve_top_n: int = 50
    rrf_k: int = 60
    rerank_top_k: int = 5
    context_chunk_chars: int = 500

    # ---------- 分块 ----------
    chunk_target_chars: int = 500
    chunk_overlap_chars: int = 80

    # ---------- 入库安全限制 ----------
    upload_max_bytes: int = 50 * 1024 * 1024
    url_max_bytes: int = 5 * 1024 * 1024
    url_max_redirects: int = 3
    url_allowed_content_types: str = "text/html,application/xhtml+xml"

    # ---------- 日志 / 会话 ----------
    log_level: str = "INFO"
    log_max_bytes: int = 20 * 1024 * 1024
    log_backup_count: int = 10

    # ---------- 其它 ----------
    data_dir: str = "./data"
    cors_origins: str = "http://localhost:5173"
    # HuggingFace 模型下载镜像。直连 huggingface.co 会超时
    hf_endpoint: str = "https://hf-mirror.com"
    # 模型缓存目录。fastembed 默认放系统临时目录，重启就没了
    fastembed_cache_path: str = str(Path.home() / ".cache" / "fastembed")

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

    @property
    def conversation_db_path(self) -> Path:
        """会话、消息和接口审计记录使用的 SQLite 数据库。"""
        return self.data_path / "app.db"

    @property
    def log_path(self) -> Path:
        """应用结构化日志目录。"""
        return self.data_path / "logs"


@lru_cache
def get_settings() -> Settings:
    """获取配置单例（FastAPI 依赖注入用）。"""
    settings = Settings()
    # .env 里的 HF_ENDPOINT / FASTEMBED_CACHE_PATH 也要生效
    # （系统环境变量优先级更高，见 hf_env）
    configure_hf_endpoint(settings.hf_endpoint)
    configure_fastembed_cache(settings.fastembed_cache_path)
    return settings
