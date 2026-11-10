"""FastAPI 依赖。"""

from __future__ import annotations

from .config import get_settings
from .llm.ollama import OllamaClient


def get_ollama() -> OllamaClient:
    """构造 Ollama 客户端。

    全项目只有 llm/ollama.py 与推理后端耦合；要换成 llama.cpp server，
    只需要改这一个文件（两者都是 OpenAI 兼容协议）。
    """
    s = get_settings()
    return OllamaClient(
        host=s.ollama_host,
        model=s.ollama_chat_model,
        keep_alive=s.ollama_keep_alive,
    )
