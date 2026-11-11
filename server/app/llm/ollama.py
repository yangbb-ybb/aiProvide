"""Ollama 客户端。

全项目唯一与推理后端耦合的文件。生成走 Ollama 的原生 /api/chat 接口
（不是 OpenAI 兼容层），因为 Ollama 的 options 字段能直接控制采样参数。

换推理后端（llama.cpp server / vLLM）时，只需要改这个文件。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class OllamaError(RuntimeError):
    """Ollama 调用失败。message 里带可操作的中文提示，不要直接抛给用户看英文异常。"""


class OllamaClient:
    """Ollama HTTP 客户端。"""

    def __init__(
        self,
        host: str,
        model: str,
        keep_alive: str = "30m",
        num_ctx: int = 4096,
        num_predict: int = 512,
        think: bool = False,
        timeout: float = 60.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.think = think
        self._timeout = timeout

    async def list_models(self) -> list[str]:
        """返回本机已安装的模型名列表。"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise OllamaError(
                "Ollama 未启动，请执行 `ollama serve`（或 `brew services start ollama`）"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"访问 Ollama 失败：{exc}") from exc
        return [m.get("name", "") for m in data.get("models", [])]

    async def chat_stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        top_p: float = 0.8,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[str]:
        """流式生成，逐段 yield 增量文本。

        temperature 默认 0.1：RAG 是「提取 + 改写」任务不是创作任务，
        低温度更忠于资料原文。

        keep_alive 默认 5m：M1 上模型冷启动要 3~10s，常驻避免每次重来。
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": num_ctx or self.num_ctx,
                "num_predict": num_predict or self.num_predict,
            },
            "think": self.think if think is None else think,
        }
        try:
            # 刻意不用 `async with a, b:` 合并写法：
            # 合并后两个 __aexit__ 的异常传播路径会变，出问题时不好排查
            async with httpx.AsyncClient(timeout=self._timeout) as client:  # noqa: SIM117
                async with client.stream("POST", f"{self.host}/api/chat", json=payload) as resp:
                    if resp.status_code == 404:
                        raise OllamaError(f"模型 {self.model} 不存在，请执行 `ollama pull {self.model}`")
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise OllamaError(f"Ollama 返回错误：{chunk['error']}")
                        message = chunk.get("message") or {}
                        piece = message.get("content")
                        if piece:
                            yield piece
        except httpx.ConnectError as exc:
            raise OllamaError(
                "Ollama 未启动，请执行 `ollama serve`（或 `brew services start ollama`）"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"Ollama 返回 HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"调用 Ollama 失败：{exc}") from exc
