"""向量化。

fastembed + Qwen/Qwen3-Embedding-0.6B-Q，1024 维，ONNX int8 推理，不需要 PyTorch。
模型缓存在 ~/.cache/fastembed（约 1.1GB）。

选型说明（详见规划文档 §8.2）：
- 原本想用 BAAI/bge-m3，但 **fastembed 0.9.1 并不支持它**（只有 bge-base-en / bge-small-zh
  等几个英文或小尺寸中文模型）。Qwen3-Embedding-0.6B-Q 是这个列表里
  「中文强 + 1024 维 + 体积可控」的唯一最优解，而且和生成模型同属 Qwen3 一家。

要点：
- **查询要加指令前缀**：`Instruct: {任务描述}\nQuery:{问题}`，文档不加。
  这是 Qwen3-Embedding 的官方用法，不加会明显掉点。
- 输出已经做过 L2 归一化（fastembed 内部 normalize 过），这里再归一一次是无害的保险
- 归一化后余弦相似度 = 点积，所以 LanceDB 索引用 metric="cosine"
- fastembed 是同步 CPU 密集调用，在 async 场景里要用 asyncio.to_thread 包起来
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 批量大小：16GB 机器上 32 是稳定值，再大容易把内存打满
_BATCH_SIZE = 32

# Qwen3-Embedding 的查询指令。任务描述要跟实际用途对上：
# 我们是「从个人知识库里检索能回答问题的段落」，不是网页搜索。
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question, retrieve relevant passages from a knowledge base "
    "that answer the question"
)


class Embedder:
    """Qwen3-Embedding 向量化器。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B-Q",
        dim: int = 1024,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.query_instruction = query_instruction
        self._model = None

    def _ensure_model(self):
        """懒加载：第一次调用时才下载 / 加载模型。"""
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("正在加载向量化模型：%s（首次运行需要下载，之后走本地缓存）", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量向量化，返回 (n, dim) 的 L2 归一化矩阵。batch=32 在 16GB 机器上稳。"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure_model()
        vectors = np.asarray(list(model.embed(texts, batch_size=_BATCH_SIZE)), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            logger.warning(
                "模型 %s 输出的维度是 %d，与配置的 VECTOR_DIM=%d 不一致，按实际维度继续",
                self.model_name,
                vectors.shape[1],
                self.dim,
            )
            self.dim = vectors.shape[1]
        return _l2_normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        """查询向量化，返回 (dim,) 的 L2 归一化向量。

        查询会加上 `Instruct: ...\\nQuery:` 前缀，文档不加 —— 这是 Qwen3-Embedding 的官方约定。
        """
        return self.embed_documents([self._prefix_query(text)])[0]

    def _prefix_query(self, text: str) -> str:
        """给查询加指令前缀。没有指令就原样返回（兼容不需要前缀的模型）。"""
        if not self.query_instruction:
            return text
        return f"Instruct: {self.query_instruction}\nQuery:{text}"


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化。零向量保持为零，避免除零产生 nan。"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
