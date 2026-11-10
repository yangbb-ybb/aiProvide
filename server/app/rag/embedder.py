"""向量化。

fastembed + BAAI/bge-m3，1024 维，ONNX 推理，不需要 PyTorch。
模型缓存在 ~/.cache/fastembed（约 1.2GB）。

要点：
- bge-m3 查询**不需要**加前缀（bge-v1.5 才需要「为这个句子生成表示以用于检索相关文章：」）
- 归一化后余弦相似度 = 点积，所以 LanceDB 索引用 metric="cosine"
- fastembed 是同步 CPU 密集调用，在 async 场景里要用 asyncio.to_thread 包起来
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """bge-m3 向量化器。"""

    def __init__(self, model_name: str = "BAAI/bge-m3", dim: int = 1024) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _ensure_model(self):
        """懒加载：第一次调用时才下载 / 加载模型。"""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """批量向量化，返回 (n, dim) 的 L2 归一化矩阵。batch=32 在 16GB 机器上稳。"""
        raise NotImplementedError("M1 阶段实现：文档向量化")

    def embed_query(self, text: str) -> np.ndarray:
        """查询向量化，返回 (dim,) 的 L2 归一化向量。"""
        raise NotImplementedError("M1 阶段实现：查询向量化")
