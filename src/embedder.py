"""语义记忆引擎 — Embedding 模型封装"""
from __future__ import annotations

import os

# 必须在导入 torch 之前设置，避免多线程 segfault（macOS / 非 PTY 环境）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Embedder:
    """轻量中文 embedding 模型封装，全 CPU 运行。

    使用 BAAI/bge-small-zh-v1.5（~90MB），512 维向量。
    模型在首次调用时懒加载，支持多进程复用。
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self._model_name = model_name
        self._model = None
        self._dim = 512

    # ── 懒加载 ──
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # 强制 CPU，避免 M4 Mac 上偶尔触发 MPS 异常
            os.environ["SENTENCE_TRANSFORMERS_NO_DEVICE"] = ""
            self._model = SentenceTransformer(
                self._model_name, device="cpu"
            )
            self._model.eval()
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    # ── 编码 ──
    def encode(
        self,
        texts: str | List[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> "NDArray[np.float32]":
        """将文本编码为归一化向量。

        Args:
            texts: 单条文本或文本列表
            normalize: 是否 L2 归一化（FAISS 内积模式需要）
            batch_size: 批次大小

        Returns:
            shape=(n, dim) 的 float32 numpy 数组
        """
        if isinstance(texts, str):
            texts = [texts]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
            device="cpu",
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_single(self, text: str) -> "NDArray[np.float32]":
        """编码单条文本，返回 1D 向量。"""
        vec = self.encode(text)
        return vec[0]

    # ── 相似度 ──
    @staticmethod
    def cosine_similarity(a: "NDArray[np.float32]", b: "NDArray[np.float32]") -> float:
        """计算两个已归一化向量的余弦相似度（即内积）。"""
        return float(np.dot(a, b))

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "lazy"
        return f"Embedder(model={self._model_name}, dim={self._dim}, status={loaded})"
