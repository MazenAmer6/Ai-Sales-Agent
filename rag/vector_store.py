"""
A minimal vector store: embeddings + metadata kept in memory as numpy arrays,
persisted to disk as a single .npz + a JSON sidecar for text/metadata. Search
is brute-force cosine similarity, which is completely fine at the scale of a
store's FAQ/policy knowledge base (dozens to low thousands of chunks) and
needs nothing beyond numpy — no compiled C++ extension, so it installs
cleanly on any machine, including Windows without a C++ build toolchain.

This intentionally exposes the same shape of methods a "real" vector DB
client would (add/delete/search by id, with metadata), so swapping in Chroma,
FAISS, or a hosted vector DB later only means changing this one file.
"""
import os
import json
import numpy as np


class SimpleVectorStore:
    def __init__(self, persist_dir: str, embeddings):
        """`embeddings` is any object exposing embed_documents(list[str]) ->
        list[list[float]] and embed_query(str) -> list[float], matching
        LangChain's Embeddings interface (e.g. HuggingFaceEmbeddings)."""
        self.persist_dir = persist_dir
        self.embeddings = embeddings
        os.makedirs(persist_dir, exist_ok=True)
        self._vectors_path = os.path.join(persist_dir, "vectors.npz")
        self._meta_path = os.path.join(persist_dir, "meta.json")
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None  # shape (n, dim)
        self._contents: list[str] = []
        self._metadatas: list[dict] = []
        self._load()

    # ---------------- persistence ----------------

    def _load(self):
        if os.path.exists(self._vectors_path) and os.path.exists(self._meta_path):
            data = np.load(self._vectors_path)
            self._vectors = data["vectors"]
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._ids = meta["ids"]
            self._contents = meta["contents"]
            self._metadatas = meta["metadatas"]
        else:
            self._vectors = np.zeros((0, 0), dtype=np.float32)

    def _save(self):
        np.savez(self._vectors_path, vectors=self._vectors)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self._ids, "contents": self._contents, "metadatas": self._metadatas}, f)

    # ---------------- CRUD ----------------

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        new_vecs = np.array(self.embeddings.embed_documents(texts), dtype=np.float32)
        new_vecs = self._normalize(new_vecs)
        if self._vectors.size == 0:
            self._vectors = new_vecs
        else:
            self._vectors = np.vstack([self._vectors, new_vecs])
        self._ids.extend(ids)
        self._contents.extend(texts)
        self._metadatas.extend(metadatas)
        self._save()

    def delete_where(self, key: str, value):
        keep = [i for i, m in enumerate(self._metadatas) if m.get(key) != value]
        if len(keep) == len(self._metadatas):
            return  # nothing matched
        self._ids = [self._ids[i] for i in keep]
        self._contents = [self._contents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self._vectors.shape[1] if self._vectors.ndim > 1 else 0), dtype=np.float32)
        self._save()

    def similarity_search(self, query: str, k: int = 3):
        if self._vectors.size == 0 or len(self._ids) == 0:
            return []
        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        q = self._normalize(q.reshape(1, -1))[0]
        scores = self._vectors @ q  # cosine similarity, since both are L2-normalized
        top_idx = np.argsort(-scores)[:k]
        return [
            {"content": self._contents[i], "metadata": self._metadatas[i], "score": float(scores[i])}
            for i in top_idx
        ]

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms
