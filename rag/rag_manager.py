"""
RAGManager owns the vector store and is the single place that talks to it.
Every write goes through here so the SQL table (KnowledgeDocument, the
human-readable source of truth shown in the dashboard) and the vector index
(what the agent actually searches) never drift apart:

    dashboard form  -->  KnowledgeDocument row (SQLite)  -->  RAGManager
                                                                  |
                                                                  v
                                                        SimpleVectorStore (numpy)

Why two stores instead of one? SQLite is what the admin edits and what the
dashboard lists/paginates easily. The vector store is optimized for "find
the k most semantically similar chunks to this query", which SQL can't do.
Keeping the SQL id as the vector store's source_id metadata is what lets
update/delete stay in sync (we just drop and re-add vectors with that id).

The vector store itself (rag/vector_store.py) is a small numpy-based
implementation rather than Chroma/FAISS — deliberately, to avoid a compiled
C++ extension dependency that's painful to install on some machines (e.g.
Windows without a C++ build toolchain). It exposes the same add/delete/search
shape a "real" vector DB client would, so swapping in Chroma or a hosted
vector DB later is a change confined to this file + vector_store.py.
"""
import os
from rag.vector_store import SimpleVectorStore
from config import Config


def _build_embeddings():
    """EMBEDDING_PROVIDER=local (default) uses a free sentence-transformers
    model that runs on your own machine — no API key, no account, no cost.
    The model weights (~80MB) download once from Hugging Face the first time
    this runs, then it works fully offline. Set EMBEDDING_PROVIDER=openai if
    you'd rather use OpenAI's embedding endpoint instead."""
    if Config.EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=Config.OPENAI_EMBEDDING_MODEL, api_key=Config.OPENAI_API_KEY)

    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=Config.LOCAL_EMBEDDING_MODEL)


class RAGManager:
    def __init__(self):
        os.makedirs(Config.VECTOR_STORE_DIR, exist_ok=True)
        self.embeddings = _build_embeddings()
        self.store = SimpleVectorStore(persist_dir=Config.VECTOR_STORE_DIR, embeddings=self.embeddings)

    # ---- CRUD, called by the dashboard routes and by seed_data.py ----

    def add_document(self, doc_id: int, title: str, category: str, content: str):
        """Embed and index one KnowledgeDocument row. Long content is chunked
        so retrieval returns focused passages instead of whole documents."""
        chunks = self._chunk(content)
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [{"source_id": doc_id, "title": title, "category": category, "chunk": i}
                     for i in range(len(chunks))]
        self.store.add(ids=ids, texts=chunks, metadatas=metadatas)

    def update_document(self, doc_id: int, title: str, category: str, content: str):
        self.delete_document(doc_id)
        self.add_document(doc_id, title, category, content)

    def delete_document(self, doc_id: int):
        self.store.delete_where("source_id", doc_id)

    def search(self, query: str, k: int = 3):
        """Return the top-k most relevant chunks for a query, used by the
        agent's `rag_search` tool."""
        results = self.store.similarity_search(query, k=k)
        return [
            {
                "title": r["metadata"].get("title"),
                "category": r["metadata"].get("category"),
                "content": r["content"],
            }
            for r in results
        ]

    @staticmethod
    def _chunk(text: str, max_len: int = 600):
        """Simple paragraph-aware chunking: keep paragraphs together until the
        max length would be exceeded. Good enough for FAQ/policy-sized docs;
        a bigger knowledge base would want a smarter splitter."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) + 1 > max_len and current:
                chunks.append(current.strip())
                current = p
            else:
                current = f"{current}\n{p}" if current else p
        if current:
            chunks.append(current.strip())
        return chunks or [text]


_rag_manager = None


def get_rag_manager() -> RAGManager:
    """Lazy singleton so we only load the embedding model once per process."""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGManager()
    return _rag_manager
