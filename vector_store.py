"""
vector_store.py — Embedding-backed semantic search over a text column,
with a pluggable embedding provider (same "configurable, not hardcoded"
pattern as build_llm's ollama/anthropic switch in production_agent.py).

Providers:
  - "tfidf"  — scikit-learn TF-IDF vectors, fully local, no model download,
               no network call. Retrieval is term-overlap based rather than
               true semantic embedding, but it's a real, deterministic,
               fully-offline fallback — and it's what lets this whole
               pipeline be tested end-to-end without a live Ollama server.
  - "ollama" — real dense embeddings via Ollama's /api/embeddings endpoint
               (e.g. model="nomic-embed-text"). This is the intended
               production backend for actual semantic (not just keyword-
               overlap) retrieval.

Backed by chromadb for storage/similarity search — a real vector DB, not a
hand-rolled numpy cosine loop, since that's the actual tool being
demonstrated here (the earlier "no black box" push in this project was
about not hiding AGENT REASONING behind a framework, not about avoiding
real infrastructure libraries).
"""
import os
from typing import Optional

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class TfidfEmbeddingFunction(EmbeddingFunction):
    """Fully local, network-free embedding via TF-IDF + SVD dimensionality
    reduction. Fit once on the corpus being indexed; queries are embedded
    with the same fitted vectorizer. Retrieval quality is term-overlap
    based (not true semantic similarity — "eco-friendly" won't match
    "sustainable" the way a real embedding model would), but it's
    deterministic, requires no external model download, and is what makes
    this whole pipeline testable without a live Ollama server."""

    def __init__(self, corpus: list[str], n_components: int = 100):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        # TF-IDF vectors are extremely high-dimensional and sparse (one
        # dimension per vocabulary term); reduce to a small dense vector so
        # chromadb's similarity search behaves like it would for a real
        # dense embedding model.
        n_components = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        n_components = max(n_components, 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._svd.fit(tfidf_matrix)

    def __call__(self, input: Documents) -> Embeddings:
        tfidf = self._vectorizer.transform(input)
        dense = self._svd.transform(tfidf)
        return dense.tolist()


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Real dense embeddings via a local Ollama server's /api/embeddings
    endpoint. Requires the model to be pulled first, e.g.:
        ollama pull nomic-embed-text
    This is the intended production backend — TF-IDF above exists so the
    pipeline can be developed and tested without depending on a live model
    server being reachable."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def __call__(self, input: Documents) -> Embeddings:
        import httpx
        embeddings = []
        with httpx.Client(timeout=60) as client:
            for text in input:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
        return embeddings


def build_embedding_function(provider: str, corpus: Optional[list[str]] = None, **kwargs) -> EmbeddingFunction:
    provider = provider.lower()
    if provider == "tfidf":
        if corpus is None:
            raise ValueError("TF-IDF embedder requires the corpus to fit on — pass corpus=...")
        return TfidfEmbeddingFunction(corpus, **kwargs)
    if provider == "ollama":
        return OllamaEmbeddingFunction(**kwargs)
    raise ValueError(f"Unknown embedding provider '{provider}'. Use 'tfidf' or 'ollama'.")


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a chromadb collection: index a list of
    (id, text, metadata) rows from a SQL table's text column, then search
    by semantic similarity and get back the original row IDs — which the
    caller uses to constrain a follow-up SQL query (the "hybrid" part of
    RAG + SQL hybrid)."""

    def __init__(self, collection_name: str, embedding_function: EmbeddingFunction,
                 persist_path: Optional[str] = None):
        self._client = (
            chromadb.PersistentClient(path=persist_path) if persist_path
            else chromadb.EphemeralClient()
        )
        # Fresh collection each time — this is a demo/dev tool indexing a
        # live DB's current contents, not a long-lived production index
        # with incremental updates.
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=collection_name, embedding_function=embedding_function
        )

    def index_rows(self, rows: list[dict], id_field: str, text_field: str, metadata_fields: Optional[list[str]] = None):
        """rows: list of dicts (e.g. from a SQL SELECT). Each row's
        `text_field` value is embedded; `id_field` becomes the row's
        identity for joining back to SQL; any `metadata_fields` are stored
        alongside for display without needing a follow-up SQL call."""
        ids = [str(r[id_field]) for r in rows]
        documents = [str(r[text_field]) for r in rows]
        metadatas = None
        if metadata_fields:
            metadatas = [{f: r.get(f) for f in metadata_fields} for r in rows]
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        result = self._collection.query(query_texts=[query], n_results=top_k)
        matches = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        metadatas = result.get("metadatas", [[]])[0] or [{}] * len(ids)
        for i, doc, dist, meta in zip(ids, docs, distances, metadatas):
            matches.append({"id": i, "text": doc, "distance": dist, **(meta or {})})
        return matches