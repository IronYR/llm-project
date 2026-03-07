import logging
import uuid
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64
_UPSERT_BATCH = 128


class EmbeddingIndex:

    def __init__(self, config: dict) -> None:
        self.config = config
        emb_cfg  = config["embeddings"]
        qdrant_cfg = config["qdrant"]
        retr_cfg = config["retrieval"]

        self.model_name     = emb_cfg["model"]
        self.collection     = qdrant_cfg["collection_name"]
        self.vector_size    = qdrant_cfg["vector_size"]
        self.top_k          = retr_cfg["top_k"]
        self.score_threshold = retr_cfg["score_threshold"]

        # Lazy-loaded to keep startup fast
        self._encoder: SentenceTransformer | None = None
        self._client:  QdrantClient | None = None

        self._qdrant_mode = qdrant_cfg["mode"]
        self._qdrant_path = qdrant_cfg.get("path", "./data/qdrant_store")


    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            if self._qdrant_mode == "memory":
                logger.info("Using Qdrant in-memory store")
                self._client = QdrantClient(":memory:")
            elif self._qdrant_mode == "server":
                url = self.config["qdrant"].get("url", "http://localhost:6333")
                logger.info("Connecting to Qdrant server: %s", url)
                self._client = QdrantClient(url=url)
            else:  # file (default)
                path = Path(self._qdrant_path)
                path.mkdir(parents=True, exist_ok=True)
                logger.info("Using Qdrant file store: %s", path)
                self._client = QdrantClient(path=str(path))
        return self._client

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not exist yet."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.collection)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of strings in batches, returning float vectors."""
        vectors = self.encoder.encode(
            texts,
            batch_size=_EMBED_BATCH,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def _doc_to_payload(self, doc: dict) -> dict:
        """Extract the Qdrant payload (metadata) from a document dict."""
        return {
            "doc_id":       doc["id"],
            "category":     doc["category"],
            "source":       doc["source"],
            "source_sheet": doc["source_sheet"],
            "question":     doc.get("question", ""),
            "answer":       doc.get("answer", ""),
            "text":         doc.get("text", ""),
        }


    def collection_exists(self) -> bool:
        """Return True if the Qdrant collection has been built."""
        try:
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection not in existing:
                return False
            info = self.client.get_collection(self.collection)
            return info.points_count > 0
        except Exception:
            return False

    def build(self, documents: list[dict]) -> None:
        # Drop & recreate
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            self.client.delete_collection(self.collection)
            logger.info("Dropped existing collection: %s", self.collection)

        self._ensure_collection()

        texts = [doc["text"] for doc in documents]
        logger.info("Embedding %d documents...", len(texts))
        vectors = self._embed_texts(texts)

        # Upsert in batches
        points: list[PointStruct] = []
        for doc, vec in zip(documents, vectors):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=self._doc_to_payload(doc),
            ))

        for i in range(0, len(points), _UPSERT_BATCH):
            batch = points[i : i + _UPSERT_BATCH]
            self.client.upsert(collection_name=self.collection, points=batch)

        logger.info("Index built: %d vectors stored in '%s'", len(points), self.collection)

    def add(self, documents: list[dict]) -> None:
        self._ensure_collection()

        texts   = [doc["text"] for doc in documents]
        vectors = self._embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=self._doc_to_payload(doc),
            )
            for doc, vec in zip(documents, vectors)
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Added %d document(s) to collection '%s'", len(points), self.collection)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.top_k

        query_vec = self._embed_texts([query])[0]

        qdrant_filter = None
        if category_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(
                    key="category",
                    match=MatchValue(value=category_filter),
                )]
            )

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=k,
            score_threshold=self.score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        results = []
        for hit in response.points:
            p = hit.payload or {}
            results.append({
                "score":        round(hit.score, 4),
                "category":     p.get("category", ""),
                "question":     p.get("question", ""),
                "answer":       p.get("answer", ""),
                "source":       p.get("source", ""),
                "source_sheet": p.get("source_sheet", ""),
                "text":         p.get("text", ""),
            })

        logger.debug("Search '%s': %d results (threshold=%.2f)", query[:60], len(results), self.score_threshold)
        return results

    def get_stats(self) -> dict:
        """Return basic collection statistics."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "collection":    self.collection,
                "vectors_count": info.points_count,
                "vector_size":   self.vector_size,
                "embedding_model": self.model_name,
            }
        except Exception as e:
            return {"error": str(e)}
