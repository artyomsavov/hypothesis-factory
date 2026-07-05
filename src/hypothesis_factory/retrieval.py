from typing import List

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from hypothesis_factory.base import BaseRetriever, BusinessRequest, DocumentChunk, DocumentMetadata


class ChromaRetriever(BaseRetriever):
    """Реализация RAG-поиска на базе ChromaDB и SentenceTransformers."""

    def __init__(self, db_path: str = None, collection_name: str = "materials_db_notebook"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

        # Поддерживаем два режима: постоянный (на диске) и временный (в ОЗУ)
        if db_path:
            self.client = chromadb.PersistentClient(path=db_path)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Векторизует чанки и сохраняет их в активную коллекцию ChromaDB."""
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        # Распаковываем объекты Pydantic
        for chunk in chunks:
            documents.append(chunk.text)
            ids.append(chunk.chunk_id)
            metadatas.append(
                {
                    "source_id": chunk.metadata.source_id,
                    "title": chunk.metadata.title or chunk.metadata.source_id,
                    "source_type": chunk.metadata.source_type,
                }
            )

        # Векторизация батчами с учетом доступности видеокарты
        batch_size = 64 if torch.cuda.is_available() else 16
        embeddings = self.embed_model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()

        # Запись в базу
        self.collection.add(
            documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    def get_context(self, request: BusinessRequest, top_k: int = 7) -> List[DocumentChunk]:
        # 1. Формируем запрос
        query_text = f"{request.target_kpi} Ограничения: {', '.join(request.constraints)}"
        query_embedding = self.embed_model.encode([query_text]).tolist()

        # 2. Ищем в базе
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)

        # 3. Упаковываем ответ в Pydantic контракты
        context_chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][i]
                text = results["documents"][0][i]
                meta = results["metadatas"][0][i]

                metadata = DocumentMetadata(
                    source_id=meta.get("source_id", f"unknown_doc_{i}"),
                    source_type=meta.get("source_type", "article"),
                    title=meta.get("title", "Неизвестный источник"),
                )
                context_chunks.append(
                    DocumentChunk(chunk_id=chunk_id, text=text, metadata=metadata)
                )

        return context_chunks
