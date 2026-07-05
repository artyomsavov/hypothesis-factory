# backend/main.py
import os
import sys
import tempfile
import time
from typing import List

import chromadb
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Добавляем корневую папку в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Импорты Docling
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.hypothesis_factory.base import BusinessRequest, DocumentChunk, DocumentMetadata
from src.hypothesis_factory.critic import YandexGPTCritic
from src.hypothesis_factory.generator import YandexGPTGenerator

# Твои импорты
from src.hypothesis_factory.ingestion_gpu import GpuDoclingReader
from src.hypothesis_factory.pipeline import HypothesisRefinementPipeline

app = FastAPI(title="Hypothesis Factory API")

# Глобальные переменные для сохранения состояния в памяти
embed_model = None
active_collection = None
docling_reader = None
DEVICE = "cpu"


@app.on_event("startup")
def startup_event():
    """Инициализация моделей и БД при запуске сервера (выполняется 1 раз)."""
    global embed_model, active_collection, docling_reader, DEVICE

    load_dotenv()
    cuda_ok = torch.cuda.is_available()
    DEVICE = "cuda" if cuda_ok else "cpu"

    print(f"Запуск бэкенда. Устройство: {DEVICE}")
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)

    # Инициализация ChromaDB (используем постоянную базу как основную)
    chroma_db_dir = os.path.join(project_root, "data", "chroma_db")
    os.makedirs(chroma_db_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
    active_collection = chroma_client.get_or_create_collection("materials_db_notebook")

    # Настройка Docling для загрузки новых файлов
    accelerator_options = AcceleratorOptions(
        num_threads=6,
        device=AcceleratorDevice.CUDA if cuda_ok else AcceleratorDevice.CPU,
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    docling_reader = GpuDoclingReader()
    docling_reader.converter = converter
    docling_reader.chunker = HierarchicalChunker()


class GenerateRequest(BaseModel):
    target_kpi: str
    constraints: List[str]


@app.post("/generate_hypotheses")
def generate_hypotheses_endpoint(req: GenerateRequest):
    """Поиск контекста и запуск LLM пайплайна."""
    request = BusinessRequest(target_kpi=req.target_kpi, constraints=req.constraints)

    # Векторный поиск
    query_text = f"{request.target_kpi} Ограничения: {', '.join(request.constraints)}"
    query_embedding = embed_model.encode([query_text]).tolist()

    results = active_collection.query(query_embeddings=query_embedding, n_results=7)

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
            context_chunks.append(DocumentChunk(chunk_id=chunk_id, text=text, metadata=metadata))

    # Запуск LLM
    with YandexGPTGenerator() as generator, YandexGPTCritic() as critic:
        pipeline = HypothesisRefinementPipeline(
            generator=generator, critic=critic, min_overall_score=7.0, max_iterations=3
        )
        final_hypotheses = pipeline.run_loop(request=request, context=context_chunks)

    # Возвращаем результаты (Pydantic сериализуется в JSON)
    return {
        "preliminary_analysis": f"Найдено {len(context_chunks)} релевантных источников в базе.",
        "hypotheses": [
            hyp.model_dump() if hasattr(hyp, "model_dump") else hyp.dict()
            for hyp in final_hypotheses
        ],
    }


@app.post("/upload_documents")
async def upload_documents_endpoint(files: List[UploadFile] = File(...)):
    """Парсинг новых PDF через Docling и добавление в ChromaDB."""
    added_chunks_count = 0

    for file in files:
        # Сохраняем файл во временную директорию
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = docling_reader.read(tmp_path)

            documents, metadatas, ids = [], [], []
            for chunk in chunks:
                documents.append(chunk.text)
                ids.append(chunk.chunk_id)
                metadatas.append(
                    {
                        "source_id": chunk.metadata.source_id,
                        "title": file.filename,  # Сохраняем оригинальное имя файла
                        "source_type": chunk.metadata.source_type,
                    }
                )

            if documents:
                embeddings = embed_model.encode(documents, batch_size=16).tolist()
                active_collection.add(
                    documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids
                )
                added_chunks_count += len(documents)
        finally:
            os.remove(tmp_path)

    return {"added_chunks": added_chunks_count}
