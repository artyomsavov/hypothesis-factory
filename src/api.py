import os
import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from hypothesis_factory.base import BusinessRequest, HypothesisList
from hypothesis_factory.retrieval import ChromaRetriever
from hypothesis_factory.generator import YandexGPTGenerator
from hypothesis_factory.critic import YandexGPTCritic
from hypothesis_factory.pipeline import HypothesisRefinementPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    logger.info("Инициализация API: Подключение к ChromaDB...")

    project_root = os.path.abspath("..")
    chroma_db_dir = os.path.join(project_root, "data", "chroma_db")

    # Инициализируем наш новый класс Retrieval
    retriever = ChromaRetriever(db_path=chroma_db_dir)
    yield


app = FastAPI(title="Hypothesis Factory API", lifespan=lifespan)


@app.post("/generate", response_model=HypothesisList)
def generate(request: BusinessRequest):
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever не инициализирован.")

    # 1. Достаем контекст через абстракцию
    context_chunks = retriever.get_context(request)

    # 2. Прогоняем через пайплайн
    try:
        with YandexGPTGenerator() as generator, YandexGPTCritic() as critic:
            pipeline = HypothesisRefinementPipeline(generator=generator, critic=critic)
            final_hypotheses = pipeline.run_loop(request=request, context=context_chunks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка LLM: {str(e)}")

    return HypothesisList(
        preliminary_analysis=f"Проанализировано {len(context_chunks)} документов.",
        hypotheses=final_hypotheses,
    )
