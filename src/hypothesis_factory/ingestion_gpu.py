import logging
from pathlib import Path
from typing import List

from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter

from hypothesis_factory.base import BaseReader, DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class GpuDoclingReader(BaseReader):
    """
    GPU-ускоренный парсер на базе IBM Docling.
    Использует нейросети (Vision-Language Models) для распознавания структуры документа.
    """

    def __init__(self):
        # Инициализируем модели один раз при старте.
        # Docling под капотом использует PyTorch и автоматически загрузит веса в вашу NVIDIA 20GB.
        logger.info("Загрузка Vision-моделей в VRAM...")
        self.converter = DocumentConverter()
        self.chunker = HierarchicalChunker()

    def read(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        chunks: List[DocumentChunk] = []

        if not path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return chunks

        if path.suffix.lower() not in [".pdf"]:
            return self._read_fallback_text(path)

        try:
            logger.info(f"GPU Парсинг файла: {path.name}...")

            # 1. Нейросетевой процессинг (Здесь работает видеокарта)
            # Модель распознает текст, таблицы, заголовки и связи между ними
            conv_result = self.converter.convert(file_path)

            metadata = DocumentMetadata(
                source_id=path.name, source_type="article", title=path.stem, authors=[]
            )

            # 2. Умный семантический чанкинг
            # Вместо разрыва по границам страниц, он бьет текст по логическим блокам и абзацам
            docling_chunks = self.chunker.chunk(conv_result.document)

            # 3. Упаковываем в ваши строгие Pydantic контракты
            for i, chunk_data in enumerate(docling_chunks):
                # chunk_data.text содержит очищенный Markdown
                text = chunk_data.text.strip()
                if not text:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"{path.name}_semantic_chunk_{i + 1}", text=text, metadata=metadata
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Ошибка при GPU парсинге {path.name}: {str(e)}")

        return chunks

    def _read_fallback_text(self, path: Path) -> List[DocumentChunk]:
        """Фоллбек для простых текстовых файлов"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                return []

            return [
                DocumentChunk(
                    chunk_id=f"{path.name}_full",
                    text=text,
                    metadata=DocumentMetadata(
                        source_id=path.name, source_type="text", title=path.stem, authors=[]
                    ),
                )
            ]
        except Exception as e:
            logger.error(f"Фоллбек не смог прочитать {path.name}: {str(e)}")
            return []
