import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pymupdf4llm

from hypothesis_factory.base import BaseReader, DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class PyMuPDFReader(BaseReader):
    """
    Парсер PDF документов с конвертацией в Markdown.
    Использует постраничную нарезку (1 страница = 1 чанк) как базовый вариант RAG.
    """

    def read(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        chunks: List[DocumentChunk] = []

        if not path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return chunks

        # MVP: Обрабатываем только PDF.
        # Если подсунут txt/md, читаем как обычный текст.
        if path.suffix.lower() not in [".pdf"]:
            return self._read_fallback_text(path)

        try:
            # 1. Извлекаем сырые метаданные из файла
            with fitz.open(path) as doc:
                raw_meta = doc.metadata or {}

            metadata = DocumentMetadata(
                source_id=path.name,
                source_type="article",  # Дефолт для хакатона, можно парсить из папки
                title=raw_meta.get("title", path.stem) or path.stem,
                authors=[raw_meta.get("author")] if raw_meta.get("author") else [],
            )

            # 2. Конвертируем PDF в Markdown с разбивкой по страницам
            # Это автоматически изолирует таблицы и заголовки в пределах страницы
            md_pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)

            # 3. Упаковываем в наши Pydantic контракты
            for i, page_data in enumerate(md_pages):
                text = page_data.get("text", "").strip()
                if not text:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"{path.name}_page_{i + 1}", text=text, metadata=metadata
                )
                chunks.append(chunk)

        except Exception as e:
            # Жесткий перехват: один битый PDF не должен уложить весь пайплайн
            logger.error(f"Ошибка при парсинге {path.name}: {str(e)}")

        return chunks

    def _read_fallback_text(self, path: Path) -> List[DocumentChunk]:
        """Фоллбек для простых текстовых файлов (.txt, .md, .csv)"""
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
