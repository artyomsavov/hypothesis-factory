import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pymupdf4llm

from hypothesis_factory.base import BaseReader, DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class PyMuPDFReader(BaseReader):
    """
    Быстрый CPU-парсер PDF документов с конвертацией в Markdown.
    Использует постраничную нарезку (1 страница = 1 чанк).

    table_strategy=None по умолчанию отключает поиск таблиц —
    это основной источник тормозов в pymupdf4llm на страницах
    с плотной версткой/графикой (научные статьи, диаграммы и т.д.).
    """

    def __init__(self, extract_tables: bool = False, graphics_limit: int = 2000):
        self.table_strategy = "lines" if extract_tables else None
        self.graphics_limit = graphics_limit

    def read(self, file_path: str) -> List[DocumentChunk]:
        path = Path(file_path)
        chunks: List[DocumentChunk] = []

        if not path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return chunks

        if path.suffix.lower() not in [".pdf"]:
            return self._read_fallback_text(path)

        try:
            with fitz.open(path) as doc:
                raw_meta = doc.metadata or {}

            metadata = DocumentMetadata(
                source_id=path.name,
                source_type="article",
                title=raw_meta.get("title", path.stem) or path.stem,
                authors=[raw_meta.get("author")] if raw_meta.get("author") else [],
            )

            md_pages = pymupdf4llm.to_markdown(
                str(path),
                page_chunks=True,
                table_strategy=self.table_strategy,  # None = без поиска таблиц (быстро)
                graphics_limit=self.graphics_limit,  # защита от "тяжелых" по графике страниц
                write_images=False,
            )

            for i, page_data in enumerate(md_pages):
                text = page_data.get("text", "").strip()
                if not text:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"{path.name}_page_{i + 1}", text=text, metadata=metadata
                )
                chunks.append(chunk)

        except Exception as e:
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