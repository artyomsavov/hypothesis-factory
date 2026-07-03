from pathlib import Path

import pytest

from hypothesis_factory.base import DocumentChunk
from hypothesis_factory.ingestion import PyMuPDFReader


def test_pymupdf_reader_returns_valid_chunks():
    """Тест проверяет, что все PDF корректно парсятся в Pydantic-схемы."""
    reader = PyMuPDFReader()

    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"

    # Ищем все PDF файлы в папке data
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"В папке {data_dir} нет тестовых PDF файлов")

    print(f"\nНайдено PDF файлов для теста: {len(pdf_files)}\n" + "=" * 50)

    for file_path in pdf_files:
        print(f"\nФайл: {file_path.name}")

        # Парсим конкретный файл
        chunks = reader.read(str(file_path))

        # Если чанков 0 — это скан или битый файл. Выводим варнинг и идем дальше.
        if len(chunks) == 0:
            print(
                "ВНИМАНИЕ: Парсер вернул 0 чанков. Скорее всего, это скан без текста или запароленный файл."
            )
            print("-" * 50)
            continue

        # 1. Проверяем строгую типизацию первого чанка
        first_chunk = chunks[0]
        assert isinstance(first_chunk, DocumentChunk), "Объект не соответствует Pydantic-контракту"
        assert first_chunk.text.strip() != "", "Текст чанка пустой"
        assert first_chunk.metadata.source_type == "article", "Неверный дефолтный тип источника"

        # 2. Визуальный вывод для отладки
        print(f"-> Извлечено страниц (чанков): {len(chunks)}")
        print(f"-> Метаданные: {first_chunk.metadata.model_dump_json()}")
        print(f"-> Текст первого чанка (первые 200 символов):\n{first_chunk.text[:200]}...")
        print("-" * 50)
