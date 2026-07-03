# Наш запрос -> Сеть -> YandexGPT -> JSON-ответ -> Парсер Instructor -> Наш Pydantic-класс
import os

import pytest
from dotenv import load_dotenv

from hypothesis_factory.base import BusinessRequest, DocumentChunk, DocumentMetadata
from hypothesis_factory.generator import YandexGPTGenerator

load_dotenv()

# используйте: RUN_API_TESTS=1 uv run pytest tests/test_generator_api.py -v -s
RUN_API = os.getenv("RUN_API_TESTS") == "1"
HAS_KEYS = bool(os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"))


@pytest.mark.skipif(
    not RUN_API or not HAS_KEYS,
    reason="Пропуск API-теста. Установите RUN_API_TESTS=1 и добавьте ключи в .env",
)
def test_generator_real_api_call():
    """Боевой запрос к Yandex Cloud для генерации батча гипотез (temperature=0.7)."""

    generator = YandexGPTGenerator()

    # 1. Формируем бизнес-запрос
    request = BusinessRequest(
        target_kpi="Снизить массу несущей рамы квадрокоптера на 15%, сохранив прочность",
        constraints=["Запрещено использовать хрупкие материалы вроде керамики", "Бюджет ограничен"],
    )

    # 2. Формируем контекст
    meta = DocumentMetadata(source_id="materials_db", source_type="wiki")
    context = [
        DocumentChunk(
            chunk_id="c_titanium",
            text="Титановые сплавы обладают высокой удельной прочностью, но сложны в механической обработке.",
            metadata=meta,
        ),
        DocumentChunk(
            chunk_id="c_carbon",
            text="Углепластик (карбон) значительно легче металлов, отлично работает на растяжение, но боится точечных ударов.",
            metadata=meta,
        ),
    ]

    # 3. Боевой вызов к API
    # Температура 0.7 должна дать нам разные векторы идей
    hypotheses = generator.generate(request, context)

    # 4. Проверка контрактов
    assert 3 <= len(hypotheses) <= 5, (
        f"Генератор вернул нестандартное количество гипотез: {len(hypotheses)}"
    )

    # Проверяем структуру первой попавшейся гипотезы
    hyp = hypotheses[0]
    assert hyp.title != ""
    assert hyp.text != ""
    assert hyp.mechanism != ""

    # Проверяем, что ссылки (если они есть) отфильтровались корректно
    valid_ids = {"c_titanium", "c_carbon"}
    for ref in hyp.source_refs:
        assert ref in valid_ids, f"Модель сгаллюцинировала ссылку: {ref}"

    print(f"\n--- СГЕНЕРИРОВАННЫЕ ГИПОТЕЗЫ (Всего: {len(hypotheses)}) ---")
    for i, h in enumerate(hypotheses, 1):
        print(f"\nГипотеза #{i}: {h.title}")
        print(f"Суть: {h.text}")
        print(f"Источники: {h.source_refs}")
    print("-------------------------------------------------------")
