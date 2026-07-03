import os

import pytest
from dotenv import load_dotenv

from hypothesis_factory.base import DocumentChunk, DocumentMetadata, Hypothesis
from hypothesis_factory.critic import YandexGPTCritic

load_dotenv()

# используйте: RUN_API_TESTS=1 uv run pytest tests/test_critic_api.py -v -s
RUN_API = os.getenv("RUN_API_TESTS") == "1"
HAS_KEYS = bool(os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"))


@pytest.mark.skipif(
    not RUN_API or not HAS_KEYS,
    reason="Пропуск интеграционного теста. Установите RUN_API_TESTS=1 и добавьте ключи в .env",
)
def test_critic_real_api_call():
    """Боевой запрос к Yandex Cloud для проверки работы instructor и сетевых доступов."""

    critic = YandexGPTCritic()

    # 1. Подготовка минимальной полезной нагрузки
    meta = DocumentMetadata(source_id="test_doc", source_type="article")
    context = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="Температура плавления стандартной стали составляет около 1500 градусов Цельсия.",
            metadata=meta,
        )
    ]

    # 2. Намеренно слабая гипотеза, нарушающая температурный режим,
    # чтобы спровоцировать модель на генерацию рисков
    hypotheses = [
        Hypothesis(
            id="test_hyp_1",
            title="Использование стали при экстремальных температурах",
            text="Предлагается использовать сталь для создания тигля, работающего при температуре 2000 градусов.",
            mechanism="Теплоемкость материала позволит удержать расплав.",
            reasoning="Сталь дешевая и доступная, что снизит затраты.",
            source_refs=["chunk_1"],
        )
    ]

    # 3. Вызов реального API
    results = critic.evaluate(hypotheses, context)

    # 4. Проверка контрактов
    assert len(results) == 1, (
        "Критик должен был вернуть одну гипотезу (или удалить её, если is_valid=False)"
    )

    evaluated_hyp = results[0]

    # Проверяем, что Instructor правильно разложил JSON в типы данных Pydantic
    assert isinstance(evaluated_hyp.novelty_score, float)
    assert 0.0 <= evaluated_hyp.novelty_score <= 10.0

    assert isinstance(evaluated_hyp.feasibility_score, float)
    assert 0.0 <= evaluated_hyp.feasibility_score <= 10.0

    assert isinstance(evaluated_hyp.overall_score, float)

    # Убеждаемся, что кастомный валидатор списков отработал
    assert isinstance(evaluated_hyp.technical_risks, list)
    assert len(evaluated_hyp.technical_risks) > 0, "Модель должна была найти хотя бы один риск"
    assert isinstance(evaluated_hyp.technical_risks[0], str)

    assert isinstance(evaluated_hyp.economic_risks, list)
    assert len(evaluated_hyp.economic_risks) > 0
    assert isinstance(evaluated_hyp.economic_risks[0], str)

    # Для отладки выведем ответ модели в консоль, чтобы посмотреть, как она мыслит
    print("\n--- ОТВЕТ LLM ---")
    print(f"Overall Score: {evaluated_hyp.overall_score}")
    print(f"Технические риски: {evaluated_hyp.technical_risks}")
    print(f"Экономические риски: {evaluated_hyp.economic_risks}")
    print("-----------------")
