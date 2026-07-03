from unittest.mock import patch

import pytest

from hypothesis_factory.base import (
    BusinessRequest,
    DocumentChunk,
    DocumentMetadata,
    Hypothesis,
    HypothesisList,
)
from hypothesis_factory.generator import YandexGPTGenerator


# 1. Подготовка данных
@pytest.fixture
def mock_generator():
    """Генератор с фейковыми ключами, чтобы не лезть в .env"""
    return YandexGPTGenerator(api_key="fake_key", folder_id="fake_folder")


@pytest.fixture
def sample_request():
    """Стандартный бизнес-запрос с ограничениями"""
    return BusinessRequest(
        target_kpi="Снизить массу детали на 20%",
        constraints=["Использовать только отечественные сплавы", "Бюджет до 1 млн рублей"],
    )


@pytest.fixture
def sample_context():
    """Контекст из двух чанков"""
    meta = DocumentMetadata(source_id="doc_1", source_type="article")
    return [
        DocumentChunk(
            chunk_id="chunk_101", text="Алюминиевые сплавы снижают массу.", metadata=meta
        ),
        DocumentChunk(chunk_id="chunk_102", text="Титановые сплавы дорогие.", metadata=meta),
    ]


@pytest.fixture
def fake_llm_response():
    """Имитация ответа от Instructor. Содержит одну валидную ссылку и одну галлюцинацию."""
    return HypothesisList(
        hypotheses=[
            Hypothesis(
                id="h_1",
                title="Алюминий",
                text="...",
                mechanism="...",
                reasoning="...",
                source_refs=["chunk_101"],  # Валидная ссылка
            ),
            Hypothesis(
                id="h_2",
                title="Титан",
                text="...",
                mechanism="...",
                reasoning="...",
                source_refs=["chunk_102", "chunk_999"],  # chunk_999 — это придумка LLM
            ),
        ]
    )


# 2. Тесты сборки промпта
def test_generator_prompt_assembly_with_context(mock_generator, sample_request, sample_context):
    """Проверяем, что ID чанков и ограничения корректно вшиваются в промпт."""

    with patch.object(mock_generator.client.chat.completions, "create") as mock_create:
        mock_create.return_value = HypothesisList(hypotheses=[])

        mock_generator.generate(sample_request, sample_context)

        # Вытаскиваем промпт, который улетел бы в модель
        user_content = mock_create.call_args.kwargs["messages"][1]["content"]

        # Проверяем склейку чанков и пришивание ID
        assert "[ID: chunk_101]" in user_content
        assert "Алюминиевые сплавы снижают массу." in user_content
        assert "[ID: chunk_102]" in user_content

        # Проверяем вставку ограничений
        assert "Использовать только отечественные сплавы, Бюджет до 1 млн рублей" in user_content


def test_generator_prompt_empty_context_and_constraints(mock_generator):
    """Проверяем поведение-заглушку при пустых вводных данных."""

    empty_req = BusinessRequest(target_kpi="Просто сделайте хорошо", constraints=[])

    with patch.object(mock_generator.client.chat.completions, "create") as mock_create:
        mock_create.return_value = HypothesisList(hypotheses=[])

        mock_generator.generate(empty_req, context=[])

        user_content = mock_create.call_args.kwargs["messages"][1]["content"]

        assert "Контекст не предоставлен. Опирайся на свои знания." in user_content
        assert "Нет жестких ограничений." in user_content


# 3. Защита от галлюцинаций (проверка пост-обработки)
def test_generator_cleans_hallucinated_refs(
    mock_generator, sample_request, sample_context, fake_llm_response
):
    """Проверяем, что генератор безжалостно удаляет несуществующие ID источников из ответа."""

    with patch.object(mock_generator.client.chat.completions, "create") as mock_create:
        # Отдаем заранее подготовленный ответ с галлюцинацией "chunk_999"
        mock_create.return_value = fake_llm_response

        hypotheses = mock_generator.generate(sample_request, sample_context)

        assert len(hypotheses) == 2

        # Первая гипотеза: валидная ссылка осталась на месте
        assert "chunk_101" in hypotheses[0].source_refs
        assert len(hypotheses[0].source_refs) == 1

        # Вторая гипотеза: chunk_102 остался, а chunk_999 был удален
        assert "chunk_102" in hypotheses[1].source_refs
        assert "chunk_999" not in hypotheses[1].source_refs
        assert len(hypotheses[1].source_refs) == 1


# 4. Выживаемость (Resilience)
def test_generator_resilience_on_crash(mock_generator, sample_request, sample_context):
    """Проверяем, что при жестком падении API генератор возвращает пустой список, а не роняет программу."""

    with patch.object(mock_generator.client.chat.completions, "create") as mock_create:
        mock_create.side_effect = Exception("Instructor Failed to Parse JSON")

        results = mock_generator.generate(sample_request, sample_context)

        assert isinstance(results, list)
        assert len(results) == 0
