from unittest.mock import MagicMock
import pytest

from hypothesis_factory.base import (
    BusinessRequest,
    DocumentChunk,
    DocumentMetadata,
    Hypothesis,
    BaseGenerator,
    BaseCritic,
)
from hypothesis_factory.pipeline import HypothesisRefinementPipeline


@pytest.fixture
def dummy_request_and_context():
    """Подготовка базовых входных данных для всех тестов."""
    request = BusinessRequest(target_kpi="Улучшить сплав", constraints=["Без титана"])
    meta = DocumentMetadata(source_id="doc_1", source_type="article")
    context = [DocumentChunk(chunk_id="chunk_1", text="Текст статьи", metadata=meta)]
    return request, context


@pytest.fixture
def sample_hypotheses():
    """Подготовка эталонных гипотез с разными баллами для проверки логики фильтрации."""
    hyp_good = Hypothesis(
        id="h_good",
        title="Хорошая идея",
        text="Описание",
        mechanism="Механизм",
        reasoning="Обоснование",
        source_refs=["chunk_1"],
        overall_score=8.5,  # Выше порога (проходит)
    )
    hyp_bad = Hypothesis(
        id="h_bad",
        title="Плохая идея",
        text="Описание",
        mechanism="Механизм",
        reasoning="Обоснование",
        source_refs=["chunk_1"],
        overall_score=4.0,  # Ниже порога (на доработку)
    )
    return hyp_good, hyp_bad


def test_pipeline_happy_path(dummy_request_and_context, sample_hypotheses):
    """
    Сценарий 1: Идеальный прогон.
    Генератор выдает гипотезу, Критик сразу ставит ей высокий балл.
    Пайплайн должен завершиться на 1-й итерации.
    """
    request, context = dummy_request_and_context
    hyp_good, _ = sample_hypotheses

    # Настраиваем моки
    mock_generator = MagicMock(spec=BaseGenerator)
    mock_generator.generate.return_value = [hyp_good]

    mock_critic = MagicMock(spec=BaseCritic)
    mock_critic.evaluate.return_value = [hyp_good]

    # Запускаем оркестратор
    pipeline = HypothesisRefinementPipeline(
        generator=mock_generator, critic=mock_critic, min_overall_score=7.0
    )
    result = pipeline.run_loop(request, context)

    # Проверки
    assert len(result) == 1
    assert result[0].id == "h_good"
    assert mock_generator.generate.call_count == 1
    assert mock_critic.evaluate.call_count == 1  # Только одна итерация проверки


def test_pipeline_empty_generator(dummy_request_and_context):
    """
    Сценарий 2: Падение генератора.
    Метод generate возвращает пустой список.
    Пайплайн должен корректно отдать [], не вызывая Критика.
    """
    request, context = dummy_request_and_context

    mock_generator = MagicMock(spec=BaseGenerator)
    mock_generator.generate.return_value = []

    mock_critic = MagicMock(spec=BaseCritic)

    pipeline = HypothesisRefinementPipeline(mock_generator, mock_critic)
    result = pipeline.run_loop(request, context)

    assert result == []
    assert mock_critic.evaluate.call_count == 0  # Критик не должен запускаться


def test_pipeline_self_correction_loop(dummy_request_and_context, sample_hypotheses):
    """
    Сценарий 3: Активация цикла обратной связи.
    На 1-й итерации Критик ставит низкий балл (4.0).
    На 2-й итерации (после симуляции доработки) Критик ставит высокий балл (8.5).
    """
    request, context = dummy_request_and_context
    hyp_good, hyp_bad = sample_hypotheses

    mock_generator = MagicMock(spec=BaseGenerator)
    mock_generator.generate.return_value = [hyp_bad]

    mock_critic = MagicMock(spec=BaseCritic)
    # Используем side_effect, чтобы задать поведение Критика на разных итерациях цикла
    mock_critic.evaluate.side_effect = [
        [hyp_bad],  # Ответ на итерации 1 (отклонено)
        [hyp_good],  # Ответ на итерации 2 (принято после доработки)
    ]

    pipeline = HypothesisRefinementPipeline(
        generator=mock_generator, critic=mock_critic, min_overall_score=7.0, max_iterations=3
    )
    result = pipeline.run_loop(request, context)

    # Проверки
    assert len(result) == 1
    assert result[0].id == "h_good"  # В итоге должна вернуться хорошая версия
    assert mock_critic.evaluate.call_count == 2  # Произошло ровно две проверки


def test_pipeline_max_iterations_reached(dummy_request_and_context, sample_hypotheses):
    """
    Сценарий 4: Достижение лимита (Deadlock).
    Критик упорно бракует гипотезу. Пайплайн должен прерваться
    при достижении max_iterations и вернуть то, что есть.
    """
    request, context = dummy_request_and_context
    _, hyp_bad = sample_hypotheses

    mock_generator = MagicMock(spec=BaseGenerator)
    mock_generator.generate.return_value = [hyp_bad]

    mock_critic = MagicMock(spec=BaseCritic)
    # Критик всегда возвращает плохую оценку
    mock_critic.evaluate.return_value = [hyp_bad]

    MAX_ITER = 3
    pipeline = HypothesisRefinementPipeline(
        generator=mock_generator, critic=mock_critic, min_overall_score=7.0, max_iterations=MAX_ITER
    )
    result = pipeline.run_loop(request, context)

    # Проверки
    assert len(result) == 1
    assert result[0].id == "h_bad"  # Вернулась забракованная гипотеза
    assert mock_critic.evaluate.call_count == MAX_ITER  # Цикл прерван строго на 3-й попытке
