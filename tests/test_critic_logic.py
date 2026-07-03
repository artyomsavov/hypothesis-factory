from unittest.mock import patch

import pytest
from pydantic import ValidationError

from hypothesis_factory.base import DocumentChunk, DocumentMetadata, Hypothesis
from hypothesis_factory.critic import EvaluationResult, YandexGPTCritic


# 1. Тесты Pydantic-контрактов
def test_evaluation_result_success():
    res = EvaluationResult(
        novelty_score=7.5,
        feasibility_score=8.0,
        technical_risks=["Риск коррозии при высоких температурах"],
        economic_risks=["Высокая стоимость редкоземельных металлов"],
        is_valid=True,
    )
    assert res.novelty_score == 7.5
    assert res.is_valid is True


def test_evaluation_result_score_overflow():
    with pytest.raises(ValidationError):
        EvaluationResult(
            novelty_score=12.5,  # Выход за верхнюю границу
            feasibility_score=5.0,
            technical_risks=["Риск"],
            economic_risks=["Риск"],
            is_valid=True,
        )

    with pytest.raises(ValidationError):
        EvaluationResult(
            novelty_score=5.0,
            feasibility_score=-1.0,  # Выход за нижнюю границу
            technical_risks=["Риск"],
            economic_risks=["Риск"],
            is_valid=True,
        )


def test_evaluation_result_empty_risks_trigger_validator():
    with pytest.raises(ValidationError) as exc_info:
        EvaluationResult(
            novelty_score=6.0,
            feasibility_score=6.0,
            technical_risks=[],  # Пустой список
            economic_risks=["Экономический риск"],
            is_valid=True,
        )
    assert "Должен быть указан хотя бы один риск" in str(exc_info.value)


# 2. Фикстуры
@pytest.fixture
def mock_critic():
    return YandexGPTCritic(api_key="mock_key", folder_id="mock_folder")


@pytest.fixture
def sample_context():
    meta = DocumentMetadata(source_id="paper_01", source_type="article")
    return [
        DocumentChunk(
            chunk_id="chunk_1", text="Добавление титана увеличивает прочность.", metadata=meta
        ),
        DocumentChunk(
            chunk_id="chunk_2", text="Расходы на логистику составляют 15%.", metadata=meta
        ),
    ]


@pytest.fixture
def sample_hypotheses():
    return [
        Hypothesis(
            id="hyp_low",
            title="Слабая гипотеза",
            text="...",
            mechanism="...",
            reasoning="...",
            source_refs=["chunk_1"],
        ),
        Hypothesis(
            id="hyp_high",
            title="Сильная гипотеза",
            text="...",
            mechanism="...",
            reasoning="...",
            source_refs=["chunk_1"],
        ),
        Hypothesis(
            id="hyp_invalid",
            title="Антинаучный бред",
            text="...",
            mechanism="...",
            reasoning="...",
            source_refs=["chunk_1"],
        ),
    ]


# 3. Тесты бизнес-логики через patch.object
def test_critic_context_filtering_logic(mock_critic, sample_context, sample_hypotheses):
    hyp = sample_hypotheses[0]
    hyp.source_refs = ["chunk_2"]

    # Изолируем вызов сети через контекстный менеджер
    with patch.object(mock_critic.client.chat.completions, "create") as mock_create:
        mock_create.return_value = EvaluationResult(
            novelty_score=5.0,
            feasibility_score=5.0,
            technical_risks=["Тест"],
            economic_risks=["Тест"],
            is_valid=True,
        )

        mock_critic._score_single_hypothesis(hyp, sample_context)

        assert mock_create.called
        user_content = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "Расходы на логистику" in user_content
        assert "Добавление титана" not in user_content


def test_critic_context_empty_refs_fallback(mock_critic, sample_context, sample_hypotheses):
    hyp = sample_hypotheses[0]
    hyp.source_refs = ["non_existent_chunk_id"]

    with patch.object(mock_critic.client.chat.completions, "create") as mock_create:
        mock_create.return_value = EvaluationResult(
            novelty_score=1.0,
            feasibility_score=1.0,
            technical_risks=["Тест"],
            economic_risks=["Тест"],
            is_valid=True,
        )

        mock_critic._score_single_hypothesis(hyp, sample_context)

        user_content = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "Контекст не предоставлен." in user_content


def test_evaluate_handles_sorting_and_filtering(mock_critic, sample_context, sample_hypotheses):
    res_low = EvaluationResult(
        novelty_score=3.0,
        feasibility_score=5.0,
        technical_risks=["Т1"],
        economic_risks=["Э1"],
        is_valid=True,
    )
    res_high = EvaluationResult(
        novelty_score=9.0,
        feasibility_score=9.0,
        technical_risks=["Т2"],
        economic_risks=["Э2"],
        is_valid=True,
    )
    res_invalid = EvaluationResult(
        novelty_score=1.0,
        feasibility_score=1.0,
        technical_risks=["Т3"],
        economic_risks=["Э3"],
        is_valid=False,
    )

    with patch.object(mock_critic.client.chat.completions, "create") as mock_create:
        # Отдаем три результата для трех итераций цикла
        mock_create.side_effect = [res_low, res_high, res_invalid]

        final_list = mock_critic.evaluate(sample_hypotheses, sample_context)

        assert len(final_list) == 2
        assert final_list[0].id == "hyp_high"
        assert final_list[0].overall_score == 9.0
        assert final_list[1].id == "hyp_low"
        assert final_list[1].overall_score == 4.0


def test_evaluate_resilience_on_api_crash(mock_critic, sample_context, sample_hypotheses):
    with patch.object(mock_critic.client.chat.completions, "create") as mock_create:
        # Провоцируем жесткое исключение
        mock_create.side_effect = Exception("Yandex Cloud API Error: Token Expired")

        final_list = mock_critic.evaluate([sample_hypotheses[0]], sample_context)

        assert len(final_list) == 1
        hyp = final_list[0]
        assert hyp.overall_score == 0.0
        assert "ОШИБКА: Сбой API Критика" in hyp.technical_risks[0]
