import logging
import os
from typing import List

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from hypothesis_factory.base import BaseCritic, DocumentChunk, Hypothesis

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Строгий контракт для ответа LLM-критика."""

    novelty_score: float = Field(..., ge=0.0, le=10.0, description="Оценка новизны от 0 до 10")
    feasibility_score: float = Field(
        ..., ge=0.0, le=10.0, description="Оценка реализуемости от 0 до 10"
    )
    technical_risks: List[str] = Field(..., description="Список технических проблем (минимум одна)")
    economic_risks: List[str] = Field(
        ..., description="Список экономических проблем (минимум одна)"
    )
    is_valid: bool = Field(
        ...,
        description="False, если гипотеза нарушает законы физики или является откровенным бредом",
    )

    @field_validator("technical_risks", "economic_risks")
    @classmethod
    def check_risks_not_empty(cls, v):
        if not v:
            raise ValueError("Должен быть указан хотя бы один риск. Идеальных гипотез не бывает.")
        return v


class YandexGPTCritic(BaseCritic):
    """
    LLM-судья на базе YandexGPT через Instructor.
    Оценивает гипотезы на реализуемость, ищет риски и фильтрует галлюцинации с автоматическим retry.
    """

    def __init__(self, api_key: str = None, folder_id: str = None):
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")

        if not self.api_key or not self.folder_id:
            logger.warning("Yandex API ключи не заданы! Критик работать не сможет.")

        # Настраиваем базовый OpenAI-клиент на эндпоинт Яндекса
        base_client = OpenAI(
            api_key=self.api_key,
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={"Authorization": f"Api-Key {self.api_key}"},
        )

        # Оборачиваем клиент в Instructor
        # YandexGPT лучше работает с JSON режимом, чем с нативным Tool Calling от OpenAI
        self.client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"

    def evaluate(
        self, hypotheses: List[Hypothesis], context: List[DocumentChunk]
    ) -> List[Hypothesis]:

        evaluated_hypotheses = []

        for hyp in hypotheses:
            logger.info(f"Оценка гипотезы: {hyp.title}")
            eval_result = self._score_single_hypothesis(hyp, context)

            if not eval_result.is_valid:
                logger.warning(f"Отбраковано: '{hyp.title}' (нарушение логики/физики).")
                continue

            hyp.novelty_score = eval_result.novelty_score
            hyp.feasibility_score = eval_result.feasibility_score
            hyp.technical_risks = eval_result.technical_risks
            hyp.economic_risks = eval_result.economic_risks
            hyp.overall_score = round((hyp.novelty_score + hyp.feasibility_score) / 2, 2)

            evaluated_hypotheses.append(hyp)

        evaluated_hypotheses.sort(key=lambda x: x.overall_score, reverse=True)
        return evaluated_hypotheses

    def _score_single_hypothesis(
        self, hyp: Hypothesis, context: List[DocumentChunk]
    ) -> EvaluationResult:
        used_context = [c.text for c in context if c.chunk_id in hyp.source_refs]
        context_str = "\n---\n".join(used_context) if used_context else "Контекст не предоставлен."

        system_prompt = """
        Ты — объективный, строгий главный инженер и научный рецензент.
        Твоя задача — оценить технические гипотезы на основе заданного контекста.

        Общие правила:
        1. Оценивай только на основе переданного контекста и законов физики.
        2. Всегда выявляй:
            - технические ограничения (физика, инженерные и технологические лимиты)
            - экономические риски (затраты, бизнес и рыночные риски)
        3. Всегда давай отдельные оценки:
            - новзина гипотезы (novelty_score) от 0 до 10
            - техническая реализуемость (feasibility_score) от 0 до 10
        4. Если гипотеза нарушает законы физики, явно противоречит контексту, считай её невалидной:
            - "is_valid": false
            - "feasibility_score": 0
            во всех остальных случаях "is_valid": true.
        5. Будь критичен, но объективен. Избегай эмоциональных оценок и субъективных суждений.

        Формат вывода:
        - Отвечай строго в виде структурированного объекта согласно определённой схеме.
        - Соблюдай точные имена полей и типы значений.
        """

        user_prompt = f"""
        КОНТЕКСТ (Источники):
        {context_str}

        ГИПОТЕЗА:
        Название: {hyp.title}
        Суть: {hyp.text}
        Механизм: {hyp.mechanism}

        ЗАДАЧА:
        На основе этого контекста оцени данную гипотезу в соответствии со своими общими правилами.
        1. Выяви ключевые технические ограничения и потенциальные точки отказа.
        2. Выяви основыне экономические риски и факторы, которые могут сделать гипотезу нерентабельной.
        3. Оцени новизну и техническую реализуемость по своим шаклам.
        4. Определи, является ли гипотеза валидной (не нарушает ли законы физики и не противоречит ли контексту).

        Используй только данный контекст, если информации недостаточно для уверенной оценки, явно укажи это в заметках.
        """

        try:
            # Instructor автоматически заставит модель переписать ответ,
            # если Pydantic-валидаторы выкинут ошибку (max_retries=3)
            result = self.client.chat.completions.create(
                model=self.model_uri,
                response_model=EvaluationResult,
                max_retries=3,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return result

        except Exception as e:
            logger.error(f"Сбой Критика для '{hyp.title[:20]}...': {str(e)}")
            return EvaluationResult(
                novelty_score=0.0,
                feasibility_score=0.0,
                technical_risks=["ОШИБКА: Сбой API Критика"],
                economic_risks=["ОШИБКА: Сбой API Критика"],
                is_valid=True,
            )
