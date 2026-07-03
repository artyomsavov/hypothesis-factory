import logging
import os
from typing import List

import instructor
from openai import OpenAI

from hypothesis_factory.base import (
    BaseGenerator,
    BusinessRequest,
    DocumentChunk,
    Hypothesis,
    HypothesisList,
)

logger = logging.getLogger(__name__)


class YandexGPTGenerator(BaseGenerator):
    def __init__(self, api_key: str = None, folder_id: str = None):
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")

        if not self.api_key or not self.folder_id:
            logger.warning("Yandex API ключи не заданы! Генератор работать не сможет.")

        self.model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"

        self.base_client = OpenAI(
            api_key=self.api_key or "fake",
            base_url="https://llm.api.cloud.yandex.net/v1",
            default_headers={"Authorization": f"Api-Key {self.api_key}"},
        )
        self.client = instructor.from_openai(self.base_client, mode=instructor.Mode.JSON)

    def generate(self, request: BusinessRequest, context: List[DocumentChunk]) -> List[Hypothesis]:
        if not context:
            logger.warning("Контекст пуст. Генерация гипотез пойдет только по весам модели.")
            context_str = "Контекст не предоставлен. Опирайся на свои знания."
        else:
            context_blocks = []
            for chunk in context:
                context_blocks.append(f"[ID: {chunk.chunk_id}]\n{chunk.text}")
            context_str = "\n\n---\n\n".join(context_blocks)

        system_prompt = """
        Ты — ведущий R&D инженер и научный исследователь.
        Твоя задача — на основе бизнес-цели, ограничений и контекста генерировать набор перспективных
        физически реализуемых технических гипотез для решения бизнес-задач.

        Общие правила:
        1. Учитывай реальные физические и технологиеческие ограничения.
            - Не предлагай идей, нарушающие законы физики.
        2. Опираться на контекст (источники) при генерации гипотез:
            - Используй факты, тренды, ограничения и идеи, которые явно присутствуют в источниках.
        3. Стремись к разнообразию:
            - при генерации набора гипотез смещайся в разные векторы решения
            (материалы, процессы, геометрии, архитектурные подходы, организационные изменения и т.д.).
        4. Пиши кратко и технично, избегай общих фраз и маркетинговых формулировок.

        Формат вывода:
        - Отвечай строго в виде структурированного JSON-объекта согласно переданной схеме.
        - Сначала обязательно заполни поле "preliminary_analysis" логическими рассуждениями.
        - Только после проведения анализа формируй массив "hypotheses".
        """

        constraints_str = (
            ", ".join(request.constraints) if request.constraints else "Нет жестких ограничений."
        )

        user_prompt = f"""
        БИЗНЕС-ЦЕЛЬ:
        {request.target_kpi}

        ОГРАНИЧЕНИЯ:
        {constraints_str}

        КОНТЕКСТ (Источники):
        {context_str}

        ИНСТРУКЦИЯ:
        1. Изучи бизнес-цель, ограничения и контекст.
        2. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ (`preliminary_analysis`): Построчно сопоставь каждое ограничение с фактами из источников. Явно укажи, какие материалы или подходы использовать КАТЕГОРИЧЕСКИ НЕЛЬЗЯ из-за нарушения физики или ограничений ТЗ.
        3. Опираясь на выводы из анализа, предложи от 3 до 5 независимых технических гипотез.
        4. Обязательно используй разные векторы решений: материалы, процессы, геометрии.
        5. В поле `source_refs` гипотезы обязательно укажи список ID тех источников из контекста, на которые ты опираешься.
        6. ВАЖНО: Если контекст не содержит нужной информации и ты генерируешь гипотезу из собственных весов, оставь `source_refs` пустым, а поле `reasoning` обязательно начни с фразы: "ВНИМАНИЕ: Недостаточно данных в источниках. Гипотеза основана на базовых знаниях."
        """

        try:
            logger.info("Запуск генерации гипотез через LLM...")

            result: HypothesisList = self.client.chat.completions.create(
                model=self.model_uri,
                response_model=HypothesisList,
                max_retries=3,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            hypotheses = result.hypotheses
            logger.info(f"Успешно сгенерировано гипотез: {len(hypotheses)}")

            valid_chunk_ids = {c.chunk_id for c in context}
            for hyp in hypotheses:
                hyp.source_refs = [ref for ref in hyp.source_refs if ref in valid_chunk_ids]

            return hypotheses

        except Exception as e:
            logger.error(f"Критический сбой при генерации гипотез: {str(e)}")
            return []

    def close(self):
        self.base_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
