import logging
from typing import List

from hypothesis_factory.base import (
    BaseCritic,
    BaseGenerator,
    BasePipeline,
    BusinessRequest,
    DocumentChunk,
    Hypothesis,
    HypothesisList,
)

logger = logging.getLogger(__name__)


class HypothesisRefinementPipeline(BasePipeline):
    """
    Оркестратор (Pipeline) с циклом обратной связи (Self-Correction Loop).
    Управляет взаимодействием Генератора и Критика.
    """

    def __init__(
        self,
        generator: BaseGenerator,
        critic: BaseCritic,
        min_overall_score: float = 6.0,
        max_iterations: int = 3,
    ):
        self.generator = generator
        self.critic = critic
        self.min_overall_score = min_overall_score
        self.max_iterations = max_iterations

    def run(self, request: BusinessRequest, input_dir: str, output_path: str) -> None:
        logger.info(f"Запуск пайплайна. Вход: {input_dir}, Выход: {output_path}")
        raise NotImplementedError(
            "Для работы используйте метод `run_loop` напрямую, передавая контекст в памяти."
        )

    def run_loop(self, request: BusinessRequest, context: List[DocumentChunk]) -> List[Hypothesis]:
        logger.info("Старт итерационного процесса генерации и верификации гипотез.")

        # Шаг 1: Первичная батч-генерация
        current_hypotheses = self.generator.generate(request, context)
        if not current_hypotheses:
            logger.error("Генератор не смог предложить базовый набор гипотез.")
            return []

        accepted_hypotheses: List[Hypothesis] = []

        # Шаг 2: Цикл рецензирования
        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                f"\n=== [ИТЕРАЦИЯ {iteration} ИЗ {self.max_iterations}] Проверка Критиком ==="
            )

            # Используем стандартный публичный метод из контракта BaseCritic
            evaluated_hypotheses = self.critic.evaluate(current_hypotheses, context)
            needs_refinement: List[Hypothesis] = []

            for hyp in evaluated_hypotheses:
                # Если гипотеза была забракована (is_valid = False), её overall_score уже равен 0.0
                if hyp.overall_score >= self.min_overall_score:
                    logger.info(
                        f"Гипотеза '{hyp.title}' успешно прошла контракт. Балл: {hyp.overall_score}"
                    )
                    accepted_hypotheses.append(hyp)
                else:
                    logger.warning(
                        f"Гипотеза '{hyp.title}' отклонена ({hyp.overall_score} < {self.min_overall_score}). Направлено на доработку."
                    )
                    needs_refinement.append(hyp)

            # Завершаем цикл, если правок не требуется или исчерпан лимит
            if not needs_refinement or iteration == self.max_iterations:
                if needs_refinement:
                    logger.warning(
                        f"Достигнут лимит итераций. {len(needs_refinement)} гипотез возвращаются как есть."
                    )
                    accepted_hypotheses.extend(needs_refinement)
                break

            # Шаг 3: Генерация исправлений на основе замечаний Критика
            logger.info(
                f"Отправка {len(needs_refinement)} гипотез генератору на внесение правок..."
            )
            current_hypotheses = self._refine_hypotheses(request, context, needs_refinement)

        # Финальное ранжирование гипотез по общему баллу
        accepted_hypotheses.sort(key=lambda x: x.overall_score, reverse=True)
        logger.info(f"Пайплайн завершен. Всего гипотез к выдаче: {len(accepted_hypotheses)}")
        return accepted_hypotheses

    def _refine_hypotheses(
        self,
        request: BusinessRequest,
        context: List[DocumentChunk],
        hypotheses_to_refine: List[Hypothesis],
    ) -> List[Hypothesis]:
        """
        Внутренний метод для отправки проваленных гипотез на доработку.
        Содержит защитную проверку для работы с Dummy-моками в тестах.
        """
        # Защита от падения в unit-тестах
        if not hasattr(self.generator, "client") or not hasattr(self.generator, "model_uri"):
            logger.warning(
                "Генератор не поддерживает метод детальной доработки (Dummy). Возвращаем исходные гипотезы."
            )
            return hypotheses_to_refine

        context_blocks = [f"[ID: {c.chunk_id}]\n{c.text}" for c in context]
        context_str = (
            "\n\n---\n\n".join(context_blocks) if context_blocks else "Контекст не предоставлен."
        )

        feedback_blocks = []
        for hyp in hypotheses_to_refine:
            tech_risks = ", ".join(hyp.technical_risks) if hyp.technical_risks else "Не указаны"
            econ_risks = ", ".join(hyp.economic_risks) if hyp.economic_risks else "Не указаны"

            block = (
                f"ID: {hyp.id}\nНазвание: {hyp.title}\nСуть: {hyp.text}\nМеханизм: {hyp.mechanism}\n"
                f"КРИТИКА:\n  - Тех. риски: {tech_risks}\n  - Эконом. риски: {econ_risks}\n"
                f"  - Оценки: Новизна {hyp.novelty_score}, Реализуемость {hyp.feasibility_score}"
            )
            feedback_blocks.append(block)

        feedback_str = "\n\n====================\n\n".join(feedback_blocks)
        constraints_str = (
            ", ".join(request.constraints) if request.constraints else "Нет жестких ограничений."
        )

        system_prompt = (
            "Ты — R&D инженер. Твоя задача — скорректировать технические гипотезы, которые не прошли рецензию.\n"
            "1. Измени параметры так, чтобы полностью закрыть технические и экономические риски.\n"
            "2. Сохраняй исходные `id` гипотез.\n"
            "3. Ответ должен быть JSON-объектом схемы HypothesisList.\n"
            "4. Обязательно начни с заполнения поля `preliminary_analysis`."
        )

        user_prompt = (
            f"БИЗНЕС-ЦЕЛЬ:\n{request.target_kpi}\n\nОГРАНИЧЕНИЯ:\n{constraints_str}\n\n"
            f"КОНТЕКСТ:\n{context_str}\n\nГИПОТЕЗЫ И ЗАМЕЧАНИЯ:\n{feedback_str}"
        )

        try:
            result: HypothesisList = self.generator.client.chat.completions.create(
                model=self.generator.model_uri,
                response_model=HypothesisList,
                max_retries=3,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            valid_chunk_ids = {c.chunk_id for c in context}
            for hyp in result.hypotheses:
                hyp.source_refs = [ref for ref in hyp.source_refs if ref in valid_chunk_ids]

            return result.hypotheses

        except Exception as e:
            logger.error(f"Критическая ошибка генерации правок: {str(e)}")
            return hypotheses_to_refine
