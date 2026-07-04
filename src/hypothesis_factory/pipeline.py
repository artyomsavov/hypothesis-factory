import logging
from typing import List
import os

from hypothesis_factory.base import (
    BasePipeline,
    BusinessRequest,
    DocumentChunk,
    Hypothesis,
    HypothesisList,
)
from hypothesis_factory.generator import YandexGPTGenerator
from hypothesis_factory.critic import YandexGPTCritic

logger = logging.getLogger(__name__)


class HypothesisRefinementPipeline(BasePipeline):
    """
    Оркестратор (Pipeline) с циклом обратной связи (Self-Correction Loop).
    Управляет взаимодействием Генератора и Критика, отправляя гипотезы на доработку 
    в случае обнаружения критических рисков или низких оценок.
    """

    def __init__(
        self,
        generator: YandexGPTGenerator,
        critic: YandexGPTCritic,
        min_overall_score: float = 6.0,
        max_iterations: int = 3,
    ):
        self.generator = generator
        self.critic = critic
        self.min_overall_score = min_overall_score
        self.max_iterations = max_iterations

    def run(self, request: BusinessRequest, input_dir: str, output_path: str) -> None:
        """
        Реализация абстрактного метода из BasePipeline.
        В реальной R&D системе здесь бы происходил вызов BaseReader и BaseRetriever.
        Для интерактивной работы в Jupyter Notebook используйте метод `run_loop`.
        """
        logger.info(f"Запуск пайплайна. Вход: {input_dir}, Выход: {output_path}")
        raise NotImplementedError(
            "Для работы в Jupyter Notebook вызывайте метод `run_loop` напрямую, передавая контекст в памяти."
        )

    def run_loop(self, request: BusinessRequest, context: List[DocumentChunk]) -> List[Hypothesis]:
        """
        Основной рабочий цикл для Jupyter Notebook. 
        Проводит гипотезы через итеративный процесс генерации, критики и фильтрации/исправления.
        """
        logger.info("Старт итерационного процесса генерации и верификации гипотез.")

        # Шаг 1: Первичная батч-генерация гипотез
        current_hypotheses = self.generator.generate(request, context)
        if not current_hypotheses:
            logger.error("Генератор не смог предложить базовый набор гипотез.")
            return []

        accepted_hypotheses: List[Hypothesis] = []

        # Шаг 2: Цикл рецензирования и исправления (до max_iterations раз)
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n=== [ИТЕРАЦИЯ {iteration} ИЗ {self.max_iterations}] Проверка гипотез Критиком ===")
            
            needs_refinement: List[Hypothesis] = []
            
            for hyp in current_hypotheses:
                # Используем внутренний метод критика для оценки одиночной гипотезы,
                # чтобы перехватить детальные риски и статус валидности (is_valid=False)
                eval_result = self.critic._score_single_hypothesis(hyp, context)
                
                # Обогащаем объект гипотезы метаданными от критика
                hyp.novelty_score = eval_result.novelty_score
                hyp.feasibility_score = eval_result.feasibility_score
                hyp.technical_risks = eval_result.technical_risks
                hyp.economic_risks = eval_result.economic_risks
                
                # Проверяем выполнение условий контракта
                if not eval_result.is_valid:
                    hyp.overall_score = 0.0
                    logger.warning(f"Забраковано критиком: '{hyp.title}' (Нарушение физики/логики). Направлено на переработку.")
                    needs_refinement.append(hyp)
                else:
                    hyp.overall_score = round((eval_result.novelty_score + eval_result.feasibility_score) / 2, 2)
                    
                    if hyp.overall_score < self.min_overall_score:
                        logger.info(
                            f"Гипотеза '{hyp.title}' валидна, но балл ниже целевого ({hyp.overall_score} < {self.min_overall_score}). Направлено на доработку."
                        )
                        needs_refinement.append(hyp)
                    else:
                        logger.info(f"Гипотеза '{hyp.title}' успешно прошла контракт. Балл: {hyp.overall_score}")
                        accepted_hypotheses.append(hyp)

            # Если все гипотезы прошли контракт или это последняя попытка — завершаем цикл
            if not needs_refinement or iteration == self.max_iterations:
                if needs_refinement:
                    logger.warning(
                        f"Достигнут лимит в {self.max_iterations} правок. {len(needs_refinement)} гипотез возвращаются пользователю в текущем виде."
                    )
                    accepted_hypotheses.extend(needs_refinement)
                break

            # Шаг 3: Генерация исправлений на основе замечаний Критика
            logger.info(f"Отправка {len(needs_refinement)} гипотез генератору на внесение правок...")
            current_hypotheses = self._refine_hypotheses(request, context, needs_refinement)

        # Финальное ранжирование гипотез по общему баллу
        accepted_hypotheses.sort(key=lambda x: x.overall_score, reverse=True)
        logger.info(f"Пайплайн завершен. Всего гипотез к выдаче: {len(accepted_hypotheses)}")
        return accepted_hypotheses

    def _refine_hypotheses(
        self, request: BusinessRequest, context: List[DocumentChunk], hypotheses_to_refine: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Внутренний метод для отправки проваленных гипотез и замечаний Критика обратно в LLM-генератор.
        """
        context_blocks = []
        for chunk in context:
            context_blocks.append(f"[ID: {chunk.chunk_id}]\n{chunk.text}")
        context_str = "\n\n---\n\n".join(context_blocks) if context_blocks else "Контекст не предоставлен."

        # Аккумулируем замечания Критика по каждой проблемной гипотезе
        feedback_blocks = []
        for hyp in hypotheses_to_refine:
            block = (
                f"Идентификатор гипотезы: {hyp.id}\n"
                f"Название: {hyp.title}\n"
                f"Исходное описание суть: {hyp.text}\n"
                f"Предложенный механизм: {hyp.mechanism}\n"
                f"КРИТИКА И ЗАМЕЧАНИЯ:\n"
                f"  - Технические риски: {', '.join(hyp.technical_risks) if hyp.technical_risks else 'Не указаны'}\n"
                f"  - Экономические риски: {', '.join(hyp.economic_risks) if hyp.economic_risks else 'Не указаны'}\n"
                f"  - Оценка новизны: {hyp.novelty_score}, Реализуемости: {hyp.feasibility_score}"
            )
            feedback_blocks.append(block)
        feedback_str = "\n\n====================\n\n".join(feedback_blocks)

        system_prompt = """
        Ты — ведущий R&D инженер и научный исследователь.
        Твоя задача — скорректировать и доработать технические гипотезы, которые не прошли жесткое рецензирование у Главного Инженера (Критика).
        Тебе будут предоставлены исходные варианты гипотез и подробный список рисков/замечаний.

        Обязательные правила доработки:
        1. Внимательно изучи каждый технический и экономический риск. Измени параметры, материалы, геометрию или физико-химический режим обработки в гипотезах так, чтобы полностью закрыть эти уязвимости.
        2. Сохраняй исходные строковые `id` гипотез в ответе, чтобы система могла сопоставить исправленные варианты.
        3. Ответ должен быть строго в виде структурированного JSON-объекта схемы HypothesisList.
        4. Обязательно начни с заполнения поля `preliminary_analysis`, где пошагово разбери замечания Критика и опиши, как именно ты их устраняешь.
        """

        constraints_str = ", ".join(request.constraints) if request.constraints else "Нет жестких ограничений."

        user_prompt = f"""
        БИЗНЕС-ЦЕЛЬ:
        {request.target_kpi}

        ОГРАНИЧЕНИЯ:
        {constraints_str}

        КОНТЕКСТ (Источники):
        {context_str}

        ГИПОТЕЗЫ И ЗАМЕЧАНИЯ КРИТИКА ДЛЯ ИСПРАВЛЕНИЯ:
        {feedback_str}

        ИНСТРУКЦИЯ:
        1. Изучи замечания Критика.
        2. Заполни поле `preliminary_analysis` с разбором стратегии исправления.
        3. Сформируй массив `hypotheses` с исправленными версиями. Поля оценок (`novelty_score`, `feasibility_score`, `overall_score`) оставь равными 0.0 — их пересчитает Критик на следующем шаге.
        """

        try:
            # Используем инстанс instructor клиента из генератора
            result: HypothesisList = self.generator.client.chat.completions.create(
                model=self.generator.model_uri,
                response_model=HypothesisList,
                max_retries=3,
                temperature=0.3,  # Снижаем температуру для более точной работы над ошибками
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            # Проверяем source_refs на корректность относительно исходного контекста
            valid_chunk_ids = {c.chunk_id for c in context}
            for hyp in result.hypotheses:
                hyp.source_refs = [ref for ref in hyp.source_refs if ref in valid_chunk_ids]

            return result.hypotheses

        except Exception as e:
            logger.error(f"Критическая ошибка при попытке генерации правок: {str(e)}")
            # Если сеть упала, возвращаем текущие гипотезы, чтобы не ломать пайплайн
            return hypotheses_to_refine