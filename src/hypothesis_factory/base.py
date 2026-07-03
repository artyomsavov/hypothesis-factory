from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

# ==========================================
# Data Contracts
# ==========================================


class DocumentMetadata(BaseModel):
    """Метаданные источника для обеспечения цитируемости и прозрачности."""

    source_id: str = Field(..., description="Уникальный идентификатор (имя файла или хэш)")
    source_type: str = Field(..., description="Тип документа: article, patent, report, table")
    title: Optional[str] = Field(default=None, description="Название документа")
    authors: Optional[List[str]] = Field(default=None, description="Авторы или организация")


class DocumentChunk(BaseModel):
    """Базовая единица контекста, подаваемая в LLM."""

    chunk_id: str = Field(..., description="Уникальный ID чанка текста")
    text: str = Field(..., description="Сырой текст или markdown-таблица")
    metadata: DocumentMetadata


class BusinessRequest(BaseModel):
    """Формализованный бизнес-запрос от пользователя."""

    target_kpi: str = Field(..., description="Цель: например, 'повысить жаропрочность на 15%'")
    constraints: List[str] = Field(..., description="Ограничения: бюджет, сырье, оборудование")


class Hypothesis(BaseModel):
    """
    Строгий контракт итоговой гипотезы.
    Описания (description) критически важны — они отправляются в LLM как инструкции.
    """

    id: str = Field(..., description="Уникальный строковый идентификатор гипотезы")
    title: str = Field(..., description="Краткое название гипотезы (1-2 предложения)")
    text: str = Field(
        ..., description="Подробное описание: что добавить, в какой пропорции, режим обработки"
    )
    mechanism: str = Field(
        ..., description="Ожидаемый физико-химический или металлургический механизм влияния"
    )
    reasoning: str = Field(..., description="Детальное обоснование гипотезы на основе контекста")
    source_refs: List[str] = Field(
        ..., description="Список source_id из контекста, на которые опирается гипотеза"
    )

    # Блок оценки (заполняется модулем Critic)
    novelty_score: float = Field(default=0.0, description="Оценка новизны от 0 до 10")
    feasibility_score: float = Field(default=0.0, description="Оценка реализуемости от 0 до 10")
    technical_risks: List[str] = Field(
        default_factory=list, description="Список возможных технических проблем"
    )
    economic_risks: List[str] = Field(
        default_factory=list, description="Список возможных экономических проблем"
    )
    overall_score: float = Field(default=0.0, description="Интегральная оценка гипотезы")

    # Опциональный блок из ТЗ
    road_map: Optional[List[str]] = Field(
        default=None, description="Шаги для лабораторной проверки (дорожная карта)"
    )


class HypothesisList(BaseModel):
    """Обертка для удобного парсинга батч-генерации с обязательным CoT."""

    preliminary_analysis: str = Field(
        ...,
        description="Пошаговый анализ: жесткое сопоставление ограничений из бизнес-цели с физическими свойствами из контекста.",
    )
    hypotheses: List[Hypothesis]


# ==========================================
# Interfaces
# ==========================================


class BaseReader(ABC):
    """Парсер входящих данных. Переводит гетерогенные файлы в стандартизированные чанки."""

    @abstractmethod
    def read(self, file_path: str) -> List[DocumentChunk]:
        """Считывает файл и возвращает список размеченных кусков текста с метаданными."""
        pass


class BaseRetriever(ABC):
    """Модуль поиска (RAG). Фильтрует базу знаний под конкретный запрос."""

    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """Добавляет чанки в векторную базу данных или индекс."""
        pass

    @abstractmethod
    def get_context(self, request: BusinessRequest, top_k: int = 5) -> List[DocumentChunk]:
        """Ищет релевантные куски текста под конкретный бизнес-запрос."""
        pass


class BaseGenerator(ABC):
    """Ядро генерации. Принимает запрос + контекст, отдает сырые гипотезы."""

    @abstractmethod
    def generate(self, request: BusinessRequest, context: List[DocumentChunk]) -> List[Hypothesis]:
        """Генерирует гипотезы на основе контекста (без финального скоринга)."""
        pass


class BaseCritic(ABC):
    """Модуль фильтрации и ранжирования."""

    @abstractmethod
    def evaluate(
        self, hypotheses: List[Hypothesis], context: List[DocumentChunk]
    ) -> List[Hypothesis]:
        """
        Проверяет гипотезы на бред, выставляет оценки (novelty, feasibility),
        считает overall_score и сортирует список.
        """
        pass


class BasePipeline(ABC):
    """Оркестратор. Связывает модули, обрабатывает исключения, формирует выгрузку."""

    @abstractmethod
    def run(self, request: BusinessRequest, input_dir: str, output_path: str) -> None:
        """Запускает полный цикл от чтения файлов до генерации отчета."""
        pass
