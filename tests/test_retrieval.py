from unittest.mock import MagicMock, patch

import numpy as np

from hypothesis_factory.base import BusinessRequest, DocumentChunk, DocumentMetadata
from hypothesis_factory.retrieval import ChromaRetriever


@patch("hypothesis_factory.retrieval.chromadb.PersistentClient")
@patch("hypothesis_factory.retrieval.chromadb.Client")
@patch("hypothesis_factory.retrieval.SentenceTransformer")
def test_retriever_initialization_modes(mock_st, mock_ephemeral_client, mock_persistent_client):
    """
    Проверяем, что класс корректно переключается между постоянной БД на диске
    и временной сессионной БД в ОЗУ в зависимости от переданного пути.
    """
    # 1. Сценарий: Задан путь (Persistent DB)
    retriever_persistent = ChromaRetriever(db_path="/fake/path")
    mock_persistent_client.assert_called_once_with(path="/fake/path")
    mock_ephemeral_client.assert_not_called()

    mock_persistent_client.reset_mock()

    # 2. Сценарий: Путь не задан (Ephemeral DB - временная)
    retriever_memory = ChromaRetriever(db_path=None)
    mock_ephemeral_client.assert_called_once()
    mock_persistent_client.assert_not_called()


@patch("hypothesis_factory.retrieval.SentenceTransformer")
@patch("hypothesis_factory.retrieval.chromadb.Client")
def test_retriever_add_documents(mock_client_cls, mock_st):
    """
    Проверяем, что метод add_documents корректно распаковывает Pydantic-модели,
    векторизует текст и отправляет правильные словари в метод .add() ChromaDB.
    """
    # 1. Настройка моков
    mock_collection = MagicMock()
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    mock_embed_instance = mock_st.return_value
    # Имитируем возврат numpy массива для 2 чанков (как это делает SentenceTransformer)
    mock_embed_instance.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    retriever = ChromaRetriever(db_path=None)

    # 2. Подготовка входных данных (два моковых чанка)
    chunk1 = DocumentChunk(
        chunk_id="chunk_1",
        text="Сплав ВЖ159",
        metadata=DocumentMetadata(source_id="doc1.pdf", source_type="article", title="Заголовок 1"),
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_2",
        text="Термообработка",
        metadata=DocumentMetadata(source_id="doc2.pdf", source_type="patent", title="Заголовок 2"),
    )

    # 3. Вызов метода
    retriever.add_documents([chunk1, chunk2])

    # 4. Проверка вызова модели векторизации
    mock_embed_instance.encode.assert_called_once()
    call_args = mock_embed_instance.encode.call_args[0][0]
    assert call_args == ["Сплав ВЖ159", "Термообработка"], (
        "В модель должны уйти только чистые тексты"
    )

    # 5. Проверка записи в БД
    mock_collection.add.assert_called_once()
    add_kwargs = mock_collection.add.call_args[1]

    assert add_kwargs["documents"] == ["Сплав ВЖ159", "Термообработка"]
    assert add_kwargs["ids"] == ["chunk_1", "chunk_2"]
    assert len(add_kwargs["metadatas"]) == 2
    assert add_kwargs["metadatas"][0]["source_id"] == "doc1.pdf"
    assert add_kwargs["metadatas"][1]["source_type"] == "patent"

    # Проверка, что numpy array успешно конвертировался в list of lists через .tolist()
    assert add_kwargs["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@patch("hypothesis_factory.retrieval.SentenceTransformer")
@patch("hypothesis_factory.retrieval.chromadb.Client")
def test_retriever_get_context(mock_client_cls, mock_st):
    """
    Проверяем, что get_context корректно склеивает KPI и ограничения для поиска,
    а затем правильно оборачивает ответ ChromaDB обратно в контракты DocumentChunk.
    """
    # 1. Настройка моков БД
    mock_collection = MagicMock()
    # Эмулируем типичную вложенную структуру ответа ChromaDB
    mock_collection.query.return_value = {
        "ids": [["chunk_123"]],
        "documents": [["Релевантный текст для гипотезы"]],
        "metadatas": [
            [{"source_id": "test_patent.pdf", "source_type": "patent", "title": "Test Title"}]
        ],
    }
    mock_client_cls.return_value.get_or_create_collection.return_value = mock_collection

    # Настройка мока энкодера
    mock_embed_instance = mock_st.return_value
    mock_embed_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    retriever = ChromaRetriever(db_path=None)

    # 2. Подготовка запроса от фронтенда
    request = BusinessRequest(target_kpi="Снизить себестоимость", constraints=["Использовать ГОСТ"])

    # 3. Вызов метода
    chunks = retriever.get_context(request, top_k=1)

    # 4. Проверки
    # Проверка формирования строки поиска
    mock_embed_instance.encode.assert_called_once_with(
        ["Снизить себестоимость Ограничения: Использовать ГОСТ"]
    )

    # Проверка правильности упаковки в Pydantic
    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].chunk_id == "chunk_123"
    assert chunks[0].text == "Релевантный текст для гипотезы"
    assert chunks[0].metadata.source_id == "test_patent.pdf"
