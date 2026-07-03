import streamlit as st
import time

# Импортируем контракты напрямую из твоего модуля
# Убедись, что папка src в sys.path или проект установлен через uv
from hypothesis_factory.base import BusinessRequest, Hypothesis

# Настройка страницы
st.set_page_config(page_title="Hypothesis Factory", page_icon="🏭", layout="wide")

st.title("🏭 Hypothesis Factory")
st.markdown("### Интеллектуальный генератор научно-технических гипотез")

# === 1. САЙДБАР: INGESTION ===
with st.sidebar:
    st.header("1. База знаний")
    uploaded_files = st.file_uploader(
        "Загрузите материалы (PDF/TXT)",
        accept_multiple_files=True,
        help="Документы будут обработаны через PyMuPDFReader и векторизованы."
    )
    
    if st.button("📥 Индексировать документы", use_container_width=True):
        if uploaded_files:
            # Заглушка работы Ingestion
            with st.spinner("Конвертация в Markdown и разбивка на чанки..."):
                time.sleep(1.5)
            with st.spinner("Векторизация (multilingual-e5-large)..."):
                time.sleep(2)
            st.success(f"Успешно обработано файлов: {len(uploaded_files)}. Векторная база обновлена.")
        else:
            st.warning("Сначала загрузите файлы.")

# === 2. ГЛАВНЫЙ ЭКРАН: BUSINESS REQUEST ===
st.header("2. Бизнес-запрос")
col1, col2 = st.columns(2)

with col1:
    target_kpi = st.text_input(
        "🎯 Цель (Target KPI)",
        value="Повысить жаропрочность титанового сплава на 15%",
        help="Какую конкретно метрику мы хотим улучшить?"
    )

with col2:
    constraints_input = st.text_input(
        "🚧 Ограничения (через запятую)",
        value="Бюджет ограничен, без использования дорогих компонентов",
        help="Сырье, бюджет, производственные ограничения"
    )

# === 3. ЗАПУСК PIPELINE ===
if st.button("🚀 Сгенерировать гипотезы", type="primary", use_container_width=True):
    # Упаковываем данные интерфейса в наш строгий контракт
    constraints = [c.strip() for c in constraints_input.split(",")]
    request = BusinessRequest(target_kpi=target_kpi, constraints=constraints)

    # Визуализация работы Pipeline (заглушки)
    with st.status("Запуск оркестратора (Pipeline)...", expanded=True) as status:
        st.write("🔍 Поиск контекста (Retrieval: гибридный поиск E5 + BM25)...")
        time.sleep(1.5)
        
        st.write("🧠 Генерация гипотез (Generator + структурированный вывод Pydantic)...")
        time.sleep(2.5)
        
        st.write("⚖️ Оценка и фильтрация на галлюцинации (Critic)...")
        time.sleep(1.5)
        
        status.update(label="Пайплайн успешно завершен!", state="complete", expanded=False)

    # === 4. ВЫВОД РЕЗУЛЬТАТОВ ===
    st.header("3. Результаты генерации")

    # Создаем идеальный мок-объект гипотезы, заполненный по всем правилам base.py
    mock_hypothesis = Hypothesis(
        id="HYP-001",
        title="Микролегирование ниобием с изотермической ковкой",
        text="Добавление 0.5% ниобия в расплав с последующей термообработкой при 850°C в течение 4 часов.",
        mechanism="Ниобий способствует выделению мелкодисперсных карбидных фаз, которые блокируют движение дислокаций при высоких температурах.",
        reasoning="В контексте указано, что ниобий повышает прочность на 20% за счет измельчения зерна. Это удовлетворяет KPI и не нарушает ограничение по бюджету, так как доля присадки минимальна.",
        source_refs=["doc_titanium_research_v2.pdf", "patent_ru_2024.txt"],
        novelty_score=7.5,
        feasibility_score=9.0,
        technical_risks=["Требуется жесткий контроль температуры расплава", "Возможное снижение пластичности"],
        economic_risks=["Слегка усложняется цикл термообработки"],
        overall_score=8.25,
        road_map=["1. Компьютерное моделирование фазового состава", "2. Выплавка опытного слитка 5 кг", "3. Тест на длительную прочность"]
    )

    # Отрисовка интерфейса карточки гипотезы
    with st.expander(f"🟢 {mock_hypothesis.title} (Оценка: {mock_hypothesis.overall_score})", expanded=True):
        st.markdown(f"**📝 Суть:** {mock_hypothesis.text}")
        st.markdown(f"**⚙️ Механизм:** {mock_hypothesis.mechanism}")
        st.markdown(f"**🧠 Обоснование (от Generator):** {mock_hypothesis.reasoning}")
        
        st.divider()
        
        # Блок от модуля Critic
        st.markdown("### Оценка (Critic)")
        score_col1, score_col2 = st.columns(2)
        with score_col1:
            st.metric("💡 Оценка новизны", mock_hypothesis.novelty_score)
            st.markdown("**Технические риски:**")
            for risk in mock_hypothesis.technical_risks:
                st.markdown(f"- {risk}")
                
        with score_col2:
            st.metric("🛠 Реализуемость", mock_hypothesis.feasibility_score)
            st.markdown("**Экономические риски:**")
            for risk in mock_hypothesis.economic_risks:
                st.markdown(f"- {risk}")
                
        st.divider()
        st.markdown("### 🗺 Дорожная карта внедрения")
        for step in mock_hypothesis.road_map:
            st.markdown(f"- {step}")
            
        st.caption(f"📚 Опирается на источники: {', '.join(mock_hypothesis.source_refs)}")