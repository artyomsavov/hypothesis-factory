import datetime
import sys
import textwrap
import time
from pathlib import Path

import streamlit as st
from htbuilder import div, styles
from htbuilder.units import rem

from frontend.api_client import generate_hypotheses, upload_documents
from src.hypothesis_factory.base import BusinessRequest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


# ==========================================
# Настройки страницы
# ==========================================
st.set_page_config(page_title="Фабрика гипотез | R&D Assistant", page_icon="🌀", layout="wide")

# ==========================================
# Данные для быстрого старта (Suggestions)
# ==========================================
SUGGESTIONS = {}


# ==========================================
# UI-Компоненты
# ==========================================
@st.dialog("Правовая оговорка")
def show_disclaimer_dialog():
    st.caption("""
        Данный ИИ-ассистент разработан для помощи R&D-подразделениям.
        Предложенные гипотезы генерируются на основе исторических данных, патентов и публикаций.
        Любые физико-химические механизмы и дорожные карты требуют верификации и участия эксперта.
        Не вводите в систему чувствительные данные, составляющие коммерческую тайну, без развертывания on-premise.
    """)


def show_feedback_controls(hypothesis_id: str):
    """Компонент обратной связи (Human-in-the-loop)."""
    st.write("")

    with st.popover("Оценить гипотезу"):
        with st.form(key=f"feedback-{hypothesis_id}", border=False):
            with st.container():
                st.markdown(":small[Насколько гипотеза реализуема и полезна?]")
                rating = st.feedback(options="stars")

            details = st.text_area("Комментарий или причина отказа (опционально)")

            st.write("")  # Add some space

            if st.form_submit_button("Отправить отзыв"):
                # Здесь будет отправка в БД для RLHF (Human-in-the-loop)
                st.success("Отзыв сохранен. Модель учтет это в будущем!")


def clear_request():
    st.session_state.generated_results = None
    st.session_state.selected_suggestion = None
    st.session_state.kpi_input = ""
    st.session_state.constraints_input = ""


def apply_suggestion():
    suggestion = st.session_state.get("selected_suggestion")
    if suggestion:
        st.session_state.kpi_input = SUGGESTIONS[suggestion]["kpi"]
        st.session_state.constraints_input = SUGGESTIONS[suggestion]["constraints"]


# ==========================================
# Состояние сессии
# ==========================================
if "generated_results" not in st.session_state:
    st.session_state.generated_results = None
if "kpi_input" not in st.session_state:
    st.session_state.kpi_input = ""
if "constraints_input" not in st.session_state:
    st.session_state.constraints_input = ""

# ==========================================
# Отрисовка интерфейса
# ==========================================

# 1. Декоративная шапка

st.markdown(
    """
    <style>
    /* Подключаем шрифт Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    /* Применяем шрифт ко всему приложению */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }

    /* Скрываем стандартное меню Streamlit (три точки) и верхнюю полосу */
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Красивые границы для контейнеров */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #eeeef0 !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
    <div style="margin-bottom: -25px; margin-top: -10px;">
        <span class="material-symbols-rounded" style="font-size: 8rem; color: #3f3f46; line-height: 1;">
            neurology
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Оставляем только две колонки
col_title, col_btn = st.columns([1, 0.15])

with col_title:
    st.title("Фабрика гипотез", anchor=False)

with col_btn:
    if st.session_state.generated_results:
        st.button("🔄 Новый запрос", on_click=clear_request, use_container_width=True)

st.caption("R&D-конвейер: от KPI к проверяемым гипотезам с оценкой рисков и дорожной картой.")

st.divider()

# 2. Разделение на Форму (слева) и Результаты (справа)
col_input, col_results = st.columns([1, 1.8], gap="large")

with col_input:
    st.subheader("Постановка задачи", anchor=False)

    # Горизонтальный radio для быстрого выбора (замена pills)
    # st.radio(
    #     "Примеры",
    #     label_visibility="collapsed",
    #     options=list(SUGGESTIONS.keys()),
    #     index=None,
    #     horizontal=True,
    #     key="selected_suggestion",
    #     on_change=apply_suggestion,
    # )

    # Целевое свойство
    st.caption(":material/target: Целевой KPI / Проблема")
    target_kpi = st.text_area(
        "Целевой KPI / Проблема",
        placeholder="Например: Повысить жаропрочность сплава на 15%...",
        height=80,
        key="kpi_input",
        label_visibility="collapsed",
    )

    # Ограничения
    st.caption(":material/gavel: Ограничения (ресурсы, бюджет, ГОСТы)")
    constraints = st.text_area(
        "Ограничения (ресурсы, бюджет, ГОСТы)",
        placeholder="- Использовать отечественное сырье\n- Бюджет до 5 млн руб...",
        height=100,
        key="constraints_input",
        label_visibility="collapsed",
    )

    st.caption(":material/database: База знаний (контекст для RAG)")
    kb_option = st.radio(
        "Источник данных для поиска",
        options=[
            "Локальная база предприятия",
            "Загрузить новые документы",
        ],
        label_visibility="collapsed",
    )

    uploaded_files = None
    if kb_option == "Загрузить новые документы":
        uploaded_files = st.file_uploader(
            "Загрузите PDF с результатами экспериментов", accept_multiple_files=True
        )

    submitted = st.button("Генерировать гипотезы", type="primary", use_container_width=True)
    st.caption("⚠️ *Сгенерированные гипотезы требуют верификации эксперта.*")

    if submitted and target_kpi:
        st.session_state.generated_results = None  # Сброс старых результатов

        with st.status("Синтез гипотез...", expanded=True) as status:
            # Встроенная загрузка файлов
            if kb_option == "Загрузить новые документы" and uploaded_files:
                st.write("📥 Векторизация и парсинг новых документов...")
                try:
                    upload_result = upload_documents(uploaded_files)
                    st.write(
                        f"✅ Добавлено фрагментов в базу: {upload_result.get('added_chunks', 0)}"
                    )
                except Exception as e:
                    status.update(label="Ошибка загрузки", state="error", expanded=False)
                    st.error(f"Не удалось загрузить файлы: {e}")
                    st.stop()

            st.write(":material/input: Анализ ограничений и парсинг запроса...")
            time.sleep(1)
            st.write(f":material/search: Векторный поиск по источнику: *{kb_option}*...")
            time.sleep(1.5)
            st.write(":material/psychology: Генерация первичных идей (LLM Generator)...")
            time.sleep(1.5)
            st.write(":material/balance: Оценка новизны и рисков (LLM Critic)...")

            # Парсинг ограничений и формирование Pydantic-модели
            constraints_list = [c.strip("- \n") for c in constraints.split("\n") if c.strip()]
            req = BusinessRequest(target_kpi=target_kpi, constraints=constraints_list)

            try:
                results = generate_hypotheses(
                    target_kpi=req.target_kpi, constraints=req.constraints
                )
                st.session_state.generated_results = results
                status.update(label="Успешно завершено", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Ошибка генерации", state="error", expanded=False)
                st.error(f"Не удалось связаться с API. Убедитесь, что бэкенд запущен.\nДетали: {e}")
                st.stop()

        st.rerun()

# 3. Правая колонка: Дашборд результатов
with col_results:
    if not st.session_state.generated_results:
        st.container(height=1, border=False)
        st.markdown(
            ":gray[👈 Заполните параметры слева или выберите быстрый пример, "
            "чтобы запустить конвейер генерации гипотез.]"
        )
    else:
        st.subheader("Сгенерированные гипотезы", anchor=False)

        # Вывод предварительного анализа от LLM
        st.info(st.session_state.generated_results.get("preliminary_analysis", ""))

        # Динамическая отрисовка гипотез из ответа API
        for idx, hyp in enumerate(st.session_state.generated_results.get("hypotheses", [])):
            with st.container(border=True):
                st.markdown(f"##### 💡 {hyp.get('title')}")

                met1, met2, met3 = st.columns(3)
                met1.metric(label="Интегральный скор", value=f"{hyp.get('overall_score')} / 10")
                met2.metric(label="Новизна", value=f"{hyp.get('novelty_score')} / 10")
                met3.metric(label="Реализуемость", value=f"{hyp.get('feasibility_score')} / 10")

                tab_mech, tab_risks, tab_roadmap, tab_sources = st.tabs(
                    [
                        ":material/science: Механизм",
                        ":material/warning: Риски",
                        ":material/route: Дорожная карта",
                        ":material/library_books: Источники",
                    ]
                )

                with tab_mech:
                    st.markdown("**Суть гипотезы:**")
                    st.write(hyp.get("text"))
                    st.markdown("**Ожидаемый физико-химический механизм:**")
                    st.write(hyp.get("mechanism"))
                    st.markdown("**Обоснование на основе контекста (Reasoning):**")
                    st.write(hyp.get("reasoning"))

                with tab_risks:
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**Технические риски**")
                        for risk in hyp.get("technical_risks", []):
                            st.error(f"- {risk}")
                    with col_r2:
                        st.markdown("**Экономические риски**")
                        for risk in hyp.get("economic_risks", []):
                            st.warning(f"- {risk}")

                with tab_roadmap:
                    if hyp.get("road_map"):
                        st.markdown("**Последовательность проверки:**")
                        for step_idx, step in enumerate(hyp.get("road_map"), 1):
                            st.write(f"**Шаг {step_idx}:** {step}")
                    else:
                        st.write("Дорожная карта не сформирована.")

                with tab_sources:
                    st.markdown("Гипотеза опирается на следующие документы:")
                    for source in hyp.get("source_refs", []):
                        st.write(f"- 📄 **{source}**")

                show_feedback_controls(hyp.get("id"))
