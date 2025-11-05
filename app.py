import streamlit as st
import pandas as pd

# Настройка страницы
st.set_page_config(
    page_title="Анализ данных Титаника",
    layout="wide"
)

# Заголовок приложения
st.title("Интерактивный анализ данных пассажиров Титаника")
st.markdown("---")


@st.cache_data
def load_data():
    """Загрузка данных из CSV файла"""
    try:
        df = pd.read_csv('titanic_train.csv')
        return df
    except FileNotFoundError:
        st.error("Файл titanic_train.csv не найден!")
        return None


def preprocess_data(df):
    """Предварительная обработка данных"""
    df = df.copy()

    # Заполнение пропущенных значений
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    return df


def calculate_overall_statistics(df):
    """Расчет общей статистики"""
    total_passengers = len(df)
    survived_count = df['Survived'].sum()
    survival_rate = (survived_count / total_passengers * 100) if total_passengers > 0 else 0

    return {
        'total_passengers': total_passengers,
        'survived_count': survived_count,
        'died_count': total_passengers - survived_count,
        'survival_rate': survival_rate
    }


def calculate_port_statistics(df):
    """Расчет статистики по портам посадки"""
    port_names = {
        'S': 'Southampton (Саутгемптон)',
        'C': 'Cherbourg (Шербур)',
        'Q': 'Queenstown (Квинстаун)'
    }

    port_details = []

    for port in df['Embarked'].unique():
        if pd.notna(port):
            port_data = df[df['Embarked'] == port]
            port_name = port_names.get(port, port)

            survived = port_data['Survived'].sum()
            total = len(port_data)
            survival_rate = (survived / total * 100) if total > 0 else 0

            port_details.append({
                'Порт посадки': port_name,
                'Всего пассажиров': total,
                'Выжившие': survived,
                'Погибшие': total - survived,
                'Выживаемость (%)': f"{survival_rate:.1f}%",
                'Минимальный возраст': f"{port_data['Age'].min():.1f} лет",
                'Средний возраст': f"{port_data['Age'].mean():.1f} лет",
                'Максимальный возраст': f"{port_data['Age'].max():.1f} лет",
                'Медианный возраст': f"{port_data['Age'].median():.1f} лет"
            })

    return port_details


def main():
    # Загрузка данных
    df = load_data()

    if df is None:
        return

    # Предварительная обработка
    df_processed = preprocess_data(df)

    # Общая статистика
    st.subheader("Общая статистика")

    # Расчет метрик
    stats = calculate_overall_statistics(df_processed)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Всего пассажиров", stats['total_passengers'])

    with metric_col2:
        st.metric("Выжившие", stats['survived_count'])

    with metric_col3:
        st.metric("Погибшие", stats['died_count'])

    with metric_col4:
        st.metric("Выживаемость", f"{stats['survival_rate']:.1f}%")

    # Детальная информация по каждому порту
    st.subheader("Детальная информация по каждому порту")

    if not df_processed.empty:
        port_details = calculate_port_statistics(df_processed)

        if port_details:
            result_df = pd.DataFrame(port_details)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        else:
            st.write("Нет данных для отображения.")


if __name__ == "__main__":
    main()
