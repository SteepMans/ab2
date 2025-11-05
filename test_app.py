import pytest
import pandas as pd
import numpy as np
from app import preprocess_data, calculate_overall_statistics, calculate_port_statistics


@pytest.fixture
def sample_titanic_data():
    """Создает тестовые данные для проверки функций"""
    data = {
        'PassengerId': [1, 2, 3, 4, 5, 6],
        'Survived': [0, 1, 1, 1, 0, 0],
        'Pclass': [3, 1, 3, 1, 3, 3],
        'Name': ['Test1', 'Test2', 'Test3', 'Test4', 'Test5', 'Test6'],
        'Sex': ['male', 'female', 'female', 'female', 'male', 'male'],
        'Age': [22.0, 38.0, np.nan, 35.0, np.nan, 54.0],
        'SibSp': [1, 1, 0, 1, 0, 0],
        'Parch': [0, 0, 0, 0, 0, 0],
        'Ticket': ['A123', 'B456', 'C789', 'D012', 'E345', 'F678'],
        'Fare': [7.25, 71.28, 7.93, 53.10, np.nan, 8.46],
        'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan],
        'Embarked': ['S', 'C', 'S', 'S', np.nan, 'Q']
    }
    return pd.DataFrame(data)


def test_preprocess_data_fills_missing_values(sample_titanic_data):
    """Тестирует, что функция preprocess_data заполняет пропущенные значения"""
    # Проверяем, что в исходных данных есть пропуски
    assert sample_titanic_data['Age'].isna().sum() > 0
    assert sample_titanic_data['Fare'].isna().sum() > 0
    assert sample_titanic_data['Embarked'].isna().sum() > 0

    # Обрабатываем данные
    processed_df = preprocess_data(sample_titanic_data)

    # Проверяем, что пропуски заполнены
    assert processed_df['Age'].isna().sum() == 0
    assert processed_df['Fare'].isna().sum() == 0
    assert processed_df['Embarked'].isna().sum() == 0

    # Проверяем, что исходный DataFrame не изменился
    assert sample_titanic_data['Age'].isna().sum() > 0


def test_calculate_overall_statistics_correct_calculations(sample_titanic_data):
    """Тестирует правильность расчета общей статистики"""
    processed_df = preprocess_data(sample_titanic_data)
    stats = calculate_overall_statistics(processed_df)

    # Проверяем структуру возвращаемых данных
    expected_keys = ['total_passengers', 'survived_count', 'died_count', 'survival_rate']
    assert all(key in stats for key in expected_keys)

    # Проверяем правильность расчетов
    assert stats['total_passengers'] == 6
    assert stats['survived_count'] == 3  # из тестовых данных выжили 3 человека
    assert stats['died_count'] == 3
    assert stats['survival_rate'] == 50.0  # 3/6 * 100 = 50%


def test_calculate_overall_statistics_empty_dataframe():
    """Тестирует поведение функции с пустым DataFrame"""
    empty_df = pd.DataFrame(columns=['Survived'])
    stats = calculate_overall_statistics(empty_df)

    assert stats['total_passengers'] == 0
    assert stats['survived_count'] == 0
    assert stats['died_count'] == 0
    assert stats['survival_rate'] == 0


def test_calculate_port_statistics_empty_dataframe():
    """Тестирует поведение функции с пустым DataFrame"""
    empty_df = pd.DataFrame(columns=['Embarked', 'Survived', 'Age'])
    port_stats = calculate_port_statistics(empty_df)

    assert isinstance(port_stats, list)
    assert len(port_stats) == 0


if __name__ == "__main__":
    pytest.main([__file__])
