# Импортируем библиотеку Pandas
import pandas as pd

# Устанавливаем пакет ucimlrepo
!pip install ucimlrepo

# Подключаемся к репозиторию UCI Machine Learning Repository
from ucimlrepo import fetch_ucirepo

# Загружаем датасет Adult Dataset
adult_dataset = fetch_ucirepo(id=2)

# Объединяем признаки и целевую переменную в единый DataFrame
dataset = pd.concat([adult_dataset.data.features, adult_dataset.data.targets], axis=1)

# Проверяем первые строки объединенного набора данных
print(dataset.head())

# Получаем размерность набора данных
print("Размерность:", dataset.shape)

# Проверяем наличие пропущенных значений
missing_columns = [col for col in dataset.columns if dataset[col].isnull().sum() > 0]
if missing_columns:
    print("Столбцы с пропусками:", missing_columns)
else:
    print("Нет пропусков")

# Анализируем распределение категорий в столбце 'race'
race_distribution = dataset['race'].value_counts()
print("Распределение рас:")
print(race_distribution)

# Определяем медианное значение рабочего времени в неделю
weekly_hours_median = dataset['hours-per-week'].median()
print(f"Медиана рабочих часов в неделю: {weekly_hours_median}")

# Сравниваем количество мужчин и женщин с доходом больше указанного порога
male_above_threshold = len(dataset[(dataset['sex'] == 'Male') & (dataset['income'] != '>50K')])
female_above_threshold = len(dataset[(dataset['sex'] == 'Female') & (dataset['income'] != '>50K')])

if male_above_threshold > female_above_threshold:
    print("Мужчин больше")
elif male_above_threshold < female_above_threshold:
    print("Женщин больше")
else:
    print("Одинаковое количество")

# Заполняем пропущенные значения наиболее частым значением в каждом столбце
dataset['workclass'] = dataset['workclass'].fillna(dataset['workclass'].mode()[0])
dataset['occupation'] = dataset['occupation'].fillna(dataset['occupation'].mode()[0])
dataset['native-country'] = dataset['native-country'].fillna(dataset['native-country'].mode()[0])

# Проверяем наличие оставшихся пропусков
remaining_nulls = dataset.isnull().any().sum()
print(f"Пропусков осталось: {remaining_nulls}")