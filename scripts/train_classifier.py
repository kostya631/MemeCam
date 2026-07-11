import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Алгоритм случайного леса
from sklearn.metrics import accuracy_score
import pickle
import os

# Находим датасет
CSV_FILE = "data/gesture_dataset.csv"
MODEL_FILE = "models/gesture_model.pkl"

def train_model():
    print("1. Загрузка данных...")
    if not os.path.exists(CSV_FILE):
        print(f"Ошибка: Файл {CSV_FILE} не найден. Необходимо создать датасет!")
        return

    # Читаем датасет
    df = pd.read_csv(CSV_FILE)
    
    # Сколько данных по каждому классу созданно 
    print("\nСобрано примеров по классам:")
    print(df['class_id'].value_counts())

    # Подготовка данных
    X = df.drop("class_id", axis=1) # Признаки (фичи) 
    y = df["class_id"] # Целевая переменная (таргет)

    # Делим данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    print("\n2. Обучение модели ...")
    # Создание случайного леса
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Обучение 
    model.fit(X_train, y_train)

    # Проверка точности на тестовой выборке
    print("3. Проверка точности...")
    y_pred = model.predict(X_test)
    # Находим точность предсказаний
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")

    # Сохранение модели
    os.makedirs("models", exist_ok=True) # Убеждаемся, что папка models существует
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nМодель сохранена в файл '{MODEL_FILE}'.")

if __name__ == "__main__":
    train_model()