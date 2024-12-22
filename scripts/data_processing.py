"""
Скрипт для предварительной обработки данных продаж аптечной сети.
Выполняет загрузку, очистку и преобразование данных.
"""

import pandas as pd
import time


def print_execution_time(start_time):
    """Вывод времени выполнения операции"""
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")


# Проверка структуры данных на маленьком наборе
print("Проверка структуры данных...")
start_time = time.time()
data_small = pd.read_csv("../data/sales.csv", nrows=5)
columns = data_small.columns.tolist()
print("Доступные колонки:", columns)
print_execution_time(start_time)

# Загрузка полного набора данных с выбранными колонками
print("\nЗагрузка полного набора данных...")
start_time = time.time()
cols = [
    "sale_date_date",
    "contact_id",
    "shop_id",
    "product_id",
    "product_sub_category_id",
    "product_category_id",
    "brand_id",
    "quantity",
]
data = pd.read_csv("../data/sales.csv", usecols=cols)
print("Информация о загруженных данных:")
print(data.info(verbose=False))
print_execution_time(start_time)

# Предварительная очистка данных
print("\nОчистка и преобразование данных...")
start_time = time.time()

# Преобразование типов данных
print("1. Преобразование типов данных...")
# Преобразование количества из строки с запятой в число
data["quantity"] = data["quantity"].str.replace(",", ".", regex=False).astype("float")
# Преобразование даты в datetime
data["sale_date_date"] = data["sale_date_date"].astype("datetime64")
# Преобразование ID колонок в целые числа
id_columns = [
    "contact_id",
    "shop_id",
    "product_id",
    "product_sub_category_id",
    "product_category_id",
    "brand_id",
]
data[id_columns] = data[id_columns].astype(int)

# Фильтрация некорректных значений
print("2. Фильтрация некорректных значений...")
filters = {
    "quantity": data["quantity"] != -1,
    "product_id": data["product_id"] != -1,
    "product_sub_category_id": data["product_sub_category_id"] != -1,
    "product_category_id": data["product_category_id"] != -1,
}

# Применение всех фильтров
data = data[pd.concat(filters.values(), axis=1).all(axis=1)]
print("Размер данных после фильтрации:", data.shape)

print_execution_time(start_time)

# Сохранение очищенных данных
print("\nСохранение очищенных данных...")
start_time = time.time()
data.to_csv("../data/sales.csv", index=False)
print("Данные сохранены в data/sales.csv")
print_execution_time(start_time)

print("\nОбработка данных завершена!")
