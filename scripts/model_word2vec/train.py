import time

from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, StringType

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка данных
data = spark.read.csv("../../data/sales.csv", header=True, inferSchema=True)

# Подготовка данных
# Преобразование contact_id в StringType и sale_date_date в DateType
data = (
    data.select("sale_date_date", "contact_id", "shop_id", "product_id", "quantity")
    .withColumn(colName="sale_date_date", col=data["sale_date_date"].cast(DateType()))
    .withColumn(colName="product_id", col=data["product_id"].cast(StringType()))
)

# Извлечение 90% ID клиентов для обучения
users = data.select("contact_id").distinct()
(users_train, users_valid) = users.randomSplit(weights=[0.9, 0.1], seed=5)

print("Количество пользователей для обучения:", users_train.count())
print("Количество пользователей для валидации:", users_valid.count())

# Разделение данных на обучающую и валидационную выборки
train_df = data.join(other=users_train, on="contact_id", how="inner")
validation_df = data.join(other=users_valid, on="contact_id", how="inner")

print("Количество строк в обучающей выборке:", train_df.count())
print("Количество строк в валидационной выборке:", validation_df.count())


# Создание колонки с номером чека и удаление лишних колонок
def create_col_orders(df):
    return (
        df.select(
            F.concat_ws("_", data.sale_date_date, data.shop_id, data.contact_id).alias(
                "order_id"
            ),
            "product_id",
            "quantity",
        )
        .groupBy("order_id")
        .agg(F.collect_list(col="product_id"))
        .withColumnRenamed(existing="collect_list(product_id)", new="actual_products")
    )


train_orders = create_col_orders(df=train_df)
validation_orders = create_col_orders(df=validation_df)

# Обучение Word2Vec модели
word2Vec = Word2Vec(
    vectorSize=100,  # Размерность векторного пространства
    minCount=5,  # Минимальное количество появлений слова
    numPartitions=1,  # Количество партиций для параллельных вычислений
    seed=33,  # Seed для воспроизводимости результатов
    windowSize=3,  # Размер окна контекста
    inputCol="actual_products",
    outputCol="result",
)

print("Начало обучения модели...")
start = time.time()
model = word2Vec.fit(dataset=train_orders)
print("Время обучения = " + str(time.time() - start))

# Сохранение модели
print("Сохранение модели...")
model.save("../models/word2vec_model_2024_12_15")
print("Модель успешно сохранена")
