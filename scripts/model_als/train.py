import time

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Создание DataFrame
data = spark.read.csv("../../data/sales.csv", header=True, inferSchema=True)
data = (
    data.select("sale_date_date", "contact_id", "product_id", "quantity")
    .withColumn(
        "quantity", F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))
    )
    .withColumnRenamed(existing="product_id", new="item_id")
    .withColumnRenamed(existing="contact_id", new="user_id")
    .withColumn("week_of_year", F.weekofyear(F.col("sale_date_date")))
)


def sample_by_week(df, week_col_name, split_size_weeks):
    """Выборка данных за последние n недель"""
    threshold_week = (
        int(data.select(F.max(week_col_name)).collect()[0][0]) - split_size_weeks
    )
    df_before = df.filter(F.col(week_col_name) < threshold_week)
    df_after = df.filter(F.col(week_col_name) >= threshold_week)
    return df_before, df_after


# Отбираем только 15 последних недель для обучения
print("Подготовка данных...")
before, data = sample_by_week(
    df=data, week_col_name="week_of_year", split_size_weeks=15
)


def basic_statistics_of_data():
    """Вывод базовой статистики по данным"""
    numerator = data.select("quantity").count()
    num_users = data.select("user_id").distinct().count()
    num_items = data.select("item_id").distinct().count()
    denominator = num_users * num_items
    sparsity = (1.0 - (numerator * 1.0) / denominator) * 100
    return spark.createDataFrame(
        data=[
            (
                "общее количество строк",
                str("{0:,}".format(numerator).replace(",", "'")),
            ),
            (
                "количество пользователей",
                str("{0:,}".format(num_users).replace(",", "'")),
            ),
            ("количество товаров", str("{0:,}".format(num_items).replace(",", "'"))),
            ("разреженность", str(sparsity)[:5] + "% пустых"),
        ],
        schema=["статистика", "значение"],
    )


print("\nСтатистика данных:")
basic_statistics_of_data().show(truncate=False)

# Разделение на тестовую и обучающую выборки
print("\nРазделение данных на выборки...")
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)

# Создание и обучение ALS модели
print("\nОбучение модели...")
als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="quantity",
    nonnegative=True,
    implicitPrefs=True,
    coldStartStrategy="drop",
)
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="quantity", predictionCol="prediction"
)

start = time.time()
model = als.fit(train)
print("Время обучения = " + str(time.time() - start))

# Сохранение модели
print("\nСохранение модели...")
model.save("../models/als_model_2024_12_15.model")

# Создание таблицы с реальными и предсказанными товарами
print("\nПодготовка метрик качества...")
train_actual_items = (
    train.select("user_id", "item_id")
    .groupBy("user_id")
    .agg(F.collect_list(col="item_id"))
    .withColumnRenamed(existing="collect_list(item_id)", new="actual")
)

train_recs_items = model.recommendForAllUsers(numItems=5).select(
    "user_id", F.col("recommendations.item_id").alias("recs_ALS")
)

result = train_actual_items.join(other=train_recs_items, on="user_id", how="inner")

# Метрики качества
test_predictions = model.transform(test)
metrics = RankingMetrics(
    predictionAndLabels=result.select("actual", "recs_ALS").rdd.map(tuple)
)
metrics_df = spark.createDataFrame(
    data=[
        ("RMSE", evaluator.evaluate(test_predictions)),
        ("precision@k", metrics.precisionAt(5)),
        ("ndcg@k", metrics.ndcgAt(5)),
        ("meanAVGPrecision", metrics.meanAveragePrecision),
    ],
    schema=["метрика", "значение"],
)

print("\nМетрики качества модели:")
metrics_df.withColumn("значение", F.round("значение", 5)).show(truncate=False)

# Вывод параметров модели
print("\nПараметры модели:")
spark.createDataFrame(
    data=[
        ("Rank", str(model.rank)),
        ("MaxIter", str(als.getMaxIter())),
        ("RegParam", str(als.getRegParam())),
    ],
    schema=["параметр", "значение"],
).show()
