from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка моделей и данных
print("Загрузка моделей и данных...")
w2v_model = Word2VecModel.load("models/word2vec_model_2024_12_15")
product_vectors = w2v_model.getVectors().withColumnRenamed(
    existing="word", new="product_id"
)

products = spark.read.csv(
    "data/products.csv", header=True, inferSchema=True
).withColumn("name", F.regexp_replace("name", r"(\(\d+\) )", ""))


def kmeans_model_fit(k, dataset):
    """Обучение K-means модели"""
    kmeans = KMeans(featuresCol="vector", maxIter=20, seed=3)
    kmeans_model = kmeans.fit(dataset=dataset, params={kmeans.k: k})
    predictions = kmeans_model.transform(dataset)
    return predictions


def show_products_of_one_cluster(num_cluster, predictions, with_sort=True):
    """Вывод товаров из определенного кластера"""
    print(f"\nТовары кластера №{num_cluster}:")
    predictions_filtered = (
        predictions.where(condition=F.col("prediction") == num_cluster)
        .select("product_id")
        .join(other=products, on="product_id", how="inner")
    )

    if with_sort:
        predictions_filtered = predictions_filtered.orderBy("name", ascending=True)

    return predictions_filtered


# Первый уровень кластеризации
print("\nПервый уровень кластеризации...")
predictions_level_1 = kmeans_model_fit(k=21, dataset=product_vectors)
print("\nПример распределения товаров по кластерам:")
predictions_level_1.show()

# Пример анализа одного кластера (например, №3)
print("\nПодробный анализ кластера №3:")
show_products_of_one_cluster(
    num_cluster=3, predictions=predictions_level_1, with_sort=False
).show(n=30, truncate=False)

# Второй уровень кластеризации для выбранного кластера
print("\nВторой уровень кластеризации для кластера №3...")
product_vectors_lvl_2 = (
    product_vectors.join(
        other=predictions_level_1.select("product_id", "prediction"),
        on="product_id",
        how="inner",
    )
    .where(F.col("prediction") == 3)
    .select("product_id", "vector")
)

predictions_level_2 = kmeans_model_fit(k=8, dataset=product_vectors_lvl_2)

# Анализ подкластеров
print("\nАнализ подкластеров:")
for i in range(8):
    show_products_of_one_cluster(
        num_cluster=i, predictions=predictions_level_2, with_sort=True
    ).show(n=15, truncate=False)
