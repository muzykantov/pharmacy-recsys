from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка моделей и данных
print("Загрузка моделей и данных...")
w2v_model = Word2VecModel.load("models/word2vec_model_2024_12_15")
kmeans_best_params = KMeans.load("models/kmeans_2024_12_15")
kmeans_model = KMeansModel.load("models/kmeans_model_2024_12_15")

product_vectors = w2v_model.getVectors().withColumnRenamed(
    existing="word", new="product_id"
)
products = spark.read.csv(
    "data/products.csv", header=True, inferSchema=True
).withColumn("name", F.regexp_replace("name", r"(\(\d+\) )", ""))

# Получение предсказаний
print("Формирование предсказаний...")
predictions = kmeans_model.transform(product_vectors)


def show_products_of_one_cluster(num_cluster, n_rows, with_sort=True):
    """Вывод товаров из определенного кластера"""
    print(f"\nТовары кластера №{num_cluster}:")
    predictions_filtered = (
        predictions.where(condition=F.col("prediction") == num_cluster)
        .select("product_id")
        .join(other=products, on="product_id", how="left")
    )

    if with_sort:
        predictions_filtered = predictions_filtered.orderBy("name", ascending=True)

    return predictions_filtered.show(n=n_rows, truncate=False)


# Пример использования: вывод товаров из кластера №10
print("\nПример кластеризации товаров:")
show_products_of_one_cluster(num_cluster=10, n_rows=10, with_sort=True)

# Вывод примера содержимого всех кластеров
print("\nПросмотр всех кластеров:")
for i in range(21):
    show_products_of_one_cluster(num_cluster=i, n_rows=5, with_sort=False)
