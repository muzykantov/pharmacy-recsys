import time
from pprint import pprint

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка данных о продуктах
products = spark.read.csv(
    "data/products.csv", header=True, inferSchema=True
).withColumn("name", F.regexp_replace("name", r"(\(\d+\) )", ""))


class ModelALS:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Загрузка модели из файла"""
        self.model = ALSModel.load(model_path)

    def predict_to_dict(self, user_id, n_recs=5):
        """Получение рекомендаций для пользователя"""
        start = time.time()
        preds_dict = {}
        recs_df = (
            self.model.recommendForAllUsers(numItems=n_recs)
            .where(condition=F.col("user_id") == user_id)
            .withColumn(colName="rec_exp", col=F.explode("recommendations"))
            .select(F.col("rec_exp.item_id"))
        )

        preds_dict["user_id"] = user_id
        preds_dict["recommendations"] = [int(row.item_id) for row in recs_df.collect()]
        preds_dict["prediction time"] = round(number=time.time() - start, ndigits=3)
        return preds_dict

    def get_recommendations_with_names(self, user_id, products_df, n_recs=5):
        """Получение рекомендаций с названиями продуктов"""
        predictions = self.predict_to_dict(user_id, n_recs)
        products_info = products_df.filter(
            F.col("product_id").isin(predictions["recommendations"])
        )
        return products_info.select("product_id", "name").orderBy("product_id")


def main():
    # Инициализация и загрузка модели
    model_als = ModelALS()
    model_als.load_model("models/als_model_2024_12_15.model")

    # Пример использования модели
    test_user_id = 471

    print(f"\nРекомендации для пользователя {test_user_id}:")
    predictions = model_als.predict_to_dict(user_id=test_user_id, n_recs=3)
    pprint(predictions)

    print("\nРекомендации с названиями продуктов:")
    recommendations = model_als.get_recommendations_with_names(
        user_id=test_user_id, products_df=products, n_recs=3
    )
    recommendations.show(truncate=False)


if __name__ == "__main__":
    main()
