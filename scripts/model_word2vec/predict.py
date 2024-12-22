import time
from pprint import pprint

from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка данных о продуктах
products = spark.read.csv(
    "../../data/products.csv", header=True, inferSchema=True
).withColumn("name", F.regexp_replace("name", r"(\(\d+\) )", ""))


class ModelWord2Vec:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Загрузка модели из файла"""
        self.model = Word2VecModel.load(model_path)

    def predict_to_dict(self, product_id, n_recs=5):
        """Выдача предсказаний в виде словаря"""
        start = time.time()
        preds_dict = {}
        recs_df = (
            self.model.findSynonyms(word=str(product_id), num=n_recs)
            .withColumnRenamed(existing="word", new="product_id")
            .orderBy("similarity", ascending=False)
        )

        preds_dict["product_id"] = product_id
        preds_dict["recommendations"] = [
            int(row.product_id) for row in recs_df.collect()
        ]
        preds_dict["prediction time"] = round(number=time.time() - start, ndigits=3)
        return preds_dict

    def get_name_product_id(self, products_df, product_id):
        """Получение названия продукта по его ID"""
        name = (
            products_df.where(condition=F.col("product_id") == product_id)
            .select("name")
            .collect()[0]["name"]
        )
        return name

    def predict_to_df(self, products_df, product_id, num_recs=5):
        """Получение предсказаний с детальной информацией о продуктах"""
        return (
            self.model.findSynonyms(word=str(product_id), num=num_recs)
            .withColumnRenamed(existing="word", new="product_id")
            .join(other=products_df, on="product_id", how="inner")
            .orderBy("similarity", ascending=False)
            .withColumn("similarity", F.round("similarity", 6))
            .select("product_id", "name")
        )


def main():
    # Инициализация и загрузка модели
    model_w2v = ModelWord2Vec()
    model_w2v.load_model("../models/word2vec_model_2024_12_15")

    # Пример использования модели
    test_product_id = 33569

    # Получение названия тестового продукта
    print("\nНазвание продукта:")
    product_name = model_w2v.get_name_product_id(
        products_df=products, product_id=test_product_id
    )
    print(product_name)

    # Получение рекомендаций в виде словаря
    print("\nРекомендации в формате словаря:")
    predict_w2v = model_w2v.predict_to_dict(product_id=test_product_id, n_recs=3)
    pprint(predict_w2v)

    # Получение рекомендаций в виде таблицы с названиями
    print("\nРекомендации в виде таблицы:")
    predict_w2v_df = model_w2v.predict_to_df(
        products_df=products, product_id=test_product_id, num_recs=3
    )
    predict_w2v_df.show(truncate=False)


if __name__ == "__main__":
    main()
