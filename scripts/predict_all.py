import time
from pprint import pprint

from pyspark.ml.feature import Word2VecModel
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Инициализация Spark сессии
spark = SparkSession.builder.appName("PharmacyRecsys").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Загрузка данных о продуктах из CSV
products = spark.read.csv(
    "../data/products.csv", header=True, inferSchema=True
).withColumn("name", F.regexp_replace("name", r"(\(\d+\) )", ""))


class ModelALS:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Загрузка модели ALS из файла"""
        self.model = ALSModel.load(model_path)

    def predict_to_dict(self, user_id, n_recs=5):
        """Генерация рекомендаций для пользователя"""
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


class ModelWord2Vec:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Загрузка модели Word2Vec из файла"""
        self.model = Word2VecModel.load(model_path)

    def predict_to_dict(self, product_id, n_recs=5):
        """Генерация рекомендаций на основе продукта"""
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
        """Получить название продукта по ID"""
        name = (
            products_df.where(condition=F.col("product_id") == product_id)
            .select("name")
            .collect()[0]["name"]
        )
        return name

    def predict_to_df(self, products_df, product_id, num_recs=5):
        """Генерация рекомендаций с деталями продуктов"""
        return (
            self.model.findSynonyms(word=str(product_id), num=num_recs)
            .withColumnRenamed(existing="word", new="product_id")
            .join(other=products_df, on="product_id", how="inner")
            .orderBy("similarity", ascending=False)
            .withColumn("similarity", F.round("similarity", 6))
            .select("product_id", "name")
        )


def main():
    # Инициализация моделей
    model_als = ModelALS()
    model_als.load_model("models/als_model_2024_12_15.model")

    model_w2v = ModelWord2Vec()
    model_w2v.load_model("models/word2vec_model_2024_12_15")

    # Примеры рекомендаций
    # Рекомендации ALS для пользователя с ID 471
    predict_als = model_als.predict_to_dict(user_id=471, n_recs=3)
    print("\nРекомендации ALS для пользователя с ID 471:")
    pprint(predict_als)

    # Рекомендации Word2Vec для продукта с ID 33569
    product_id = 33569
    print(f"\nНазвание продукта с ID {product_id}:")
    product_name = model_w2v.get_name_product_id(
        products_df=products, product_id=product_id
    )
    print(product_name)

    print("\nРекомендации Word2Vec:")
    predict_w2v = model_w2v.predict_to_dict(product_id=product_id, n_recs=3)
    pprint(predict_w2v)

    print("\nПодробные рекомендации Word2Vec:")
    predict_w2v_df = model_w2v.predict_to_df(
        products_df=products, product_id=product_id, num_recs=3
    )
    predict_w2v_df.show(truncate=False)


if __name__ == "__main__":
    main()
