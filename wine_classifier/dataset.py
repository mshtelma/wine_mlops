import pandas as pd
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, spark:SparkSession, random_state:float):
        self.spark = spark
        self.random_state = random_state


    def prepare_train_features(self) -> pd.DataFrame:
        return self.spark.read.table("wine_train").toPandas()  # noqa

    def prepare_scoring_features(self) -> pd.DataFrame:
        return self.spark.read.table("wine_scoring").toPandas()  # noqa

    def prepare_data(self):
        data = self.prepare_train_features()

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data, random_state=self.random_state)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["is_red"], axis=1)
        test_x = test.drop(["is_red"], axis=1)
        train_y = train[["is_red"]]
        test_y = test[["is_red"]]

        return (train_x, train_y, test_x, test_y)
