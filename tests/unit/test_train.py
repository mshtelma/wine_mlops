import pathlib
import unittest
import tempfile
import os
import shutil

import mlflow
import yaml
from pyspark.sql import SparkSession
from unittest.mock import MagicMock

from jobs.data_setup_entrypoint import DataPrepJob
from wine_classifier.jobs.train_entrypoint import TrainJob


class SampleJobUnitTest(unittest.TestCase):
    def setUp(self):
        try:
            shutil.rmtree("spark-warehouse", ignore_errors=True)
            shutil.rmtree("mlruns", ignore_errors=True)
            os.remove("mlruns.db")
        except Exception as e:
            print(e)
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        self.test_dir = tempfile.TemporaryDirectory().name
        self.spark = SparkSession.builder.master("local[1]").getOrCreate()
        self.test_config = yaml.safe_load(pathlib.Path("conf/train.yml").read_text())
        df = DataPrepJob.load_wine_dataset()
        self.spark.createDataFrame(df).createTempView("wine_train")
        self.test_config['experiment'] = "test_exp"
        self.job = TrainJob(spark=self.spark, init_conf=self.test_config)

    def test_sample(self):
        # feel free to add new methods to this magic mock to mock some particular functionality
        self.job.dbutils = MagicMock()

        self.job.launch()




if __name__ == "__main__":
    unittest.main()
