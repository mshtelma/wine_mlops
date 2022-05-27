import pathlib
import unittest
import tempfile
import os
import shutil

import mlflow
import yaml
from pyspark.sql import SparkSession

from wine_classifier.jobs.data_setup_entrypoint import DataPrepJob
from wine_classifier.jobs.scoring_entrypoint import ScoringJob
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
        self.train_conf = yaml.safe_load(pathlib.Path("conf/train.yml").read_text())
        self.scoring_conf = yaml.safe_load(pathlib.Path("conf/scoring.yml").read_text())
        df = DataPrepJob.load_wine_dataset()
        self.spark.createDataFrame(df).createTempView("wine_train")
        self.spark.createDataFrame(df).drop("is_red").createTempView("wine_scoring")
        self.train_conf['experiment'] = "test_exp"
        self.scoring_conf['experiment'] = "test_exp"
        self.scoring_conf['stage']='None'
        self.train_job = TrainJob(spark=self.spark, init_conf=self.train_conf)
        self.scoring_job = ScoringJob(spark=self.spark, init_conf=self.scoring_conf)

    def test_sample(self):
        self.train_job.launch()
        self.scoring_job.launch()




if __name__ == "__main__":
    unittest.main()
