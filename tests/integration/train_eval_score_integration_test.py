import unittest

from uuid import uuid4
from pyspark.dbutils import DBUtils  # noqa

from wine_classifier.jobs.data_setup_entrypoint import DataPrepJob
from wine_classifier.jobs.model_eval_entrypoint import ModelEvalJob
from wine_classifier.jobs.scoring_entrypoint import ScoringJob
from wine_classifier.jobs.train_entrypoint import TrainJob


class TrainEvalScoreIntegrationTest(unittest.TestCase):
    def setUp(self):

        self.dataprep_job = DataPrepJob()
        self.train_job = TrainJob()
        self.scoring_job = ScoringJob()
        self.model_eval_job = ModelEvalJob()
        self.dbutils = DBUtils(self.train_job.spark)
        self.spark = self.train_job.spark

    def test_sample(self):

        self.dataprep_job.launch()
        self.train_job.launch()
        self.model_eval_job.launch()
        self.scoring_job.launch()


if __name__ == "__main__":
    # please don't change the logic of test result checks here
    # it's intentionally done in this way to comply with jobs run result checks
    # for other tests, please simply replace the SampleJobIntegrationTest with your custom class name
    loader = unittest.TestLoader()
    tests = loader.loadTestsFromTestCase(TrainEvalScoreIntegrationTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    if not result.wasSuccessful():
        raise RuntimeError(
            "One or multiple tests failed. Please check job logs for additional information."
        )
