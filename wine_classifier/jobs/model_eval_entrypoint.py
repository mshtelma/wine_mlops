import mlflow
from wine_classifier.model import model_eval
from wine_classifier.jobs.common import Job


class ModelEvalJob(Job):
    def launch(self):
        self.logger.info("Launching sample job")

        mlflow.set_experiment(self.conf["experiment"])
        model_eval(self.conf, self.spark)

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = ModelEvalJob()
    job.launch()
