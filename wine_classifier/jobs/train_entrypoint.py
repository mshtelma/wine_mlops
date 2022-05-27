import mlflow
from wine_classifier.model import train
from wine_classifier.jobs.common import Job

class TrainJob(Job):

    def launch(self):
        self.logger.info("Launching sample job")

        mlflow.set_experiment(self.conf["experiment"])
        train(self.conf, self.spark)

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = TrainJob()
    job.launch()
