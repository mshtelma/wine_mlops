import mlflow
from wine_classifier.model import train
from wine_classifier.jobs.common import Job


class ScoringJob(Job):
    def launch(self):
        self.logger.info("Launching sample job")

        mlflow.set_experiment(self.conf["experiment"])
        self.spark.udf.register(
            "model",
            mlflow.pyfunc.spark_udf(
                self.spark,
                model_uri=f"models:/{self.conf['model_name']}/{self.conf['stage']}",
            ),
        )
        sdf = self.spark.read.table("wine_scoring")
        columns = [c for c in sdf.columns]
        self.spark.sql(f"select model({','.join(columns)}) from wine_scoring").show(
            10, True
        )

        self.logger.info("Sample job finished!")


if __name__ == "__main__":
    job = ScoringJob()
    job.launch()
