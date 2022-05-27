import pandas as pd
from wine_classifier.jobs.common import Job
DATA_URI = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-{label}.csv"
class DataPrepJob(Job):

    def launch(self):
        self.logger.info("Launching sample job")

        sdf = self.spark.createDataFrame(DataPrepJob.load_wine_dataset())
        train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2])
        train_sdf.write.format("delta").mode("overwrite").saveAsTable("wine_train")
        test_sdf.drop("is_red").write.format("delta").mode("overwrite").saveAsTable("wine_scoring")

        self.logger.info("Sample job finished!")

    @staticmethod
    def load_wine_dataset()->pd.DataFrame:
        red_wine_df = pd.read_csv(DATA_URI.format(label="red"), sep=";")
        white_wine_df = pd.read_csv(DATA_URI.format(label="white"), sep=";")
        red_wine_df = red_wine_df.assign(is_red=1)  # Red Wine as '1'
        white_wine_df = white_wine_df.assign(is_red=0)  # White Wine as '0'
        df = pd.concat([red_wine_df, white_wine_df], sort=False)
        df = df.rename(columns={col: col.replace(" ", "_") for col in df.columns})
        return df


if __name__ == "__main__":
    job = DataPrepJob()
    job.launch()
