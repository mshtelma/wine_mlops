import logging
import os
from typing import Dict, Any

import mlflow
import xgboost as xgb
from mlflow.models import infer_signature
from pyspark.sql import SparkSession

from .dataset import  DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-2s %(filename)s:%(lineno)s: %(message)s",
)
logger = logging.getLogger(__name__)


# class WineClassifier(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         # load the model from artifacts in model context
#
#         import xgboost as xgb
#
#         self.classifier = xgb.XGBClassifier()
#         self.classifier.load_model(
#             os.path.join(context.artifacts["model_dir"], "model", "model.json")
#         )
#
#         self.labels = LABELS
#
#     def predict(self, context, model_input):
#         import numpy as np
#
#         predictions = self.classifier.predict(model_input).astype(np.int64)
#
#         return np.array(list(map(lambda p: self.labels[p], predictions)))



def train(conf: Dict[str, Any], spark:SparkSession):
    # Fetch the dataset
    conf["model_parameters"]["random_state"] = conf["random_state"]
    train_x, train_y, test_x, test_y = DataLoader(spark, conf["random_state"]).prepare_data()

    with mlflow.start_run() as model_run:
        mlflow.log_params(
            {
                "train_examples": len(train_x.index),
                "test_examples": len(test_x.index),
            }
        )

        # Log all XGBoost Parameters to MLFlow
        mlflow.xgboost.autolog()

        # Train the XGBoost Classifier
        classifier = xgb.XGBClassifier(**conf["model_parameters"])
        classifier.fit(X=train_x, y=train_y)

        mlflow.sklearn.eval_and_log_metrics(classifier, X=test_x, y_true=test_y, prefix="test_")

        # Capture Conda Dependencies for MLFlow Model setup
        # conda_env = {
        #     "channels": ["defaults", "conda-forge", "anaconda"],
        #     "dependencies": [
        #         f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
        #         f"py-xgboost={xgb.__version__}",
        #         "pip",
        #         {
        #             "pip": [
        #                 "mlflow",
        #                 f"numpy=={np.__version__}",
        #             ],
        #         },
        #     ],
        #     "name": "mlflow-env",
        # }

        mlflow.xgboost.log_model(classifier,
                                 "model",
                                 signature=infer_signature(train_x, train_y),
                                 registered_model_name=conf["model_name"])

        logger.info(
            f"Model URI: {os.path.join(model_run.info.artifact_uri, 'model_dir')}"
        )
