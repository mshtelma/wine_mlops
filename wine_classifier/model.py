import logging
import os
from typing import Dict, Any, Optional

import mlflow
import pandas as pd
import xgboost as xgb
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession

from .dataset import DataLoader

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


def train(conf: Dict[str, Any], spark: SparkSession):
    # Fetch the dataset
    conf["model_parameters"]["random_state"] = conf["random_state"]
    train_x, train_y, test_x, test_y = DataLoader(
        spark, conf["random_state"], conf["dataset"]
    ).prepare_data()

    with mlflow.start_run() as model_run:
        mlflow.set_tag("action", "train")
        mlflow.log_params(
            {
                "train_examples": len(train_x.index),
                "test_examples": len(test_x.index),
            }
        )

        # Log all XGBoost Parameters to MLFlow
        mlflow.sklearn.autolog()

        # Train the XGBoost Classifier
        classifier = xgb.XGBClassifier(**conf["model_parameters"])
        classifier.fit(X=train_x, y=train_y)

        mlflow.sklearn.eval_and_log_metrics(
            classifier, X=test_x, y_true=test_y, prefix="test_"
        )

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

        mlflow.sklearn.log_model(
            classifier,
            "model",
            signature=infer_signature(train_x, train_y),
            registered_model_name=conf["model_name"],
        )

        logger.info(
            f"Model URI: {os.path.join(model_run.info.artifact_uri, 'model_dir')}"
        )


def model_eval(conf: Dict[str, Any], spark: SparkSession) -> None:
    from_version = get_latest_model_for_stage(conf["model_name"], conf["from_stage"])
    to_version = get_latest_model_for_stage(conf["model_name"], conf["to_stage"])
    if from_version is None:
        raise Exception(f"No model in {conf['from_stage']} found! Exiting...")
    # if to_version is None:
    #    deploy_model(conf, from_version)
    #    return

    # both of versions are available
    _deploy = False
    if conf["compare_models"]:
        with mlflow.start_run() as model_run:
            mlflow.set_tag("action", "eval")
            df = DataLoader(
                spark, conf["random_state"], conf["dataset"]
            ).load_features()
            from_metrics = calc_metrics(from_version, df)
            if to_version is not None:
                to_metrics = calc_metrics(to_version, df)
                if from_metrics["roc_auc"] >= to_metrics["roc_auc"]:
                    _deploy = True
            else:
                _deploy = True
    else:
        _deploy = True
    if _deploy:
        deploy_model(conf, from_version)


def calc_metrics(model_version: ModelVersion, df: pd.DataFrame):
    res = mlflow.evaluate(
        f"models:/{model_version.name}/{model_version.version}",
        df,
        targets="is_red",
        model_type="classifier",
    )
    return res.metrics


def deploy_model(conf: Dict[str, Any], model_version: ModelVersion) -> None:
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_version.name,
        version=model_version.version,
        stage=conf["to_stage"],
        archive_existing_versions=True,
    )


def get_latest_model_for_stage(model_name: str, stage: str) -> Optional[ModelVersion]:
    mlflow_client = MlflowClient()
    versions = mlflow_client.get_latest_versions(model_name, stages=[stage])
    if len(versions) > 0:
        return versions[0]
    else:
        return None
