import os
import sys
from urllib.parse import urlparse

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import CybersecurityModel
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

import dagshub
dagshub.init(repo_owner='sumandutta2913', repo_name='Project-ML-AWS', mlflow=True)
# os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/krishnaik06/networksecurity.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "krishnaik06"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "7104284f1bb44ece21e0e2adb4e36a250ae3251f"

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        # mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall", classification_metric.recall_score)

            mlflow.sklearn.log_model(best_model, "model")
            # if tracking_url_type_store != "file":
            #     mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model.__class__.__name__)

    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            #"Decision Tree": DecisionTreeClassifier(),
            #"Gradient Boosting": GradientBoostingClassifier(verbose=1),
            # "Logistic Regression": LogisticRegression(verbose=1),
            #"AdaBoost": AdaBoostClassifier(),
            "SVM": SVC(probability=True)
        }

        params = {
            # "Decision Tree": {
            #     'criterion': ['gini', 'entropy', 'log_loss'],
            #     'splitter': ['best', 'random'],
            #     'max_features': ['sqrt', 'log2']
            # },
            "Random Forest": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features': ['sqrt', 'log2', None],
                'n_estimators': [8, 16, 32, 128, 256]
            },
            # "Gradient Boosting": {
            #     'loss': ['log_loss', 'exponential'],
            #     'learning_rate': [.1, .01, .05, .001],
            #     'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
            #     'criterion': ['squared_error', 'friedman_mse'],
            #     'max_features': ['auto', 'sqrt', 'log2'],
            #     'n_estimators': [8, 16, 32, 64, 128, 256]
            # },
            # #"Logistic Regression": {},
            # "AdaBoost": {
            #     'learning_rate': [.1, .01, .001],
            #     'n_estimators': [8, 16, 32, 64, 128, 256]
            # },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }

        model_report = evaluate_models(X_train, y_train, X_test, y_test, models=models, param=params)
        best_model_score = max(model_report.values())
        best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
        best_model = models[best_model_name]

        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, classification_test_metric)

        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        cybersecurity_model = CybersecurityModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=cybersecurity_model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
