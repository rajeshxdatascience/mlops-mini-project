# model test v1 - loading the model

import unittest
import mlflow
import os


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get DagsHub token from environment variable
        dagshub_token = os.getenv("DAGSHUB_PAT")

        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        # DagsHub credentials for MLflow
        os.environ["MLFLOW_TRACKING_USERNAME"] = "rajeshxdatascience"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # MLflow tracking URI
        repo_owner = "rajeshxdatascience"
        repo_name = "mlops-mini-project"

        mlflow.set_tracking_uri(
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
        
        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)
        

if __name__ == "__main__":
    unittest.main()