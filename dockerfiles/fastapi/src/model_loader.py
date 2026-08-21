import os
import tempfile
import time

import boto3
import cloudpickle
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "student_performance_model"
MODEL_ALIAS = "champion"
RELOAD_CHECK_INTERVAL_SECONDS = int(os.getenv("RELOAD_CHECK_INTERVAL_SECONDS", "300"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def _download_from_s3(s3_uri: str) -> bytes:
    bucket, key = s3_uri.replace("s3://", "", 1).split("/", 1)
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)  # cerramos el fd: en Windows boto3 no puede escribir en un archivo que ya tenemos abierto
    try:
        s3.download_file(bucket, key, tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.remove(tmp_path)

class ModelWrapper:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.model = None
        self.version = None
        self.run_id = None
        self.f1_weighted = None
        self.preprocessor_s3_path = None
        self._last_check = 0.0
        self._try_load()

    def _try_load(self):
        "Intenta cargar el champion. Si no existe todavia, no revienta: solo loguea y sigue."
        try:
            self._load()
        except Exception as e:
            print(f"No se pudo cargar el champion todavia: {e}")
            self.model = None
        self._last_check = time.time()

    def _load(self):
        version_info = self.client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.sklearn.load_model(model_uri)

        run = self.client.get_run(version_info.run_id)
        preprocessor_s3_path = run.data.params["preprocessor_s3_path"]
        preprocessor = cloudpickle.loads(_download_from_s3(preprocessor_s3_path))

        # Solo pisamos el estado si todo salio bien
        self.model = model
        self.preprocessor = preprocessor
        self.preprocessor_s3_path = preprocessor_s3_path
        self.f1_weighted = run.data.metrics.get("f1_weighted")
        self.version = version_info.version
        self.run_id = version_info.run_id
        print(f"Loaded champion v{self.version} (run={self.run_id})")

    def _maybe_reload(self):
        "Si el TTL vencio, chequea si cambio el alias (o si nunca se pudo cargar, reintenta)."
        now = time.time()
        if now - self._last_check < RELOAD_CHECK_INTERVAL_SECONDS:
            return
        self._last_check = now
        if self.model is None:
            self._try_load()
            return
        try:
            version_info = self.client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            if version_info.version != self.version:
                print(f"New champion detected: v{self.version} -> v{version_info.version}")
                self._load()
        except Exception as e:
            print(f"No se pudo chequear el champion: {e}")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, df):
        self._maybe_reload()
        if not self.is_loaded:
            raise RuntimeError("No hay ningun modelo champion registrado todavia.")
        X_processed = self.preprocessor.transform(df)
        return self.model.predict(X_processed)


model_wrapper = ModelWrapper()