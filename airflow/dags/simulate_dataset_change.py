import datetime

from airflow.decorators import dag, task

markdown_text = """
### Simulate Dataset Change (Testing utility)

Logs a fake `dataset_hash` param to MLflow so the next real run of
`process_etl_student_performance` thinks the dataset changed and
doesn't skip the pipeline.

Not part of the production pipeline — use only for testing the
change-detection logic.
"""

default_args = {
    "owner": "Grupo 10",
    "depends_on_past": False,
    "retries": 0,
}


@dag(
    dag_id="simulate_dataset_change",
    description="Testing utility: logs a fake dataset_hash to force the ETL to detect a change on its next run.",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["Testing"],
)
def simulate_dataset_change():

    @task.virtualenv(
        requirements=["mlflow==2.10.1"],
        system_site_packages=True,
    )
    def log_fake_hash() -> None:
        """
        Logs a dummy run with a dataset_hash that won't match any real
        dataset. The next real ETL run will compare against this and
        think the dataset changed.
        """
        import mlflow
        import uuid

        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.set_experiment("Student Performance")

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name="SIMULATED_change_for_testing",
        ):
            fake_hash = uuid.uuid4().hex
            mlflow.log_param("dataset_hash", fake_hash)
            print(f"Fake hash logged: {fake_hash}")

    log_fake_hash()


dag = simulate_dataset_change()