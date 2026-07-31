import mlflow
from mlflow.tracking import MlflowClient


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it exists in deleted state, it restores it and returns its ID.
    If it exists in active state, the function returns its ID.
    If not, it creates a new experiment with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    client = MlflowClient()
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

