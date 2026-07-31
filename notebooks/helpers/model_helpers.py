from plots.classifiers import plot_calibration_curve, plot_confusion_matrix, plot_precision_recall_curve, plot_prediction_probabilities, plot_roc_curve
import os
from typing import Any, Dict, Optional, Tuple, Union

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from mlflow.models import infer_signature

def get_or_create_experiment(experiment_name: str) -> str:
    """Retrieve or create an MLflow experiment by name."""
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        if experiment.lifecycle_stage == "deleted":
            client.restore_experiment(experiment.experiment_id)
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def logistic_regression_model_train_log(
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name,
    model_name,
    run_name = 'Log Reg Optuna Optimized',
    n_trials: int = 25,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> Tuple[LogisticRegression, optuna.Study, Dict[str, Any]]:
    """
    Performs hyperparameter optimization with cross-validation using Optuna for LogisticRegression,
    trains the final model with the best parameters, and logs experiments & models to MLflow.

    Parameters:
    -----------
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like or Series
        Training targets.
    experiment_name : str
        MLflow experiment name.
    n_trials : int
        Number of Optuna optimization trials.
    cv : int
        Number of folds for cross-validation.
    scoring : str
        Scoring metric for cross-validation (e.g. 'f1', 'accuracy', 'roc_auc').
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[LogisticRegression, optuna.Study, Dict[str, Any]]
        - Trained LogisticRegression model with best parameters.
        - Optuna Study object containing search history.
        - Dictionary of best parameters.
    """
    # Convert y_train to 1D array to avoid DataConversionWarning
    if hasattr(y_train, "values"):
        y_train = y_train.values.ravel()
    else:
        y_train = np.ravel(y_train)

    # Auto-adjust binary scoring metrics for multiclass targets
    unique_classes = np.unique(y_train)
    is_multiclass = len(unique_classes) > 2
    if is_multiclass:
        multiclass_metric_map = {
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "roc_auc": "roc_auc_ovr_weighted",
        }
        scoring = multiclass_metric_map.get(scoring, scoring)

    # Pre-determine valid (penalty, solver) combinations to avoid pruning trials
    if is_multiclass:
        valid_combos = [
            ("l1", "saga"),
            ("l2", "lbfgs"),
            ("l2", "saga"),
            ("none", "lbfgs"),
            ("none", "saga"),
        ]
    else:
        valid_combos = [
            ("l1", "saga"),
            ("l1", "liblinear"),
            ("l2", "lbfgs"),
            ("l2", "saga"),
            ("l2", "liblinear"),
            ("none", "lbfgs"),
            ("none", "saga"),
        ]

    combo_choices = [f"{p}:{s}" for p, s in valid_combos]

    experiment_id = get_or_create_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space for Logistic Regression
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        combo = trial.suggest_categorical("penalty_solver", combo_choices)
        max_iter = trial.suggest_int("max_iter", 200, 1000, step=100)

        penalty_str, solver = combo.split(":", 1)
        actual_penalty = None if penalty_str == "none" else penalty_str

        # Build candidate classifier
        model = LogisticRegression(
            C=C if actual_penalty is not None else 1.0,
            penalty=actual_penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )

        # Nested MLflow run per Optuna trial
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"Trial_{trial.number}",
            nested=True,
        ):
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )
            mean_cv_score = float(np.mean(cv_scores))

            # Log hyperparameters & metrics for this trial
            mlflow.log_params(
                {
                    "C": C if actual_penalty is not None else 1.0,
                    "penalty": str(actual_penalty),
                    "solver": solver,
                    "max_iter": max_iter,
                }
            )
            mlflow.log_metric(f"mean_cv_{scoring}", mean_cv_score)
            mlflow.log_metric(f"std_cv_{scoring}", float(np.std(cv_scores)))

        return mean_cv_score

    # Quiet Optuna verbose logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    # Parent MLflow Run for the full tuning session
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name
    ) as parent_run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "n_trials": n_trials,
                "cv_folds": cv,
                "scoring_metric": scoring,
            }
        )

        # Execute Optuna optimization
        study.optimize(objective, n_trials=n_trials)

        # Extract and format best parameters cleanly
        raw_best_params = study.best_params.copy()
        combo = raw_best_params.pop("penalty_solver")
        penalty_str, solver = combo.split(":", 1)

        best_params = raw_best_params
        best_params["penalty"] = None if penalty_str == "none" else penalty_str
        best_params["solver"] = solver
        if best_params["penalty"] is None:
            best_params["C"] = 1.0

        best_cv_score = study.best_value

        # Log best trial details to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_cv_{scoring}", best_cv_score)

        # Train final model on entire training set with best parameters
        best_model = LogisticRegression(
            random_state=random_state,
            **best_params
        )
        best_model.fit(X_train, y_train)

        # Log trained model to MLflow
        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            serialization_format="cloudpickle",
        )

        # Log dataset to MLflow
        if isinstance(X_train, pd.DataFrame):
            dataset = mlflow.data.from_pandas(
                df=X_train,
                targets=y_train if isinstance(y_train, (pd.Series, pd.DataFrame)) else None,
                name="training_dataset",
            )
        else:
            dataset = mlflow.data.from_numpy(
                features=np.asarray(X_train),
                targets=np.asarray(y_train),
                name="training_dataset",
            )
        mlflow.log_input(dataset, context="training")

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)

        test_f1_average = "weighted" if is_multiclass else "binary"
        test_f1_score = f1_score(y_test, y_pred, average=test_f1_average)
        mlflow.log_metric("test_f1", float(test_f1_score))

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_dict(report_dict, "classification_report.json")
        
        # Log evaluation plots from plots/classifiers.py to MLflow
        fig1, _ = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(fig1, "plots/confusion_matrix.png")
        plt.close(fig1)
        
        fig2, _ = plot_roc_curve(y_test, y_prob)
        mlflow.log_figure(fig2, "plots/roc_curve.png")
        plt.close(fig2)
        
        fig3, _ = plot_precision_recall_curve(y_test, y_prob)
        mlflow.log_figure(fig3, "plots/precision_recall_curve.png")
        plt.close(fig3)
        
        fig4, _ = plot_prediction_probabilities(y_test, y_prob)
        mlflow.log_figure(fig4, "plots/prediction_probabilities.png")
        plt.close(fig4)
        
        fig5, _ = plot_calibration_curve(y_test, y_prob)
        mlflow.log_figure(fig5, "plots/calibration_curve.png")
        plt.close(fig5)

        print("--- Optimization Finished ---")
        print(f"Best CV {scoring.upper()} Score: {best_cv_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print(f"Test F1 ({test_f1_average}) Score: {test_f1_score:.4f}")

    return best_model, study, best_params


def svc_model_training(
    X_train,
    y_train,
    experiment_name: str = "SVC_Optuna",
    n_trials: int = 25,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> Tuple[SVC, optuna.Study, Dict[str, Any]]:
    """
    Performs hyperparameter optimization with cross-validation using Optuna for Support Vector Classifier (SVC),
    trains the final model with the best parameters, and logs experiments & models to MLflow.

    Parameters:
    -----------
    X_train : array-like or DataFrame
        Training features.
    y_train : array-like or Series
        Training targets.
    experiment_name : str
        MLflow experiment name.
    n_trials : int
        Number of Optuna optimization trials.
    cv : int
        Number of folds for cross-validation.
    scoring : str
        Scoring metric for cross-validation (e.g. 'f1', 'accuracy', 'roc_auc').
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    Tuple[SVC, optuna.Study, Dict[str, Any]]
        - Trained SVC model with best parameters.
        - Optuna Study object containing search history.
        - Dictionary of best parameters.
    """
    # Convert y_train to 1D array to avoid DataConversionWarning
    if hasattr(y_train, "values"):
        y_train = y_train.values.ravel()
    else:
        y_train = np.ravel(y_train)

    # Auto-adjust binary scoring metrics for multiclass targets
    unique_classes = np.unique(y_train)
    is_multiclass = len(unique_classes) > 2
    if is_multiclass:
        multiclass_metric_map = {
            "f1": "f1_weighted",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "roc_auc": "roc_auc_ovr_weighted",
        }
        scoring = multiclass_metric_map.get(scoring, scoring)

    experiment_id = get_or_create_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        degree = trial.suggest_int("degree", 2, 5)

        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            probability=True,
            random_state=random_state,
        )

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"Trial_{trial.number}",
            nested=True,
        ):
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )
            mean_cv_score = float(np.mean(cv_scores))

            mlflow.log_params(
                {
                    "C": C,
                    "kernel": kernel,
                    "gamma": gamma,
                    "degree": degree if kernel == "poly" else "N/A",
                }
            )
            mlflow.log_metric(f"mean_cv_{scoring}", mean_cv_score)
            mlflow.log_metric(f"std_cv_{scoring}", float(np.std(cv_scores)))

        return mean_cv_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")

    with mlflow.start_run(
        experiment_id=experiment_id, run_name="Optuna_Parent_Run"
    ) as parent_run:
        mlflow.log_params(
            {
                "model_type": "SVC",
                "n_trials": n_trials,
                "cv_folds": cv,
                "scoring_metric": scoring,
            }
        )

        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params.copy()
        best_cv_score = study.best_value

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric(f"best_cv_{scoring}", best_cv_score)

        best_model = SVC(
            random_state=random_state,
            probability=True,
            **best_params,
        )
        best_model.fit(X_train, y_train)

        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="SVC_Best_Model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        # Log dataset to MLflow
        if isinstance(X_train, pd.DataFrame):
            dataset = mlflow.data.from_pandas(
                df=X_train,
                targets=y_train if isinstance(y_train, (pd.Series, pd.DataFrame)) else None,
                name="training_dataset",
            )
        else:
            dataset = mlflow.data.from_numpy(
                features=np.asarray(X_train),
                targets=np.asarray(y_train),
                name="training_dataset",
            )
        mlflow.log_input(dataset, context="training")

        print("--- SVC Optimization Finished ---")
        print(f"Best CV {scoring.upper()} Score: {best_cv_score:.4f}")
        print(f"Best Parameters: {best_params}")

    return best_model, study, best_params