"""
Classification model evaluation and visualization utilities.
Supports both binary and multiclass targets.
"""

from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Union[str, int]]] = None,
    normalize: Optional[str] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a Confusion Matrix with count or normalized percentage annotations.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot(ax=ax, cmap=cmap, colorbar=True, values_format=".2f" if normalize else "d")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(False)
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: Optional[Union[int, str]] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the ROC curve and computes Area Under the Curve (AUC).
    Supports binary (1D or 2D) and multiclass (One-vs-Rest) probability scores.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    # Multiclass One-vs-Rest ROC Curves
    if y_score.ndim == 2 and y_score.shape[1] > 2:
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_true_binary = (y_true == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_score[:, i])
            auc_i = roc_auc_score(y_true_binary, y_score[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"Class '{cls}' (AUC = {auc_i:.3f})")

        try:
            macro_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
            label_text = f"Random (Macro AUC = {macro_auc:.3f})"
        except Exception:
            label_text = "Random Classifier"

        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label=label_text)

    else:
        # Binary Classification
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        auc_val = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Classifier")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return fig, ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: Optional[Union[int, str]] = None,
    title: str = "Precision-Recall Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the Precision-Recall curve and computes Average Precision (AP).
    Supports binary and multiclass (One-vs-Rest) probability scores.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)

    if y_score.ndim == 2 and y_score.shape[1] > 2:
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_true_binary = (y_true == cls).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score[:, i])
            ap_i = average_precision_score(y_true_binary, y_score[:, i])
            ax.plot(recall, precision, lw=1.5, label=f"Class '{cls}' (AP = {ap_i:.3f})")
    else:
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=pos_label)
        ap_val = average_precision_score(y_true, y_score, pos_label=pos_label)

        ax.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR Curve (AP = {ap_val:.4f})")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return fig, ax


def plot_prediction_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    bins: int = 25,
    title: str = "Prediction Probability Distribution by Class",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (7, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the histogram and KDE distribution of predicted probabilities separated by true class label.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        # Multiclass probability distribution of assigned class max probability
        y_max_prob = np.max(y_prob, axis=1)
        sns.histplot(x=y_max_prob, bins=bins, kde=True, color="#1f77b4", ax=ax, stat="density", alpha=0.5)
        ax.set_xlabel("Max Predicted Class Probability", fontsize=10)
    else:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        classes = np.unique(y_true)
        if class_names is None:
            class_names = [f"Class '{cls}'" for cls in classes]

        palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
        for idx, cls in enumerate(classes):
            c_name = class_names[idx] if idx < len(class_names) else f"Class {cls}"
            c_color = palette[idx % len(palette)]
            sns.histplot(
                x=y_prob[y_true == cls],
                bins=bins,
                kde=True,
                color=c_color,
                label=c_name,
                ax=ax,
                stat="density",
                alpha=0.4,
            )
        ax.set_xlabel("Predicted Probability for Positive Class", fontsize=10)

    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper center", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return fig, ax


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve (Reliability Diagram)",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots calibration curve (reliability diagram) to evaluate model probability calibration.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
        classes = np.unique(y_true)
        for i, cls in enumerate(classes):
            y_true_binary = (y_true == cls).astype(int)
            prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, i], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", lw=1.5, label=f"Class '{cls}'")
    else:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="o", lw=2, color="#9467bd", label="Model Calibration")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    return fig, ax
