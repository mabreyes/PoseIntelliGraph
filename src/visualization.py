#!/usr/bin/env python3
"""
Visualization utilities for the violence detection model.

This module provides functions to plot and visualize various aspects of the
violence detection model, including training metrics, classification performance,
and model-specific visualizations.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def plot_training_metrics(
    metrics: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot and save training and validation metrics.

    Args:
        metrics: Dictionary containing training metrics history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Plot loss curves
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)

    # Plot AUC curves
    plt.subplot(2, 2, 2)
    plt.plot(metrics["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.title("Validation ROC AUC Score")  # More specific title
    plt.grid(True, alpha=0.3)

    # If available, plot additional metrics like F1 score
    if "val_f1" in metrics:
        plt.subplot(2, 2, 3)
        plt.plot(metrics["val_f1"], label="Validation F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.title("Validation F1 Score")
        plt.grid(True, alpha=0.3)

    # Plot accuracy if available
    if "val_accuracy" in metrics:
        plt.subplot(2, 2, 4)
        plt.plot(metrics["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation Accuracy")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()  # Adjust layout to prevent overlapping titles/labels

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def plot_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot comprehensive classification metrics including ROC, PR curves
    and confusion matrix.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)
        threshold: Classification threshold
        output_path: Path to save the plot
    """
    plt.figure(figsize=(18, 10))

    # ROC Curve
    plt.subplot(2, 3, 1)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")

    # Mark threshold point
    if thresholds.size > 0 : # Ensure thresholds is not empty
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        plt.plot(
            fpr[threshold_idx],
            tpr[threshold_idx],
            "ro",
            label=f"Threshold = {threshold:.4f}",
        )

    plt.plot([0, 1], [0, 1], "k--", label="No Skill") # Added label for clarity
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    plt.subplot(2, 3, 2)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    
    # Ensure pr_thresholds is not empty before finding closest threshold
    # pr_thresholds does not include an explicit threshold for recall=0, precision=undefined (or P=1, R=0 for some conventions)
    # The last precision and recall values are 1.0 and 0.0 respectively, pr_thresholds is shorter by 1.
    
    plt.plot(recall, precision, label="Precision-Recall Curve")
    
    # Find point on PR curve closest to the chosen threshold for marking
    # This can be tricky as pr_thresholds may not align perfectly with `threshold`
    # A common approach is to find the point where PR curve is closest to (1,1) or other strategy.
    # For now, let's mark the point based on the F1 score at the given threshold.
    y_pred_at_threshold = (y_score >= threshold).astype(int)
    p_at_thresh = precision_score(y_true, y_pred_at_threshold, zero_division=0)
    r_at_thresh = recall_score(y_true, y_pred_at_threshold, zero_division=0)
    if p_at_thresh > 0 or r_at_thresh > 0: # Only plot if the point is meaningful
        plt.plot(r_at_thresh, p_at_thresh, "ro", label=f"Point at Threshold {threshold:.4f}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left") # Adjusted for better visibility
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 3, 3)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Explicitly use labels for consistency
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Non-violent", "Violent"], 
                yticklabels=["Non-violent", "Violent"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    # plt.xticks([0.5, 1.5], ["Non-violent", "Violent"]) # Handled by heatmap xticklabels
    # plt.yticks([0.5, 1.5], ["Non-violent", "Violent"]) # Handled by heatmap yticklabels

    # Threshold vs. F1 score
    plt.subplot(2, 3, 4)
    f1_scores = []
    # np.linspace can sometimes miss exact points like 0 or 1 if num is not chosen carefully.
    # Using a slightly adjusted range or more points for smoother curve if needed.
    threshold_range = np.linspace(0.01, 0.99, 100) 
    for t_scan in threshold_range:
        y_pred_scan = (y_score >= t_scan).astype(int)
        f1 = f1_score(y_true, y_pred_scan, zero_division=0)
        f1_scores.append(f1)

    plt.plot(threshold_range, f1_scores, label="F1 Score per Threshold")
    plt.axvline(
        x=threshold,
        color="r",
        linestyle="--",
        label=f"Selected Threshold: {threshold:.4f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs. F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics at threshold (Text display or Bar Chart)
    plt.subplot(2, 3, 5)
    # y_pred already calculated
    accuracy = accuracy_score(y_true, y_pred)
    # p_at_thresh, r_at_thresh already calculated
    f1_val = f1_score(y_true, y_pred, zero_division=0) # F1 at the chosen threshold

    metrics_vals = [accuracy, p_at_thresh, r_at_thresh, f1_val]
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

    y_pos = np.arange(len(metrics_names))
    bars = plt.barh(y_pos, metrics_vals, align="center", color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.yticks(y_pos, metrics_names)
    plt.xlim([0, 1.05]) # Adjusted limit for text
    plt.xlabel("Score")
    plt.title(f"Metrics at Threshold = {threshold:.4f}")
    plt.grid(True, alpha=0.3, axis="x")

    # Add values to bars
    for bar in bars:
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{bar.get_width():.4f}", va='center')


    # Distribution of scores
    plt.subplot(2, 3, 6)
    # Filter out NaN or inf scores if any, though y_score should be probabilities
    y_score_finite = y_score[np.isfinite(y_score)]
    positive_scores = y_score_finite[y_true[np.isfinite(y_score)] == 1]
    negative_scores = y_score_finite[y_true[np.isfinite(y_score)] == 0]


    plt.hist(negative_scores, bins=20, alpha=0.7, label="Non-violent (True)", color="green")
    plt.hist(positive_scores, bins=20, alpha=0.7, label="Violent (True)", color="red")
    plt.axvline(
        x=threshold, color="blue", linestyle="--", label=f"Selected Threshold: {threshold:.4f}"
    )
    plt.xlabel("Predicted Score")
    plt.ylabel("Frequency")
    plt.title("Score Distributions by True Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout() # Adjust layout

    if output_path:
        plt.savefig(output_path)
        print(f"Classification metrics plot saved to {output_path}")
    else:
        plt.show()


def plot_learning_curve(
    metrics: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot learning curve to assess model training process.

    Args:
        metrics: Dictionary containing training metrics history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Training Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot AUC score
    plt.subplot(1, 2, 2)
    plt.plot(metrics["val_auc"], label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC Score")
    plt.title("Learning Curve - AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Learning curve plot saved to {output_path}")
    else:
        plt.show()


def plot_pose_graph(
    keypoints: np.ndarray,
    graph_edges: List[Tuple[int, int]],
    is_violent: bool,
    output_path: Optional[Path] = None,
) -> None:
    """
    Visualize a pose graph with its node connections.

    Args:
        keypoints: NumPy array of shape [num_keypoints, 2] containing (x, y) coordinates
        graph_edges: List of tuples representing edges between keypoints
        is_violent: Whether the pose is from a violent sample
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))

    # Plot keypoints
    plt.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c="red" if is_violent else "blue",
        s=100,
        marker="o",
    )

    # Plot edges
    for edge in graph_edges:
        plt.plot(
            [keypoints[edge[0], 0], keypoints[edge[1], 0]],
            [keypoints[edge[0], 1], keypoints[edge[1], 1]],
            "k-",
            alpha=0.6,
        )

    # Add labels for keypoints
    for i, (x, y) in enumerate(keypoints):
        plt.text(x, y, str(i), fontsize=10, ha="center", va="center")

    plt.title(f"Pose Graph ({'Violent' if is_violent else 'Non-violent'})")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path)
        print(f"Pose graph visualization saved to {output_path}")
    else:
        plt.show()
