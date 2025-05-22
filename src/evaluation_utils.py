"""
Utilities for model evaluation and metrics calculation.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

def find_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate the optimal classification threshold using multiple methods.

    Implements several threshold optimization techniques:
    1. Youden's J statistic (maximizing sensitivity + specificity - 1)
    2. Minimum distance to perfect classifier (0,1) point in ROC space
    3. Maximum F1 score

    The primary method used is Youden's J statistic, which is widely accepted
    in the academic literature for binary classification threshold optimization.

    Args:
        y_true: Ground truth binary labels
        y_score: Predicted scores (probabilities)

    Returns:
        Tuple of (optimal threshold, dictionary of metrics at that threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate Youden's J statistic (J = Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    if not j_scores.size:  # Handle empty j_scores (e.g., all same class predictions)
        optimal_threshold_j = 0.5  # Default threshold
        optimal_idx_j = 0
        if thresholds.size:
            optimal_threshold_j = thresholds[0]
            # optimal_idx_j = 0 # Already set
        else: # thresholds is empty
            optimal_idx_j = -1 # Placeholder index, indicates no valid thresholds from roc_curve
        
        j_scores = np.array([0.0])  # Dummy value for j_scores
        
        # If optimal_idx_j indicates no valid threshold, ensure one exists for later use
        if optimal_idx_j == -1 or optimal_idx_j >= len(thresholds):
            thresholds = np.append(thresholds, optimal_threshold_j) # Add the default
            optimal_idx_j = len(thresholds) - 1 # Point to the newly added default

    else:
        optimal_idx_j = np.argmax(j_scores)
        optimal_threshold_j = thresholds[optimal_idx_j]


    # Calculate distance to (0,1) point in ROC space
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    if not distances.size:
        optimal_threshold_d = 0.5
    else:
        optimal_idx_d = np.argmin(distances)
        optimal_threshold_d = thresholds[optimal_idx_d]


    # Calculate F1 score at different thresholds
    # Ensure precision_recall_curve has enough points, and thresholds are valid
    if y_true.size > 0 and np.unique(y_true).size > 1: # Check for non-empty y_true and presence of both classes
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        
        # Calculate F1 for all valid (finite) thresholds from roc_curve
        f1_scores = []
        # Use roc_curve's thresholds as they are generally more comprehensive for this.
        valid_roc_thresholds = thresholds[np.isfinite(thresholds)]

        if not valid_roc_thresholds.size:
            optimal_threshold_f1 = 0.5
            # f1_scores_array = np.array([0.0]) # Not strictly needed if optimal_idx_f1 is 0
            optimal_idx_f1 = 0 # To avoid error with j_scores[optimal_idx_f1] later if used
        else:
            for t in valid_roc_thresholds:
                y_pred_f1 = (y_score >= t).astype(int)
                f1 = f1_score(y_true, y_pred_f1)
                f1_scores.append(f1)
            
            if not f1_scores: # Should not happen if valid_roc_thresholds is not empty
                optimal_threshold_f1 = 0.5
                optimal_idx_f1 = 0
            else:
                f1_scores_array = np.array(f1_scores)
                optimal_idx_f1 = np.argmax(f1_scores_array)
                optimal_threshold_f1 = valid_roc_thresholds[optimal_idx_f1]
    else:  # Handle single class in y_true or empty y_true
        optimal_threshold_f1 = 0.5
        # f1_scores_array = np.array([0.0]) # optimal_idx_f1 = 0 is enough
        optimal_idx_f1 = 0


    # Choose Youden's J as the primary method
    optimal_threshold = optimal_threshold_j

    # Calculate confusion matrix at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    
    # Ensure y_true and y_pred are not empty for confusion_matrix
    if y_true.size == 0 or y_pred.size == 0:
        tn, fp, fn, tp = 0, 0, 0, 0 # Default to zeros
    else:
        # Check if y_true has only one class. If so, confusion_matrix might return smaller array.
        if np.unique(y_true).size == 1:
            # Handle single-class case. Example: if all true labels are 0 (non-violent),
            # then tp and fn will be 0. tn is count of true negatives, fp is count of false positives.
            # Or if all true labels are 1 (violent), then tn and fp are 0.
            # We need to ensure ravel() gets 4 values.
            # sklearn's confusion_matrix with labels=[0, 1] can help ensure 2x2 matrix.
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
        else:  # Both classes are present in y_true
            cm = confusion_matrix(y_true, y_pred)
            # If y_pred contains only one class, cm might not be 2x2.
            # Example: y_true=[0,1,0,1], y_pred=[0,0,0,0] -> cm=[[2,0],[2,0]] (tn, fp, fn, tp)
            # Example: y_true=[0,0], y_pred=[0,1] -> cm=[[1,1],[0,0]]
            # The labels=[0,1] in the single-class case above handles this better.
            # For safety, if cm is not 2x2, use labels here too.
            if cm.size != 4:
                 cm_labeled = confusion_matrix(y_true, y_pred, labels=[0,1])
                 tn, fp, fn, tp = cm_labeled.ravel()
            else: # cm is already 2x2
                 tn, fp, fn, tp = cm.ravel()
             else:
                 tn, fp, fn, tp = cm.ravel()


    # Calculate various metrics at the optimal threshold
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 score at Youden's J optimal threshold.
    # Recalculate F1 for optimal_threshold_j for clarity and robustness,
    # as optimal_idx_j might relate to a raw 'thresholds' array that can include -inf/inf
    # or be misaligned if f1_scores_array was based on a filtered set of thresholds.
    if y_true.size > 0: # Ensure y_true is not empty for f1_score calculation
        y_pred_at_j_thresh = (y_score >= optimal_threshold_j).astype(int)
        f1_at_j_thresh = f1_score(y_true, y_pred_at_j_thresh)
    else:
        f1_at_j_thresh = 0.0

    # Ensure optimal_idx_j is valid for j_scores array
    # (especially after potential modification of thresholds array)
    valid_optimal_idx_j = optimal_idx_j if optimal_idx_j < len(j_scores) else (len(j_scores) -1 if len(j_scores) > 0 else 0)
    youdens_j_value = j_scores[valid_optimal_idx_j] if j_scores.size > 0 else 0.0


    metrics = {
        "threshold_j": float(optimal_threshold_j),
        "threshold_distance": float(optimal_threshold_d),
        "threshold_f1": float(optimal_threshold_f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_val),
        "f1_score": float(f1_at_j_thresh),  # F1 score at the chosen Youden's J threshold
        "youdens_j": float(youdens_j_value),
    }

    return float(optimal_threshold), metrics
