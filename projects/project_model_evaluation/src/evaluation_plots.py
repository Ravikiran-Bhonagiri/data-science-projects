import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, calibration_curve

def plot_confusion_matrix_custom(y_true, y_pred, title="Confusion Matrix"):
    """Plots a custom confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_profit_curve(thresholds, profits, optimal_threshold, max_profit):
    """Plots the profit curve against probability thresholds."""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, label='Profit Curve', color='green', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.axhline(y=max_profit, color='blue', linestyle=':', label=f'Max Profit: ${max_profit:,.0f}')
    
    plt.title("Business Value: Profit vs. Decision Threshold")
    plt.xlabel("Probability Threshold (Risk Score Cutoff)")
    plt.ylabel("Total Profit ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_calibration_curve_custom(y_true, probs, model_name="Model"):
    """Plots calibration curve (reliability diagram)."""
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.title(f"Calibration Curve: {model_name}")
    plt.ylabel("Fraction of Positives (Actual Default Rate)")
    plt.xlabel("Mean Predicted Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_threshold_metrics(thresholds, precisions, recalls, f1_scores):
    """Plots Precision, Recall, and F1 against thresholds."""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.plot(thresholds, f1_scores[:-1], 'r:', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall-F1 vs Threshold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()
