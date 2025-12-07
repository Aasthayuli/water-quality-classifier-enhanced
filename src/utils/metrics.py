"""
Metrics Module
--------------
Evaluation metrics for model performance assessment.

Usage:
    from src.utils.metrics import calculate_metrics, plot_confusion_matrix
    
    metrics = calculate_metrics(y_true, y_pred, class_names)
    plot_confusion_matrix(cm, class_names, save_path='cm.png')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
# from src.utils.logger import setup_logger # for standalone file testing
from src.utils.logger import get_logger

# Setup logger
# logger = setup_logger('metrics', 'outputs/logs/test_metrics.log') # for standalone testing of this module
logger = get_logger('evaluation')

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Accuracy (0-100)
    """
    acc = accuracy_score(y_true, y_pred) * 100
    return acc


def calculate_metrics(y_true, y_pred, class_names=None):
    """
    Calculate all classification metrics
    
    Args:
        y_true: True labels (list or array)
        y_pred: Predicted labels (list or array)
        class_names: List of class names (optional)
        
    Returns:
        dict: Dictionary containing all metrics
        
    Example:
        metrics = calculate_metrics(y_true, y_pred, ['clean', 'muddy', 'polluted'])
    """
    logger.info("Calculating metrics...")
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Per-class metrics (weighted average)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    
    # Per-class metrics (individual)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0) * 100
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    # Add class names if provided
    if class_names is not None:
        metrics['class_names'] = class_names
    
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Precision: {precision:.2f}%")
    logger.info(f"Recall: {recall:.2f}%")
    logger.info(f"F1-Score: {f1:.2f}%")
    
    return metrics


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Example:
        print_classification_report(y_true, y_pred, ['clean', 'muddy', 'polluted'])
    """
    logger.info("="*60)
    logger.info("Classification Report")
    logger.info("="*60)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    logger.info("\n" + report)
    
    return report


def calculate_per_class_accuracy(y_true, y_pred, class_names=None):
    """
    Calculate accuracy for each class
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Per-class accuracy
        
    Example:
        acc = calculate_per_class_accuracy(y_true, y_pred, ['clean', 'muddy', 'polluted'])
        # {'clean': 95.5, 'muddy': 92.3, 'polluted': 98.1}
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    if class_names is not None:
        result = {class_names[i]: per_class_acc[i] for i in range(len(class_names))}
    else:
        result = {f"Class {i}": per_class_acc[i] for i in range(len(per_class_acc))}
    
    logger.info("Per-class accuracy:")
    for class_name, acc in result.items():
        logger.info(f"  {class_name} accuracy: {acc:.2f}%")
    
    return result


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix as heatmap
    
    Args:
        cm: Confusion matrix (numpy array or list)
        class_names: List of class names
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
        
    Example:
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, ['clean', 'muddy', 'polluted'], 'outputs/graphs/cm.png')
    """
    logger.info("Plotting confusion matrix...")
    
    # Convert to numpy if needed
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def plot_normalized_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    """
    Plot normalized confusion matrix (percentages) as heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        figsize: Figure size
    """
    logger.info("Plotting normalized confusion matrix...")
    
    # Convert to numpy if needed
    if isinstance(cm, list):
        cm = np.array(cm)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    
    plt.title('Normalized Confusion Matrix (%)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Normalized confusion matrix saved to: {save_path}")
    
    plt.close()


def save_metrics_to_file(metrics, save_path):
    """
    Save metrics to text file
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save file
        
    Example:
        save_metrics_to_file(metrics, 'outputs/metrics/results.txt')
    """
    logger.info(f"Saving metrics to: {save_path}")
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        
        # Overall metrics
        f.write("Overall Performance:\n")
        f.write(f"  Accuracy:  {metrics['accuracy']:.2f}%\n")
        f.write(f"  Precision: {metrics['precision']:.2f}%\n")
        f.write(f"  Recall:    {metrics['recall']:.2f}%\n")
        f.write(f"  F1-Score:  {metrics['f1_score']:.2f}%\n\n")
        
        # Per-class metrics
        if 'class_names' in metrics:
            f.write("Per-Class Performance:\n")
            for i, class_name in enumerate(metrics['class_names']):
                f.write(f"\n  {class_name.upper()}:\n")
                f.write(f"    Precision: {metrics['precision_per_class'][i]:.2f}%\n")
                f.write(f"    Recall:    {metrics['recall_per_class'][i]:.2f}%\n")
                f.write(f"    F1-Score:  {metrics['f1_per_class'][i]:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
    
    logger.info("Metrics saved successfully")


# Testing
if __name__ == "__main__":
    """
    Test metrics functions
    Run: python src/utils/metrics.py
    """
    
    logger.info("="*60)
    logger.info("Testing Metrics Module")
    logger.info("="*60)
    
    # Sample data
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2, 0]
    class_names = ['clean', 'muddy', 'polluted']
    
    # Test 1: Calculate metrics
    logger.info("1. Testing calculate_metrics...")
    metrics = calculate_metrics(y_true, y_pred, class_names)
    logger.info("Metrics calculated successfully!")
    
    # Test 2: Classification report
    logger.info("2. Testing classification report...")
    print_classification_report(y_true, y_pred, class_names)
    
    # Test 3: Per-class accuracy
    logger.info("3. Testing per-class accuracy...")
    per_class_acc = calculate_per_class_accuracy(y_true, y_pred, class_names)
    
    # Test 4: Confusion matrix plot
    logger.info("4. Testing confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, 'outputs/graphs/test_cm.png')
    plot_normalized_confusion_matrix(cm, class_names, 'outputs/graphs/test_cm_normalized.png')
    
    # Test 5: Save metrics
    logger.info("5. Testing save metrics...")
    save_metrics_to_file(metrics, 'outputs/test_metrics.txt')
    
    logger.info("="*60)
    logger.info("Metrics module tests complete!")
    logger.info("="*60)