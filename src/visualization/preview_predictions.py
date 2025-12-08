"""
Prediction Preview & Visualization
-----------------------------------
Visualize training history, confusion matrix, and sample predictions.

Usage:
    python src/visualization/preview_predictions.py --history history.json --model best_model.pth
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet18_model import load_model
from src.data.preprocessing import preprocess_single_image
from src.utils.logger import setup_logger


# Setup logger
logger = setup_logger('visualization', 'outputs/logs/visualization.log')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Class names
CLASS_NAMES = ['clean', 'muddy', 'polluted']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize Training Results')
    
    parser.add_argument(
        '--history',
        type=str,
        help='Path to training history JSON file'
    )
    
    parser.add_argument(
        '--confusion_matrix',
        type=str,
        help='Path to confusion matrix JSON'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to model for sample predictions'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Sample images to predict'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/graphs',
        help='Directory to save visualizations'
    )
    
    return parser.parse_args()


def plot_training_curves(history, save_path='outputs/graphs'):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history: Dictionary with train_loss, train_acc, val_loss, val_acc
        save_path: Path to save plot
    """
    logger.info("Plotting training curves...")
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_path, 'training_curves.png')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to: {save_path}")
    
    plt.close()


def plot_learning_rate_schedule():
    pass

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix (2D array)
        save_path: Path to save plot
    """
    logger.info("Plotting confusion matrix...")
    
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    plt.close()


def visualize_sample_predictions(model, image_paths, device, save_path=None):
    """
    Visualize predictions for sample images
    
    Args:
        model: Trained model
        image_paths: List of image paths
        device: Device to use
        save_path: Path to save visualization
    """
    logger.info(f"Visualizing predictions for {len(image_paths)} images...")
    
    n_images = len(image_paths)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    model.eval()
    
    for idx, img_path in enumerate(image_paths):
        if idx >= len(axes):
            break
        
        # Load and preprocess image
        img = Image.open(img_path)
        img_tensor = preprocess_single_image(img_path, mode='test')
        img_tensor = img_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASS_NAMES[predicted.item()]
        confidence_value = confidence.item() * 100
        
        # Plot
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Color based on prediction
        colors = {'clean': 'blue', 'muddy': 'brown', 'polluted': 'black'}
        color = colors[predicted_class]
        
        title = f'{predicted_class.upper()}\n{confidence_value:.1f}%'
        axes[idx].set_title(title, fontsize=12, fontweight='bold', color=color)
    
    # Hide extra subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Sample predictions saved to: {save_path}")
    
    plt.close()


def create_summary_visualization(history, cm, save_path=None):
    """
    Create a comprehensive summary visualization
    
    Args:
        history: Training history
        cm: Confusion matrix
        save_path: Path to save
    """
    logger.info("Creating summary visualization...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Training Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Best Metrics Summary
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    best_val_acc = max(history['val_acc'])
    best_val_epoch = history['val_acc'].index(best_val_acc) + 1
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    
    summary_text = f"""
    TRAINING SUMMARY
    ================
    
    Best Val Accuracy: {best_val_acc:.2f}%
    Best Epoch: {best_val_epoch}
    
    Final Train Acc: {final_train_acc:.2f}%
    Final Val Acc: {final_val_acc:.2f}%
    
    Total Epochs: {len(epochs)}
    """
    
    ax3.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[1, :])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax4,
        square=True
    )
    ax4.set_title('Confusion Matrix', fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Summary visualization saved to: {save_path}")
    
    plt.close()


def main():
    """Main visualization function"""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Training Results Visualization")
    logger.info("="*60)
    
    # Create output directory
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs", "graphs")
    os.makedirs(os.path.join(base_output_dir, args.output_dir), exist_ok=True)
    
    # Load training history
    if args.history:
        logger.info(f"Loading training history from: {args.history}")
        try:
            with open(args.history, 'r') as f:
                history = json.load(f)
            
            # Plot training curves
            curves_path = os.path.join(base_output_dir,args.output_dir)
            plot_training_curves(history, save_path=curves_path)
            
        except Exception as e:
            logger.error(f"Failed to plot training curves: {str(e)}")
    
    # Load and plot confusion matrix
    if args.confusion_matrix:
        logger.info(f"Loading confusion matrix from: {args.confusion_matrix}")
        try:
            with open(args.confusion_matrix, 'r') as f:
                data = json.load(f)
                cm = np.array(data['confusion_matrix'])
            
            cm_path = os.path.join(args.output_dir, 'confusion_matrix_viz.png')
            plot_confusion_matrix(cm, save_path=cm_path)
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}")
    
    # Create summary if both available
    if args.history and args.confusion_matrix:
        try:
            summary_path = os.path.join(args.output_dir, 'training_summary.png')
            create_summary_visualization(history, cm, save_path=summary_path)
        except Exception as e:
            logger.error(f"Failed to create summary: {str(e)}")
    
    # Visualize sample predictions
    if args.model and args.images:
        logger.info("Visualizing sample predictions...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = load_model(args.model, num_classes=3, device=device)
            
            samples_path = os.path.join(args.output_dir, 'sample_predictions.png')
            visualize_sample_predictions(
                model,
                args.images,
                device,
                save_path=samples_path
            )
        except Exception as e:
            logger.error(f"Failed to visualize predictions: {str(e)}")
    
    logger.info("="*60)
    logger.info(f"Visualizations saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()