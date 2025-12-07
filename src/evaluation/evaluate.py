"""
Model Evaluation Script
-----------------------
Evaluate trained model on test dataset and generate metrics.

Usage:
    python src/evaluation/evaluate.py --model models/resnet18/best_model.pth
"""

import os
import sys
import argparse
import json
import torch
from tqdm import tqdm
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet18_model import load_model
from src.data.dataset_loader import get_dataloaders
from src.utils.config_loader import load_config
from src.utils.metrics import (
    calculate_metrics,
    print_classification_report,
    calculate_per_class_accuracy,
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    save_metrics_to_file
)
from src.utils.logger import create_timestamped_log


# Class names
CLASS_NAMES = ['clean', 'muddy', 'polluted']


def parse_args():
    """Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments

    Examples:
    #### Default
    - python src/evaluation/evaluate.py --model models/resnet18/best_model.pth
    - python src/evaluation/evaluate.py won't use default config and output dir
    
    #### Custom
    - python src/evaluation/evaluate.py --model models/resnet18/best_model.pth --device cuda
    - python src/evaluation/evaluate.py --config configs/custom_config.yaml --device cpu --output_dir outputs/custom_evaluation

        args.model = 'models/resnet18/best_model.pth'
        args.config = 'configs/config.yaml'
        args.device = None
        args.output_dir = 'outputs/evaluation'
    
    """
    parser = argparse.ArgumentParser(description='Evaluate Water Quality Model')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/evaluation',
        help='Directory to save evaluation results'
    )
    
    return parser.parse_args()


def setup_device(device=None):
    """Setup evaluation device"""
    if device is not None:
        selected_device = device
    else:
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {selected_device}")
    return selected_device


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and collect predictions
    
    Args:
        model: Trained model
        dataloader: Test DataLoader
        device: Device to use
        
    Returns:
        tuple: (y_true, y_pred, y_probs)
    """
    logger.info("Running model evaluation...")
    
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for images, labels in pbar:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probabilities.cpu().numpy())
    
    logger.info(f"Evaluated {len(y_true)} samples")
    
    return y_true, y_pred, y_probs


def save_evaluation_results(metrics, y_true, y_pred, output_dir):
    """
    Save evaluation results to files
    
    Args:
        metrics: Dictionary of metrics
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save results
    """
    logger.info(f"Saving evaluation results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save metrics as JSON
    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved: {metrics_json_path}")
    
    # 2. Save metrics as text
    metrics_txt_path = os.path.join(output_dir, 'metrics.txt')
    save_metrics_to_file(metrics, metrics_txt_path)
    
    # 3. Save confusion matrix plot
    cm = metrics['confusion_matrix']
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=cm_path)
    
    # 4. Save normalized confusion matrix
    cm_norm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plot_normalized_confusion_matrix(cm, CLASS_NAMES, save_path=cm_norm_path)
    
    # 5. Save predictions
    predictions_path = os.path.join(output_dir, 'predictions.json')
    predictions_data = {
        'y_true': [int(x) for x in y_true],
        'y_pred': [int(x) for x in y_pred],
        'class_names': CLASS_NAMES
    }
    with open(predictions_path, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    logger.info(f"Predictions saved: {predictions_path}")


def main():
    """Main evaluation function"""
    # Setup logger
    LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
    global logger
    logger = create_timestamped_log('evaluation', LOG_DIR)

    args = parse_args()
    
    logger.info("="*60)
    logger.info("Water Quality Classification - Model Evaluation")
    logger.info("="*60)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    logger.info("="*60)
    logger.info("Loading Configuration")
    logger.info("="*60)
    
    try:
        config = load_config(args.config)
        logger.info(f"Config loaded from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        sys.exit(1)
    
    # Setup device
    logger.info("="*60)
    logger.info("Setting Up Device")
    logger.info("="*60)
    
    device = setup_device(args.device)
    
    # Load model
    logger.info("="*60)
    logger.info("Loading Model")
    logger.info("="*60)
    
    try:
        model = load_model(
            checkpoint_path=args.model,
            num_classes=config['model']['num_classes'],
            device=device
        )
        logger.info(f"Model loaded from: {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Load test data
    logger.info("="*60)
    logger.info("Loading Test Dataset")
    logger.info("="*60)
    
    try:
        _, test_loader = get_dataloaders(
            train_dir=config['data']['train_dir'],
            test_dir=config['data']['test_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers']
        )
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        sys.exit(1)
    
    # Evaluate model
    logger.info("="*60)
    logger.info("Evaluating Model")
    logger.info("="*60)
    
    try:
        y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Calculate metrics
    logger.info("="*60)
    logger.info("Calculating Metrics")
    logger.info("="*60)
    
    try:
        metrics = calculate_metrics(y_true, y_pred, CLASS_NAMES)
        
        # Print classification report
        print_classification_report(y_true, y_pred, CLASS_NAMES)
        
        # Per-class accuracy
        per_class_acc = calculate_per_class_accuracy(y_true, y_pred, CLASS_NAMES)
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {str(e)}")
        sys.exit(1)
    
    # Save results
    logger.info("="*60)
    logger.info("Saving Results")
    logger.info("="*60)
    
    try:
        # Create timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
        
        save_evaluation_results(metrics, y_true, y_pred, output_dir)
        
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
    
    # Print summary
    logger.info("="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Test Samples: {len(y_true)}")
    logger.info(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.2f}%")
    logger.info(f"Recall: {metrics['recall']:.2f}%")
    logger.info(f"F1-Score: {metrics['f1_score']:.2f}%")
    
    logger.info("Per-Class Accuracy:")
    for class_name, acc in per_class_acc.items():
        logger.info(f"  {class_name}: {acc:.2f}%")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    logger.info("="*20+"Evaluation completed successfully!"+"="*20)


if __name__ == "__main__":
    main()