"""
Image Prediction Script
-----------------------
Predict water quality for a single image.

Usage:
    python src/inference/predict_image.py --image test.jpg --model models/resnet18/best_model.pth
"""

import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet18_model import load_model
from src.data.preprocessing import preprocess_single_image, denormalize_image
from src.utils.logger import setup_logger


# Setup logger
logger = setup_logger('predict_image', 'outputs/logs/predict_image.log')


# Class names and colors
CLASS_NAMES = ['clean', 'muddy', 'polluted']
CLASS_COLORS = {
    'clean': 'blue',
    'muddy': 'brown',
    'polluted': 'black'
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict water quality from image')
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save prediction visualization (optional)'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display prediction result'
    )
    
    return parser.parse_args()


def setup_device(device=None):
    """Setup prediction device"""
    if device is not None:
        selected_device = device
    else:
        selected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {selected_device}")
    return selected_device


def predict_image(image_path, model, device):
    """
    Predict water quality for an image
    
    Args:
        image_path: Path to image
        model: Trained model
        device: Device to use
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    logger.info(f"Processing image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Preprocess image
    img_tensor = preprocess_single_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get results
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_value = confidence.item() * 100
    all_probs = probabilities[0].cpu().numpy() * 100
    
    logger.info(f"Prediction: {predicted_class} ({confidence_value:.2f}%)")
    
    return predicted_class, confidence_value, all_probs


def visualize_prediction(image_path, predicted_class, confidence, all_probs, save_path=None, show=False):
    """
    Visualize prediction result
    
    Args:
        image_path: Path to original image
        predicted_class: Predicted class name
        confidence: Confidence percentage
        all_probs: All class probabilities
        save_path: Path to save visualization (optional)
        show: Whether to display plot
    """
    logger.info("Creating visualization...")
    
    # Load original image
    img = Image.open(image_path)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original image with prediction
    ax1.imshow(img)
    ax1.axis('off')
    
    # Add prediction text
    color = CLASS_COLORS[predicted_class]
    title = f'Prediction: {predicted_class.upper()}\nConfidence: {confidence:.2f}%'
    ax1.set_title(title, fontsize=14, fontweight='bold', color=color)
    
    # Plot 2: Probability bar chart
    colors = [CLASS_COLORS[cls] for cls in CLASS_NAMES]
    bars = ax2.barh(CLASS_NAMES, all_probs, color=colors, alpha=0.7)
    
    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
    
    ax2.set_xlabel('Confidence (%)', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 105)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def print_prediction_summary(image_path, predicted_class, confidence, all_probs):
    """
    Print prediction summary to console
    
    Args:
        image_path: Path to image
        predicted_class: Predicted class
        confidence: Confidence value
        all_probs: All probabilities
    """
    print("\n" + "="*60)
    print("Water Quality Prediction")
    print("="*60)
    print(f"\nImage: {image_path}")
    print(f"\nPredicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll Class Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        bar = '█' * int(all_probs[i] / 2)  # Visual bar
        print(f"  {class_name:10s}: {all_probs[i]:5.2f}% {bar}")
    print("\n" + "="*60 + "\n")


def main():
    """Main prediction function"""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Water Quality Prediction - Single Image")
    logger.info("="*60)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model
    logger.info("Loading model...")
    try:
        model = load_model(
            checkpoint_path=args.model,
            num_classes=3,
            device=device
        )
        logger.info(f"Model loaded from: {args.model}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        sys.exit(1)
    
    # Predict
    try:
        predicted_class, confidence, all_probs = predict_image(
            args.image,
            model,
            device
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)
    
    # Print summary
    print_prediction_summary(args.image, predicted_class, confidence, all_probs)
    
    # Visualize
    if args.save or args.show:
        try:
            visualize_prediction(
                args.image,
                predicted_class,
                confidence,
                all_probs,
                save_path=args.save,
                show=args.show
            )
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    
    logger.info("="*60)
    logger.info("Prediction completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()