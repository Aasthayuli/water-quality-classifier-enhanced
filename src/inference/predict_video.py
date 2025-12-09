"""
Video Prediction Script
-----------------------
Predict water quality for video frames and create annotated output.

Usage:
    python src/inference/predict_video.py --video input.mp4 --model best_model.pth --output output.mp4
"""

import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet18_model import load_model
from src.data.preprocessing import get_transforms
from src.inference.utils_video import (
    get_video_properties,
    extract_frames,
    frame_to_pil,
)
from src.utils.logger import create_timestamped_log

# Class names
CLASS_NAMES = ['clean', 'muddy', 'polluted']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict water quality from video')
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions/predicted_video.mp4',
        help='Path to save output video'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--skip_frames',
        type=int,
        default=5,
        help='Process every Nth frame (1=all, 5=every 5th, etc.)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference'
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


def predict_frames_batch(model, frames, transform, device, batch_size=8):
    """
    Predict classes for a batch of frames
    
    Args:
        model: Trained model
        frames: List of frames
        transform: Image transforms
        device: Device to use
        batch_size: Batch size for inference
        
    Returns:
        list: List of (predicted_class, confidence) tuples
    """
    model.eval()
    predictions = []
    
    # Process in batches
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        
        # Preprocess frames
        batch_tensors = []
        for frame in batch_frames:
            # Convert OpenCV frame to PIL
            pil_image = frame_to_pil(frame)
            # Apply transforms
            tensor = transform(pil_image)
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
        
        # Store results
        for pred_idx, conf in zip(predicted, confidences):
            pred_class = CLASS_NAMES[pred_idx.item()]
            conf_value = conf.item() * 100
            predictions.append((pred_class, conf_value))
    
    return predictions


def create_prediction_summary(predictions):
    """
    Create summary statistics from predictions
    
    Args:
        predictions: List of (class, confidence) tuples
        
    Returns:
        dict: Summary statistics
    """
    from collections import Counter
    
    # Count predictions
    pred_classes = [pred[0] for pred in predictions]
    class_counts = Counter(pred_classes)
    
    # Calculate percentages
    total = len(predictions)
    summary = {
        'total_frames': total,
        'class_distribution': {},
        'average_confidences': {}
    }
    
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        summary['class_distribution'][class_name] = {
            'count': count,
            'percentage': percentage
        }
        
        # Average confidence for this class
        class_confidences = [conf for cls, conf in predictions if cls == class_name]
        avg_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0
        summary['average_confidences'][class_name] = avg_conf
    
    # Dominant class
    if class_counts:
        dominant_class = class_counts.most_common(1)[0][0]
        summary['dominant_class'] = dominant_class
    else:
        summary['dominant_class'] = None
    
    return summary


def print_summary(summary):
    """
    Print prediction summary to console
    
    Args:
        summary: Summary dictionary
    """
    print("\n" + "="*60)
    print("Video Prediction Summary")
    print("="*60)
    print(f"\nTotal Frames Analyzed: {summary['total_frames']}")
    print(f"\nDominant Classification: {summary['dominant_class'].upper()}")
    
    print("\nClass Distribution:")
    for class_name in CLASS_NAMES:
        dist = summary['class_distribution'][class_name]
        avg_conf = summary['average_confidences'][class_name]
        bar = '█' * int(dist['percentage'] / 2)
        print(f"  {class_name:10s}: {dist['count']:4d} frames ({dist['percentage']:5.1f}%) | Avg Conf: {avg_conf:5.1f}% {bar}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main video prediction function"""
    args = parse_args()

    # Setup logger
    LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")
    global logger
    logger = create_timestamped_log('predict_video', LOG_DIR)
    
    logger.info("="*60)
    logger.info("Water Quality Prediction - Video")
    logger.info("="*60)
    
    # Setup device
    device = setup_device(args.device)
    
    # Get video properties
    logger.info("Analyzing video...")
    try:
        props = get_video_properties(args.video)
        logger.info(f"Video: {args.video}")
        logger.info(f"Resolution: {props['width']}x{props['height']}")
        logger.info(f"FPS: {props['fps']}")
        logger.info(f"Total frames: {props['frames']}")
    except Exception as e:
        logger.error(f"Failed to read video: {str(e)}")
        sys.exit(1)
    
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
    
    # Get transforms
    transform = get_transforms('test')
    
    # Extract frames
    logger.info(f"Extracting frames (skip={args.skip_frames})...")
    try:
        frames = extract_frames(args.video, skip_frames=args.skip_frames)
        logger.info(f"Extracted {len(frames)} frames for processing")
    except Exception as e:
        logger.error(f"Failed to extract frames: {str(e)}")
        sys.exit(1)
    
    # Predict
    logger.info("Predicting water quality for frames...")
    try:
        predictions = predict_frames_batch(
            model,
            frames,
            transform,
            device,
            batch_size=args.batch_size
        )
        logger.info(f"Completed predictions for {len(predictions)} frames")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)
    
    # Create summary
    summary = create_prediction_summary(predictions)
    
    # Print summary
    print_summary(summary)
    
    # Log summary
    logger.info("Prediction Summary:")
    logger.info(f"Dominant class: {summary['dominant_class']}")
    for class_name in CLASS_NAMES:
        dist = summary['class_distribution'][class_name]
        logger.info(f"{class_name}: {dist['count']} frames ({dist['percentage']:.1f}%)")
    
    logger.info("="*60)
    logger.info("Video prediction completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()