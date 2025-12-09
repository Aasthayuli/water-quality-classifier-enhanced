"""
Video Utility Functions
-----------------------
Helper functions for video processing and frame extraction.

Usage:
    from src.inference.utils_video import extract_frames
"""

import cv2
import os
import numpy as np
from PIL import Image
from src.utils.logger import get_logger
# from src.utils.logger import setup_logger # for standalone file testing


# Setup logger
# logger = setup_logger('utils_video', 'outputs/logs/test_utils_video.log')
logger = get_logger('predict_video')

def get_video_properties(video_path):
    """
    Get video properties (fps, resolution, frame count)
    
    Args:
        video_path (str): Path to video file
        
    Returns:
        dict: Video properties
        
    Example:
        props = get_video_properties('video.mp4')
        # {'fps': 30, 'width': 1920, 'height': 1080, 'frames': 300}
    """
    logger.info(f"Reading video properties: {video_path}")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    properties = {
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    
    logger.info(f"Video properties: {properties}")
    
    return properties


def extract_frames(video_path, output_dir=None, skip_frames=1):
    """
    Extract frames from video
    
    Args:
        video_path (str): Path to video file
        output_dir (str): Directory to save frames (optional)
        skip_frames (int): Process every Nth frame (1 = all frames)
        
    Returns:
        list: List of frames as numpy arrays
        
    Example:
        frames = extract_frames('video.mp4', skip_frames=10)
        # Returns every 10th frame
    """
    logger.info(f"Extracting frames from: {video_path}")
    logger.info(f"Skip frames: {skip_frames}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    frames = []
    frame_count = 0
    saved_count = 0
    
    # Create output directory if saving
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process every Nth frame
        if frame_count % skip_frames == 0:
            frames.append(frame)
            
            # Save frame if output directory provided
            if output_dir:
                frame_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
                cv2.imwrite(frame_path, frame)
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    
    logger.info(f"Total frames: {frame_count}")
    logger.info(f"Extracted frames: {saved_count}")
    
    return frames


def frame_to_pil(frame):
    """
    Convert OpenCV frame (BGR) to PIL Image (RGB)
    
    Args:
        frame (numpy.ndarray): OpenCV frame
        
    Returns:
        PIL.Image: RGB image
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def pil_to_frame(pil_image):
    """
    Convert PIL Image (RGB) to OpenCV frame (BGR)
    
    Args:
        pil_image (PIL.Image): PIL image
        
    Returns:
        numpy.ndarray: OpenCV frame
    """
    # Convert PIL to numpy array
    frame_rgb = np.array(pil_image)
    # Convert RGB to BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_bgr


def draw_prediction_on_frame(frame, prediction, confidence, position='top-left'):
    """
    Draw prediction text on frame
    
    Args:
        frame (numpy.ndarray): OpenCV frame
        prediction (str): Predicted class
        confidence (float): Confidence percentage
        position (str): Text position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
    Returns:
        numpy.ndarray: Frame with text overlay
    """
    # Copy frame to avoid modifying original
    frame_copy = frame.copy()
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Text properties
    text = f'{prediction.upper()}: {confidence:.1f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Color based on prediction
    colors = {
        'clean': (0, 255, 0),      # Green
        'muddy': (0, 165, 255),    # Orange
        'polluted': (0, 0, 255)    # Red
    }
    color = colors.get(prediction.lower(), (255, 255, 255))
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position
    padding = 20
    if position == 'top-left':
        x, y = padding, padding + text_height
    elif position == 'top-right':
        x, y = width - text_width - padding, padding + text_height
    elif position == 'bottom-left':
        x, y = padding, height - padding
    elif position == 'bottom-right':
        x, y = width - text_width - padding, height - padding
    else:
        x, y = padding, padding + text_height
    
    # Draw background rectangle
    cv2.rectangle(
        frame_copy,
        (x - 10, y - text_height - 10),
        (x + text_width + 10, y + baseline + 10),
        (0, 0, 0),
        -1  # Filled
    )
    
    # Draw text
    cv2.putText(
        frame_copy,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return frame_copy


# Testing
if __name__ == "__main__":
    """
    Test video utilities
    Run: python src/inference/utils_video.py
    """
    
    logger.info("="*60)
    logger.info("Testing Video Utilities")
    logger.info("="*60)
    
    test_video = 'data/polluted_vdo2.mp4'
    
    if os.path.exists(test_video):
        # Test 1: Get properties
        logger.info("1. Testing get_video_properties...")
        try:
            props = get_video_properties(test_video)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
        
        # Test 2: Extract frames
        logger.info("2. Testing extract_frames...")
        try:
            frames = extract_frames(test_video, skip_frames=30)
            logger.info(f"Extracted {len(frames)} frames")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
    else:
        logger.info(f"Test video not found: {test_video}")
        logger.info("Skipping tests (no video available)")

    
    logger.info("="*60)
    logger.info("Video utilities tests complete!")
    logger.info("="*60)