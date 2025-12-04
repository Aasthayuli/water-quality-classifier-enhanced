"""
Data Preprocessing
------------------
Image preprocessing and augmentation for water quality classification.
Defines transforms for training and testing.

Usage:
    from src.data.preprocessing import get_transforms
    
    train_transforms = get_transforms(mode='train')
    test_transforms = get_transforms(mode='test')
"""

import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


# from src.utils.logger import setup_logger
from src.utils.logger import get_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# Setup logger for testing
# logger = setup_logger(
#     __name__,
#     log_file=os.path.join(LOG_DIR, 'preprocessing_test.log')
# ) # for standalone testing
logger = get_logger(__name__)

# ImageNet statistics (ResNet18 pre-trained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image size for ResNet18
IMAGE_SIZE = 224


def get_train_transforms(image_size=IMAGE_SIZE):
    """
    Get data augmentation transforms for training
    
    Augmentation helps model generalize better by:
    - Creating variations of existing images
    - Preventing overfitting
    - Making model robust to different conditions
    
    Args:
        image_size (int): Target image size (default: 224 for ResNet)
        
    Returns:
        torchvision.transforms.Compose: Training transforms
    """
    
    train_transforms = transforms.Compose([
        # Resize to slightly larger size first
        transforms.Resize((image_size + 32, image_size + 32)),
        
        # Random crop back to target size (adds variation)
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),  # Crop between 80% to 100% of image
            ratio=(0.9, 1.1)   # Slight aspect ratio variation
        ),
        
        # Random horizontal flip (50% chance)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Random rotation (±15 degrees)
        transforms.RandomRotation(degrees=15),
        
        # Color jittering (brightness, contrast, saturation)
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness
            contrast=0.2,    # ±20% contrast
            saturation=0.2,  # ±20% saturation
            hue=0.1          # ±10% hue
        ),
        
        # Convert PIL Image to Tensor
        transforms.ToTensor(),
        
        # Normalize using ImageNet statistics
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return train_transforms


def get_test_transforms(image_size=IMAGE_SIZE):
    """
    Get transforms for testing/validation
    
    No augmentation for testing - only basic preprocessing:
    - Resize to fixed size
    - Convert to tensor
    - Normalize
    
    Args:
        image_size (int): Target image size (default: 224)
        
    Returns:
        torchvision.transforms.Compose: Test transforms
    """
    
    test_transforms = transforms.Compose([
        # Resize to target size
        transforms.Resize((image_size, image_size)),
        
        # Convert PIL Image to Tensor
        transforms.ToTensor(),
        
        # Normalize using same statistics as training
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return test_transforms


def get_transforms(mode='train', image_size=IMAGE_SIZE):
    """
    Get appropriate transforms based on mode
    
    Args:
        mode (str): 'train' or 'test'/'val'
        image_size (int): Target image size
        
    Returns:
        torchvision.transforms.Compose: Transforms for specified mode
        
    Example:
        train_tf = get_transforms('train')
        test_tf = get_transforms('test')
    """
    
    if mode == 'train':
        return get_train_transforms(image_size)
    elif mode in ['test', 'val', 'validation']:
        return get_test_transforms(image_size)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'test'")


def denormalize_image(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize a normalized image tensor for visualization
    
    Useful for displaying images after preprocessing
    
    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W)
        mean (list): Mean values used for normalization
        std (list): Std values used for normalization
        
    Returns:
        torch.Tensor: Denormalized image tensor
        
    Example:
        normalized_img = transforms(image)
        original_img = denormalize_image(normalized_img)
    """
    
    # Clone tensor to avoid modifying original
    denorm_tensor = tensor.clone()
    
    # Denormalize each channel
    for tensor, m, s in zip(denorm_tensor, mean, std):
        tensor.mul_(s).add_(m)  # tensor = tensor * std + mean
    
    # Clip values to [0, 1]
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    return denorm_tensor


def preprocess_single_image(image_path, mode='test'):
    """
    Preprocess a single image file
    
    Useful for inference on new images
    
    Args:
        image_path (str): Path to image file
        mode (str): 'train' or 'test' preprocessing
        
    Returns:
        torch.Tensor: Preprocessed image tensor (1, C, H, W)
        
    Example:
        img_tensor = preprocess_single_image('test.jpg', mode='test')
        prediction = model(img_tensor)
    """
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get appropriate transforms
    transform = get_transforms(mode)
    
    # Apply transforms
    img_tensor = transform(image)
    
    # Add batch dimension (1, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor


def check_image_quality(image_path):
    """
    Check if image meets quality requirements
    
    Basic quality checks:
    - File can be opened
    - Image has valid dimensions
    - Image is not too small
    
    Args:
        image_path (str): Path to image
        
    Returns:
        tuple: (is_valid, message)
        
    Example:
        valid, msg = check_image_quality('image.jpg')
        if not valid:
            print(f'Invalid image: {msg}')
    """
    
    try:
        # Try opening image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        img = img.convert('RGB')
        
        # Check dimensions
        width, height = img.size
        
        if width < 50 or height < 50:
            return False, f"Image too small: {width}x{height}"
        
        # Check if image data is valid
        img_array = np.array(img)
        if img_array.size == 0:
            return False, "Empty image"
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Error loading image: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    """
    Test preprocessing functions
    Run: python src/data/preprocessing.py
    """
    
    logger.info("="*60)
    logger.info("Testing Preprocessing Functions")
    logger.info("="*60)
    
    # Test 1: Get transforms
    logger.info("1. Getting transforms...")
    train_tf = get_transforms('train')
    test_tf = get_transforms('test')
    logger.info(f"Train transforms: {len(train_tf.transforms)} operations")
    logger.info(f"Test transforms: {len(test_tf.transforms)} operations")
    
    # Test 2: Display transform details
    logger.info("2. Train transforms details:")
    for i, t in enumerate(train_tf.transforms, 1):
        logger.info(f"   {i}. {t.__class__.__name__}")
    
    logger.info("3. Test transforms details:")
    for i, t in enumerate(test_tf.transforms, 1):
        logger.info(f"   {i}. {t.__class__.__name__}")
    
    # Test 3: Check if dummy image can be created
    logger.info("4. Testing with dummy image...")
    try:
        # Create dummy RGB image
        dummy_img = Image.new('RGB', (300, 300), color='red')
        
        # Apply train transforms
        train_tensor = train_tf(dummy_img)
        logger.info(f"Train output shape: {train_tensor.shape}")
        logger.info(f"Expected: torch.Size([3, 224, 224])")
        
        # Apply test transforms
        test_tensor = test_tf(dummy_img)
        logger.info(f"Test output shape: {test_tensor.shape}")
        logger.info(f"Expected: torch.Size([3, 224, 224])")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Test 4: Denormalization
    logger.info("5. Testing denormalization...")
    try:
        normalized = test_tf(dummy_img)
        denormalized = denormalize_image(normalized)
        logger.info(f"Denormalized range: [{denormalized.min():.3f}, {denormalized.max():.3f}]")
        logger.info(f"Expected: [0.000, 1.000]")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    logger.info("="*60)
    logger.info("Preprocessing tests complete!")
    logger.info("="*60)
    