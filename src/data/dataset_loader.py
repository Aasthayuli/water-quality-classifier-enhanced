"""
Dataset Loader
--------------
PyTorch Dataset and DataLoader for water quality classification.

Usage:
    from src.data.dataset_loader import get_dataloaders
    
    train_loader, test_loader = get_dataloaders(
        train_dir='data/water_dataset/train',
        test_dir='data/water_dataset/test',
        batch_size=32
    )
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from src.data.preprocessing import get_transforms
from src.utils.logger import setup_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# Setup logger
logger = setup_logger(
    name='dataset_loader',
    log_file=os.path.join(LOG_DIR, 'dataset_loader.log')
)


# Class labels mapping
CLASS_NAMES = ['clean', 'muddy', 'polluted']


def get_dataloaders(train_dir, test_dir, batch_size=32, num_workers=4):
    """
    Create train and test dataloaders
    
    Args:
        train_dir (str): Path to training data
        test_dir (str): Path to test data
        batch_size (int): Batch size for training
        num_workers (int): number of background subprocesses that load the batches
        For Linux, you can go higher (like 8-16)num_workers for faster loading
        
    Returns:
        tuple: (train_loader, test_loader)
        
    Example:
        train_loader, test_loader = get_dataloaders(
            'data/water_dataset/train',
            'data/water_dataset/test',
            batch_size=32
        )
    """
    
    logger.info("="*60)
    logger.info("Creating DataLoaders")
    logger.info("="*60)
    
    # Get transforms
    train_transform = get_transforms('train')
    test_transform = get_transforms('test')
    
    # Create datasets
    logger.info(f"Loading training data from: {train_dir}")
    train_dataset = ImageFolder(train_dir, transform=train_transform)
    
    logger.info(f"Loading test data from: {test_dir}")
    test_dataset = ImageFolder(test_dir, transform=test_transform)
    
    # Log dataset info
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Class names: {train_dataset.classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
        num_workers=num_workers,
        pin_memory=True  # Faster transfer to GPU(batches are stored in page-locked memory & directly transferred to GPU)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for testing
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    logger.info("="*60)
    
    return train_loader, test_loader


def get_class_distribution(dataset):
    """
    Get distribution of classes in dataset
    
    Args:
        dataset: PyTorch Dataset
        
    Returns:
        dict: Class distribution {class_name: count}
        
    Example:
        dist = get_class_distribution(train_dataset)
        # {'clean': 126, 'muddy': 119, 'polluted': 126}
    """
    
    class_counts = {}
    
    for _, label in dataset:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return class_counts


def verify_dataset(data_dir):
    """
    Verify dataset structure and count images
    
    Args:
        data_dir (str): Path to dataset directory
        
    Returns:
        dict: Statistics about dataset
        
    Example:
        stats = verify_dataset('data/water_dataset/train')
    """
    
    logger.info(f"Verifying dataset: {data_dir}")
    
    stats = {
        'total_images': 0,
        'classes': {},
        'valid': True,
        'errors': []
    }
    
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        stats['valid'] = False
        stats['errors'].append(f"Directory not found: {data_dir}")
        return stats
    
    # Check each class folder
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            logger.warning(f"Class folder not found: {class_dir}")
            stats['errors'].append(f"Missing class: {class_name}")
            continue
        
        # Count images
        images = [f for f in os.listdir(class_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count = len(images)
        stats['classes'][class_name] = count
        stats['total_images'] += count
        
        logger.info(f"  {class_name}: {count} images")
    
    logger.info(f"Total images: {stats['total_images']}")
    
    return stats


# Testing
if __name__ == "__main__":
    """
    Test dataset loading
    Run: python src/data/dataset_loader.py
    """
    
    logger.info("="*60)
    logger.info("Testing Dataset Loader")
    logger.info("="*60)
    
    # Paths
    TRAIN_DIR = 'data/water_dataset/train'
    TEST_DIR = 'data/water_dataset/test'
    
    # Test 1: Verify datasets
    logger.info("\n1. Verifying datasets...")
    train_stats = verify_dataset(TRAIN_DIR)
    test_stats = verify_dataset(TEST_DIR)
    
    if not train_stats['valid'] or not test_stats['valid']:
        logger.error("Dataset verification failed!")
    else:
        logger.info("Dataset verification passed!")
    
    # Test 2: Create dataloaders
    logger.info("\n2. Creating dataloaders...")
    try:
        train_loader, test_loader = get_dataloaders(
            TRAIN_DIR,
            TEST_DIR,
            batch_size=8,
            num_workers=0  # 0 for testing on Windows
        )
        
        logger.info("DataLoaders created successfully!")
        
        # Test 3: Load one batch
        logger.info("3. Testing batch loading...")
        images, labels = next(iter(train_loader))
        
        logger.info(f"Batch images shape: {images.shape}")
        logger.info(f"Batch labels shape: {labels.shape}")
        logger.info(f"Expected images: [8, 3, 224, 224]")
        logger.info(f"Expected labels: [8]")
        
        # Test 4: Check data types
        logger.info("\n4. Checking data types...")
        logger.info(f"Images dtype: {images.dtype}")
        logger.info(f"Labels dtype: {labels.dtype}")
        logger.info(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
    
    logger.info("="*60)
    logger.info("Dataset loader tests complete!")
    logger.info("="*60)