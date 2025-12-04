"""
Dataset Train-Test Splitter
----------------------------
Splits Image Dataset in train and test folders.

"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

from src.utils.logger import setup_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# Setup logger
logger = setup_logger(
    __name__,
    log_file=os.path.join(LOG_DIR, 'data_split.log')
)

def create_directories(base_path, classes):
    """
    Creates folders for train and test 
    
    Args:
        base_path: Base directory path
        classes: List of class names ['clean', 'muddy', 'polluted']
    """
    for split in ['train', 'test']:
        for class_name in classes:
            dir_path = os.path.join(base_path, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created: {dir_path}")


def split_dataset(
    source_dir,
    dest_dir,
    classes=['clean', 'muddy', 'polluted'],
    test_size=0.2,
    random_state=42
):
    """
    To split the dataset in train and test folders
    
    Args:
        source_dir: directory of source images (data/water_dataset/)
        dest_dir: destination directory to put images in train and test (same as source_dir)
        classes: Class names list
        test_size: Test data percentage (0.2 = 20%)
        random_state: Reproducibility seed
    """
    
    logger.info("="*60)
    logger.info("Starting Dataset Splitting...")
    logger.info("="*60)
    
    # Random seed setting
    random.seed(random_state)
    
    # Creating Train and test folders 
    create_directories(dest_dir, classes)
    
    # to track Statistics
    total_train = 0
    total_test = 0
    
    # Spilitting for every class
    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        
        # Checking if the folder exists or not
        if not os.path.exists(source_class_dir):
            logger.warning(f"{source_class_dir} not found! Skipping...")
            continue
        
        # creating list of every image
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            all_images.extend(list(Path(source_class_dir).glob(ext)))
        
        # If images are not found
        if len(all_images) == 0:
            logger.warning(f"No images found in {class_name}/ folder!")
            continue
        
        # Converting Image paths to strings
        all_images = [str(img) for img in all_images]
        
        logger.info(f"Class: {class_name.upper()}")
        logger.info(f"Total images: {len(all_images)}")
        
        # Train-test split
        train_images, test_images = train_test_split(
            all_images,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        
        logger.info(f"Train: {len(train_images)} | Test: {len(test_images)} - in class: {class_name}")
        
        # Copying Train images
        train_dir = os.path.join(dest_dir, 'train', class_name)
        for img_path in train_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(train_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        # Copying Test images 
        test_dir = os.path.join(dest_dir, 'test', class_name)
        for img_path in test_images:
            img_name = os.path.basename(img_path)
            dest_path = os.path.join(test_dir, img_name)
            shutil.copy2(img_path, dest_path)
        
        total_train += len(train_images)
        total_test += len(test_images)
        
        logger.info(f"Images Copied to train/ and test/ for class: {class_name}")
    
    # Final summary
    logger.info("="*60)
    logger.info("SPLIT COMPLETE!")
    logger.info("="*60)
    logger.info("Summary:")
    logger.info(f"Total Training Images: {total_train}")
    logger.info(f"Total Testing Images: {total_test}")
    logger.info(f"Total Images: {total_train + total_test}")
    logger.info(f"Split Ratio: {int((1-test_size)*100)}% train / {int(test_size*100)}% test")


def verify_split(base_dir, classes):
    """
    After Splitting, verifying if the split is correct
    
    Args:
        base_dir: Base directory
        classes: Class names
    """
    logger.info("Verifying Split...")
    
    for split in ['train', 'test']:
        logger.info(f"{split.upper()}-")
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            if os.path.exists(dir_path):
                count = len(list(Path(dir_path).glob('*.*')))
                logger.info(f"{class_name}: {count} images")
            else:
                logger.warning(f"{class_name}: Folder not found!")


if __name__ == "__main__":
    # Configuration
    BASE_DIR = 'data/water_dataset'
    CLASSES = ['clean', 'muddy', 'polluted']
    TEST_SIZE = 0.2  # 20% test, 80% train
    RANDOM_STATE = 42
    
    # User confirmation
    print("\n" + "_"*60)
    print("Dataset Split")
    print("_"*60)
    print(f"\nSource Directory: {BASE_DIR}/")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Split Ratio: {int((1-TEST_SIZE)*100)}% Train / {int(TEST_SIZE*100)}% Test")
    print("\nFolders that will be created:")
    print(f"  - {BASE_DIR}/train/")
    print(f"  - {BASE_DIR}/test/")
    
    user_input = input("\nContinue? (yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        # Split dataset
        split_dataset(
            source_dir=BASE_DIR,
            dest_dir=BASE_DIR,
            classes=CLASSES,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Verify split
        verify_split(BASE_DIR, CLASSES)
        
    else:
        logger.info("Split cancelled by user!")