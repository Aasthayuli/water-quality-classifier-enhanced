import os
from sklearn.model_selection import train_test_split
import random
import shutil
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

IMAGE_SIZE = 224
BATCH_SIZE = 32

def create_directories(base_dir, classes):
    for split in ['train', 'test']:
        for class_name in classes:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

def split_dataset(SOURCE_DIR, DEST_DIR, CLASSES, TEST_SIZE, RANDOM_STATE):
    random.seed(RANDOM_STATE)

    if os.path.exists(DEST_DIR):
        print("Dataset already exists. Skipping split.")
        return

    create_directories(DEST_DIR, CLASSES)

    for class_name in CLASSES:
        class_dir = os.path.join(SOURCE_DIR, class_name)

        images = list(Path(class_dir).glob("*.jpg")) + list(Path(class_dir).glob("*.png"))

        train_imgs, test_imgs = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        for img in train_imgs:
            shutil.copy(img, os.path.join(DEST_DIR, 'train', class_name, img.name))

        for img in test_imgs:
            shutil.copy(img, os.path.join(DEST_DIR, 'test', class_name, img.name))

        print(f"{class_name}: {len(train_imgs)} train | {len(test_imgs)} test")

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    return train_transform, test_transform

def get_dataloaders(train_dir, test_dir, batch_size):
    train_transform, test_transform = get_transforms()

    train_dataset = ImageFolder(train_dir, transform=train_transform) 
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader( 
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader