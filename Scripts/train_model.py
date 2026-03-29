import torch
from src.train import train_model
from src.utils import plot_training_curves, visualize_samples
from src.train import evaluate_model
from src.model import WaterQualityResNet18
from src.data_pipeline import get_dataloaders
from logger.logging_config import AppLogger
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ['clean', 'muddy', 'polluted']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR =os.path.join(BASE_DIR, "data", "split_dataset", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "split_dataset", "test")

SAVE_PATH = os.path.join(BASE_DIR, "artifacts")

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(TRAIN_DIR, TEST_DIR,
                                                batch_size=32)
    
    AppLogger.set_up()
    model = WaterQualityResNet18(num_classes=3, pretrained=True, freeze_backbone=False)
    trained_model, history = train_model(model, train_loader, test_loader, epochs=15, lr=1e-3, save_path=SAVE_PATH)

    evaluate_model(trained_model, test_loader)

    plot_training_curves(history=history)

    image_paths = [os.path.join(TEST_DIR, 'muddy','muddy17.jpg'), os.path.join(TEST_DIR, 'clean','25.jpg'),
                  os.path.join(TEST_DIR, 'muddy','muddy24.jpg')]
    
    visualize_samples(trained_model, image_paths, device, CLASSES)