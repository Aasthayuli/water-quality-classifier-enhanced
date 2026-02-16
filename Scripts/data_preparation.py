from src.data_pipeline import split_dataset
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "data", "water_dataset")
DEST_DIR = os.path.join(BASE_DIR, "data", "split_dataset")

CLASSES = ['clean', 'muddy', 'polluted']
TEST_SIZE = 0.2
RANDOM_STATE = 42

if __name__ == "__main__":
    split_dataset(SOURCE_DIR, DEST_DIR, CLASSES, TEST_SIZE, RANDOM_STATE)