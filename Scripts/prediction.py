from src.predict import predict_image, predict_video
from src.data_pipeline import get_transforms
from src.utils import load_model
import os

CLASSES = ['clean', 'muddy', 'polluted']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'best_model.pth')

IMAGE = os.path.join(BASE_DIR,'data', 'split_dataset', 'test', 'muddy', 'muddy17.jpg')
VIDEO = os.path.join(BASE_DIR, 'data','clean_vdo2.mp4')

model, device = load_model(MODEL_PATH)

if __name__ == "__main__":
    _, transform = get_transforms()
    predict_image(IMAGE, transform, model, device)
    predict_video(VIDEO, transform, model, device)