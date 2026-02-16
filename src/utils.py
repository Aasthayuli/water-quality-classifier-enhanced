import torch
from src.data_pipeline import get_transforms
from src.model import WaterQualityResNet18
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASSES = ['clean', 'muddy', 'polluted']


def load_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WaterQualityResNet18(num_classes=3, pretrained=True, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, device

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss'])+1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axs[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    axs[0].plot(epochs, history['val_loss'], label='Val Loss', color='red', linewidth=2)
    axs[0].set_title("Loss Curve")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Accuracy
    axs[1].plot(epochs, history['train_acc'], label='Train Acc', color='green', linewidth=2)
    axs[1].plot(epochs, history['val_acc'], label='Val Acc', color='brown', linewidth=2)
    axs[1].set_title("Accuracy Curves")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualize_samples(model, image_paths, device, class_names):
    model.eval()
    _, transform = get_transforms()
    plt.figure(figsize=(12, 4))

    for i, img_path in enumerate(image_paths):
        if i >= 6:
            break

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            predicted_class = class_names[pred.item()]
            confidence = conf.item() * 100

        plt.subplot(2,3, i + 1)
        plt.imshow(img)
        plt.axis('off')
        colors= {'clean':'green', 'muddy':'orange', 'polluted':'red'}
        plt.title(f"{predicted_class}\n{confidence:.1f}%", color=colors[predicted_class])

    plt.tight_layout()
    plt.show()

def extract_frames(video_path, resize=(224,224)):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames