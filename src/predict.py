import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import extract_frames

CLASSES = ['clean', 'muddy', 'polluted']


def predict_image(image_path, transform, model, device):
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence = confidence.item() * 100

    plt.imshow(img)
    plt.axis('off')

    colors = {'clean' : 'green',
              'muddy' : 'orange',
              'polluted' : 'red'}
    
    plt.title(f"{predicted_class.upper()} ({confidence:.2f}%)",
              color = colors[predicted_class],
              fontsize=12,
              fontweight='bold')
    
    plt.show()
    return predicted_class, confidence


def predict_video(video_path, transform, model, device):
    frames = extract_frames(video_path)

    frame_preds = []
    frame_confidences = []
    model.eval()

    for frame in frames:
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            conf, pred = torch.max(probabilities, 1)

        frame_preds.append(pred.item())
        frame_confidences.append(conf.item())

    # Majority vote for class
    from collections import Counter
    pred_class = Counter(frame_preds).most_common(1)[0][0]
    avg_confidence = sum(frame_confidences) / len(frame_confidences)

    plt.figure(figsize=(8, 6))
    plt.imshow(frames[-1])
    plt.axis("off")
    plt.title(f"Video Prediction: {CLASSES[pred_class]} (Avg conf: {avg_confidence * 100:.2f}%)")
    plt.show()

    return pred_class, avg_confidence* 100

