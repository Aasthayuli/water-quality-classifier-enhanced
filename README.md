# Water Quality Classifier (Enhanced Version – ResNet18 + Video Pipeline)

This repository contains the next-generation upgrade of my earlier Water Quality Classifier project.
The previous version was built using a simple CNN and focused only on image-based classification.
This enhanced release brings transfer learning with ResNet18, modular preprocessing, improved logging, and upcoming support for video-based water quality analysis.

---

## 🎥 Demo

[Click to watch the video](https://drive.google.com/file/d/1qdAFsgFfZ8iw7Yd2BMAp66LhWt1OdVhI/view?usp=sharing)

---

# 🚀 Project Overview

This classifier predicts three water-quality categories:

- Clean
- Muddy
- Polluted

The system leverages:

- Transfer Learning (ResNet18) for high-accuracy feature extraction
- A dedicated preprocessing pipeline
- Structured logging for training and inference
- Video-based classification pipeline added (frame extraction + prediction).

---

## 📊 Current Results

| Metric                       | Value                        |
| ---------------------------- | ---------------------------- |
| **Best Validation Accuracy** | 88.04%                       |
| **Training Accuracy**        | 80.55%                       |
| **Model**                    | ResNet18 (Transfer Learning) |
| **Epochs Trained**           | 25                           |
| **Date**                     | Dec 10, 2025                 |
| **Status**                   | Completed                    |

> **Note:** Model training is ongoing. Results will be updated as improvements are made.

---

## 🎯 Project Status

- [x] Dataset collection & preprocessing
- [x] Model architecture (ResNet18)
- [x] Training pipeline
- [x] Initial training (88.04% accuracy)
- [x] Model evaluation & analysis
- [x] Inference scripts (image)
- [x] Video frame extraction module
- [x] Video classification pipeline
- [x] Web application (Streamlit)
- [ ] Final optimization

---

## 📁 Project Structure

```
Water-Quality-Classifier/
├── app/                # Streamlit web app
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── evaluation/     # Evaluation scripts
│   ├── inference/      # Inference scripts
│   ├── models/         # ResNet18 architecture
│   ├── training/       # Training scripts
│   └── utils/          # Utilities (logging, config)
|   └──visualizations/  # Visualization scripts
├── configs/            # Configuration files
├── models/             # Saved models (.pth files)
├── outputs/            # Logs, graphs, predictions
└── data/               # Dataset (not included)
```

---

## 🚀 Quick Start

### Training

```bash
python src/training/train.py
```

### Evaluate trained model

```bash
python -m src.evaluation.evaluate.py --model models/resnet18/checkpoints/best_model.pth
```

### Visualize training results

```bash
python -m src.visualizations.preview_predictions --history outputs/logs/history.json --model models/resnet18/checkpoints/best_model.pth
```

### Predict on single image

```bash
python -m src.inference.predict_image --image test.jpg --model models/resnet18/checkpoints/best_model.pth
```

### Predict on video

```bash
python -m src.inference.predict_video --video input.mp4 --model models/resnet18/checkpoints/best_model.pth
```

### Configuration

Edit `configs/config.yaml` to modify hyperparameters.

---

## 🔧 Technologies Used

- **Framework:** PyTorch
- **Model:** ResNet18 (pretrained on ImageNet)
- **Preprocessing:** torchvision transforms
- **Augmentation:** Rotation, flip, color jitter
- **Optimizer:** Adam
- **Scheduler:** StepLR

---

## ⚠️ Known Limitations

- Model trained on high-quality stock images
- May need fine-tuning for mobile camera images
- Future work: Add real-world mobile camera data

---

## 🤝 Contributing

Feedback and suggestions welcome!

---

## 📄 License

MIT License
