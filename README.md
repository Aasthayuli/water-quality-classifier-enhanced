# Water Quality Classifier (Enhanced Version – ResNet18 + Video Pipeline)

This repository contains the next-generation upgrade of my earlier Water Quality Classifier project.
The previous version was built using a simple CNN and focused only on image-based classification.
This enhanced release brings transfer learning with ResNet18, modular preprocessing, improved logging, and upcoming support for video-based water quality analysis.

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
- Upcoming video frame extraction module for real-time classification

---

## 📊 Current Results

| Metric                       | Value                        |
| ---------------------------- | ---------------------------- |
| **Best Validation Accuracy** | 92.39%                       |
| **Training Accuracy**        | 94.25%                       |
| **Model**                    | ResNet18 (Transfer Learning) |
| **Epochs Trained**           | 25                           |
| **Date**                     | Dec 6, 2025                  |
| **Status**                   | 🚧 Work in Progress          |

> **Note:** Model training is ongoing. Results will be updated as improvements are made.

---

## 🎯 Project Status

- [x] Dataset collection & preprocessing
- [x] Model architecture (ResNet18)
- [x] Training pipeline
- [x] Initial training (92.39% accuracy)
- [ ] Model evaluation & analysis
- [ ] Inference scripts (image/video)
- [ ] Web application (Streamlit)
- [ ] Final optimization

---

## 📁 Project Structure

```
Water-Quality-Classifier/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # ResNet18 architecture
│   ├── training/       # Training scripts
│   └── utils/          # Utilities (logging, config)
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

### Configuration

Edit `configs/config.yaml` to modify hyperparameters.

---

## 📈 Training Progress

**Latest Model:** `best_model_20251206_012806_acc92.39.pth`

Training logs available in `outputs/logs/`

---

## 🔧 Technologies Used

- **Framework:** PyTorch
- **Model:** ResNet18 (pretrained on ImageNet)
- **Preprocessing:** torchvision transforms
- **Augmentation:** Rotation, flip, color jitter
- **Optimizer:** Adam
- **Scheduler:** StepLR

---

## 📝 To-Do

- Run comprehensive evaluation
- Add confusion matrix visualization
- Implement video inference
- Create Streamlit dashboard
- Optimize model further
- Add deployment scripts

---

## ⚠️ Known Limitations

- Model trained on high-quality stock images
- May need fine-tuning for mobile camera images
- Performance on low-light conditions untested
- Future work: Add real-world mobile camera data

---

## 🤝 Contributing

This is a work-in-progress project. Feedback and suggestions welcome!

---

## 📄 License

MIT License
