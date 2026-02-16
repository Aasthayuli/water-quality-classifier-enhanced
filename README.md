# Water Quality Classification using Deep Learning

A deep learning–based image classification system that predicts water quality categories from images.

The model classifies water samples into three categories:

- Clean
- Muddy
- Polluted

---

## Project Structure

```
└── 📁WQC
    └── 📁artifacts
        ├── best_model.pth
    └── 📁data
        └── 📁split_dataset
            └── 📁test
                └── 📁clean
                └── 📁muddy
                └── 📁polluted
            └── 📁train
                └── 📁clean
                └── 📁muddy
                └── 📁polluted
        └── 📁water_dataset
            └── 📁clean
            └── 📁muddy
            └── 📁polluted
        ├── clean_vdo2.mp4   # Example Video
    └── 📁notebooks
        ├── data_pipeline.ipynb
        ├── image_prediction.ipynb
        ├── model_training.ipynb
        ├── video_prediction.ipynb
    └── 📁Scripts
        ├── data_preparation.py
        ├── prediction.py
        ├── train_model.py
    └── 📁src
        ├── __init__.py
        ├── data_pipeline.py
        ├── model.py
        ├── predict.py
        ├── train.py
        ├── utils.py
    ├── app.py
    ├── README.md
    ├── requirements.txt
    └── structure.md
```

---

## Dataset Preparation

To split the dataset into training and testing sets:

```bash
python -m Scripts.data_preparation
```

This will:

- Read images from data/water_dataset
- Split into train/test
- Save to data/split_dataset

## Model

- Architecture: ResNet-based transfer learning
- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Evaluation Metric: Accurac

## Running the Streamlit App

```bash
streamlit run app.py
```

The app allows users to:

- Upload an image
- Run prediction
- View predicted water quality class

## Deployment Notes

- Model weights are not stored in the GitHub repository.
- During deployment, weights are loaded dynamically.
- Caching is used to prevent repeated model loading.

## Installation

Create virtual environment:

```bash
python -m venv .venv
```

Activate:

```bash
.venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Author

Final Year B.Tech Project

Water Quality Classification System
