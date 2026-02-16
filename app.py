import streamlit as st
import torch
from src.data_pipeline import get_transforms
from src.model import WaterQualityResNet18
from src.predict import predict_video, predict_image
import os
import requests

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = "artifacts/best_model.pth"

@st.cache_resource
def load_transform():
    _, transform = get_transforms()
    return transform

def download_model_from_drive():
    os.makedirs("artifacts", exist_ok=True)  

    file_id = "14lnAyZietBRvBk3Ai-RgVntGYc3iqMgq"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)



@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model Weights . . .")
        download_model_from_drive()
        st.success("Download complete.")

    model = WaterQualityResNet18(num_classes=3, pretrained=True, freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = load_transform()
model = load_model()

class_names = ['clean','muddy','polluted']

st.title("Water Quality Classifier")
tab0, tab1, tab2 = st.tabs(["About Model","Image Prediction", "video Prediction"])

with tab0:
    st.markdown("""
#### Overview
This application uses a deep learning model based on **ResNet18 architecture** to classify water quality from images and videos.

The model has been fine-tuned using **transfer learning** and can categorize water samples into:

- 🟢 Clean  
- 🟤 Muddy  
- 🔴 Polluted  

---

#### Model Architecture

- Base Model: **ResNet18 (Pretrained on ImageNet)**
- Framework: PyTorch
- Input Resolution: 224 X 224
- Loss Function: CrossEntropyLoss
- Device Support: CPU / GPU

ResNet18 was selected because it provides an optimal balance between computational efficiency and classification performance.  
Its residual connections help in learning deep visual representations without vanishing gradient issues.

---

#### Inference Pipeline

##### For Images:
1. Image upload  
2. Resize & normalization  
3. Model forward pass  
4. Softmax probability  
5. Final prediction with confidence score  

##### For Videos:
1. Video upload  
2. Frame extraction  
3. Individual frame classification  
4. Aggregation of predictions  
5. Final averaged result  

---

#### Performance

The model achieves approximately **75% validation accuracy** and demonstrates consistent classification across different turbidity conditions.

---
""")

with tab1:

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        
        with st.spinner("Analyzing image. . ."):
            predicted_class, confidence = predict_image(uploaded_file, transform, model, device)
        st.image(uploaded_file)
        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")

with tab2:

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], key="video")

    if uploaded_video is not None:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        with st.spinner("Processing Video Frames. . ."):
            final_class_index, avg_confidence = predict_video(tfile.name, transform, model, device)
        st.video(uploaded_video)
        st.success(f"Final Prediction: {class_names[final_class_index]}")
        st.info(f"Average Confidence: {avg_confidence:.2f}%")

