import streamlit as st
import torch
from src.model import WaterQualityResNet18
from src.data_pipeline import get_transforms
from src.predict import predict_video, predict_image
from huggingface_hub import hf_hub_download

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_transform():
    _, transform = get_transforms()
    return transform


@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Aasthayuli/water-quality-classifier",  
        filename="best_model.pth"
    )

    model = WaterQualityResNet18(num_classes=3, pretrained=False)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Model loading failed: {e}")

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

The model achieves approximately **85% validation accuracy** and demonstrates consistent classification across different turbidity conditions.

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

