"""
Water Quality Classification - Streamlit App
--------------------------------------------
Interactive web application for water quality prediction.

Run:
    streamlit run app/app.py

"""

import os
import sys
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import tempfile
from collections import Counter

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inference.utils_video import (
    get_video_properties,
    extract_frames,
    frame_to_pil
)

from src.models.resnet18_model import WaterQualityResNet18
from src.data.preprocessing import get_transforms


# Configuration
st.set_page_config(
    page_title="Water Quality Classifier",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Constants
CLASS_NAMES = ['Clean', 'Muddy', 'Polluted']
CLASS_COLORS = {
    'Clean': '#28a745',      # Green
    'Muddy': '#fd7e14',      # Orange
    'Polluted': '#dc3545'    # Red
}

# Model configuration
LOCAL_MODEL_PATH = "models/resnet18/checkpoints/best_model.pth"
MODEL_GDRIVE_ID = st.secrets["MODEL_GDRIVE_ID"]


@st.cache_resource
def download_model_from_drive():
    """
    Download model from Google Drive if not exists locally
    
    Returns:
        str: Path to model file
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Model found !")
        return LOCAL_MODEL_PATH
    
    st.info("Downloading model ...")
    
    try:
        # Create models directory
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        
        # Download from Google Drive
        url = f"https://drive.google.com/uc?export=download&id={MODEL_GDRIVE_ID}"
        gdown.download(url, LOCAL_MODEL_PATH, quiet=False)
        
        st.success(f"Model downloaded!")
        return LOCAL_MODEL_PATH
        
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        st.info("Please check MODEL_GDRIVE_ID in Streamlit secrets")
        return None


@st.cache_resource
def load_model():
    """
    Load model with caching
    
    Returns:
        model: Loaded PyTorch model
    """
    # Download model if needed
    model_path = download_model_from_drive()
    
    if model_path is None:
        return None
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = WaterQualityResNet18(num_classes=3, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


def predict_image(image, model):
    """
    Predict water quality for uploaded image
    
    Args:
        image: PIL Image
        model: Loaded model
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    device = next(model.parameters()).device
    
    # Get transforms
    transform = get_transforms('test')
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get results
    pred_class = CLASS_NAMES[predicted.item()]
    conf_value = confidence.item() * 100
    all_probs = probabilities[0].cpu().numpy() * 100
    
    return pred_class, conf_value, all_probs


def predict_video_frames(video_path, model, skip_frames=5):
    """Predict water quality for video frames"""
    frames = extract_frames(video_path, skip_frames=skip_frames)
    
    if len(frames) == 0:
        return None
    
    predictions = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, frame in enumerate(frames):
        progress_bar.progress((i + 1) / len(frames))
        status_text.text(f"Processing frame {i+1}/{len(frames)}...")
        
        pil_image = frame_to_pil(frame)
        pred_class, confidence, _ = predict_image(pil_image, model)
        predictions.append((pred_class, confidence))
    
    progress_bar.empty()
    status_text.empty()
    
    return predictions


def create_video_summary(predictions):
    """Create summary statistics from predictions"""
    pred_classes = [p[0] for p in predictions]
    class_counts = Counter(pred_classes)
    total = len(predictions)
    
    summary = {
        'total_frames': total,
        'class_distribution': {},
        'average_confidences': {},
        'dominant_class': class_counts.most_common(1)[0][0] if class_counts else None
    }
    
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        summary['class_distribution'][class_name] = {
            'count': count,
            'percentage': percentage
        }
        
        class_confidences = [c for cls, c in predictions if cls == class_name]
        avg_conf = sum(class_confidences) / len(class_confidences) if class_confidences else 0
        summary['average_confidences'][class_name] = avg_conf
    
    return summary


def create_probability_chart(all_probs):
    """
    Create horizontal bar chart of class probabilities
    
    Args:
        all_probs: Array of probabilities for each class
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = [CLASS_COLORS[cls] for cls in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, all_probs, color=colors, alpha=0.7)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, all_probs)):
        ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_video_distribution_chart(summary):
    """Create horizontal bar chart for video class distribution"""
    fig, ax = plt.subplots(figsize=(10, 5))

    classes = CLASS_NAMES
    percentages = [summary['class_distribution'][cls]['percentage'] for cls in classes]
    colors = [CLASS_COLORS[cls] for cls in classes]

    # Horizontal bar chart
    bars = ax.barh(classes, percentages, color=colors, alpha=0.7)

    # Text on each bar (value labels)
    for bar, pct in zip(bars, percentages):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2., 
                f'{pct:.1f}%', 
                ha='left', va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Percentage of Frames (%)', fontsize=12, fontweight='bold')
    ax.set_title('Water Quality Distribution Across Video', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)  # x-axis limit instead of y-axis
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("💧 Water Quality Classifier")
    st.markdown("### Classify water quality as **Clean**, **Muddy**, or **Polluted** using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This application uses a **ResNet18** deep learning model 
            trained on water quality images to classify water into three categories:
            
            - 🟢 **Clean**: Safe, clear water
            - 🟠 **Muddy**: Turbid, sediment-laden water
            - 🔴 **Polluted**: Contaminated water
            
            **Accuracy**: 88.04% on test set
            """
        )
        
        st.header("Model Info")
        st.markdown(
            """
            - **Architecture**: ResNet18
            - **Training Images**: 365
            - **Test Images**: 92
            - **Classes**: 3
            - **Framework**: PyTorch
            """
        )
        
        st.header("Links")
        st.markdown(
            """
            - [GitHub Repository](https://github.com/Aasthayuli/water-quality-classifier-enhanced)
            - [Documentation](https://github.com/Aasthayuli/water-quality-classifier-enhanced/blob/main/README.md)
            """
        )
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Could not load model. Please check configuration.")
        st.stop()
    
    st.success("Model loaded successfully!")
    
    # Main content
    st.markdown("---")
    tab1, tab2 = st.tabs(["Image Classification", "Video Classification"])

    with tab1:           # IMAGE TAB 
        # File uploader
        st.header("Upload Water Image")
        uploaded_file = st.file_uploader(
            "Choose an image of water (JPG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of water for classification"
        )
    

        # Sample images option
        # use_sample = st.checkbox("Or use sample images")
    
        # if use_sample:
            # st.info("Sample images feature - Add your sample images to `app/samples/` folder")
            # You can add sample images here
    
        # Process uploaded image
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Predict button
                if st.button("Analyze Water Quality", type="primary", use_container_width=True):
                
                    with st.spinner("Analyzing image..."):
                        try:
                            # Predict
                            pred_class, confidence, all_probs = predict_image(image, model)
                        
                            # Display result
                            color = CLASS_COLORS[pred_class]
                        
                            st.markdown(
                            f"""
                            <div style="
                                background-color: {color}20;
                                border: 3px solid {color};
                                border-radius: 10px;
                                padding: 20px;
                                text-align: center;
                                margin: 20px 0;
                            ">
                                <h2 style="color: {color}; margin: 0;">
                                    {pred_class.upper()}
                                </h2>
                                <h3 style="margin: 10px 0;">
                                    Confidence: {confidence:.2f}%
                                </h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                            )
                        
                            # Interpretation
                            st.markdown("#### Interpretation:")
                            if pred_class == 'Clean':
                                st.success("The water appears clean and safe. Low turbidity and no visible contamination.")
                            elif pred_class == 'Muddy':
                                st.warning("The water shows high turbidity with suspended sediments. May require treatment.")
                            else:  # Polluted
                                st.error("The water appears contaminated. Not suitable for consumption without proper treatment.")
                        
                            # Probability chart
                            st.markdown("#### Classification Probabilities:")
                            fig = create_probability_chart(all_probs)
                            st.pyplot(fig)
                        
                            # Additional info
                            with st.expander("View Detailed Probabilities"):
                                for i, class_name in enumerate(CLASS_NAMES):
                                    st.metric(
                                    label=class_name,
                                    value=f"{all_probs[i]:.2f}%"
                                    )
                        
                        except Exception as e:
                            st.error(f"Prediction failed: {str(e)}")
        
        else:
            # Instructions
            st.info(
                """
                👆 **Get Started:**
                1. Upload an image of water using the file uploader above
                2. Click the "Analyze Water Quality" button
                3. View the classification results and confidence scores
                
                **Tips for best results:**
                - Use clear, well-lit images
                - Capture water from a consistent angle
                - Avoid images with excessive reflections or glare
                """
            )
        
            # Example results (placeholder)
            st.markdown("---")
            st.subheader("Example Results")
        
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🟢 Clean Water**")
                st.info("Clear, transparent water with no visible contamination")
            
            with col2:
                st.markdown("**🟠 Muddy Water**")
                st.warning("Turbid water with suspended sediments")
        
            with col3:
                st.markdown("**🔴 Polluted Water**")
                st.error("Contaminated water with visible pollution")
    
    # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: gray; padding: 20px;">
                <p>Made with ❤️ using PyTorch and Streamlit</p>
                <p>⚠️ <strong>Disclaimer:</strong> This is a demonstration project. 
                For actual water quality assessment, consult certified laboratories.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with tab2:           # VIDEO TAB 
        st.header("Upload Water Video")
    
        skip_frames = st.slider(
        "Process every Nth frame (higher = faster)",
        min_value=1, max_value=30, value=5
        )
    
        uploaded_video = st.file_uploader(
        "Choose a video of water to analyze",
        type=['mp4', 'avi', 'mov'],
        key="video_uploader"
        )
    
        if uploaded_video:
            col1, col2 = st.columns([1, 1])
        
            with col1:
                st.subheader("Uploaded Video")
                st.video(uploaded_video)
        
            with col2:
                st.subheader("Video Analysis")
            
                if st.button("Analyze Video", type="primary", use_container_width=True, key="video_predict"):
                
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_video.read())
                        video_path = tmp_file.name
                
                    try:
                        # Get video properties
                        props = get_video_properties(video_path)
                        st.info("Analyzing video ...")
                        st.info(f"{props['width']}x{props['height']} @ {props['fps']} FPS | {props['frames']} frames")
                    
                        # Predict frames
                        predictions = predict_video_frames(video_path, model, skip_frames)
                    
                        if predictions:
                            # Create summary
                            summary = create_video_summary(predictions)
                        
                            # Display dominant class
                            dominant = summary['dominant_class']
                            color = CLASS_COLORS[dominant]
                        
                            st.markdown(
                            f"""
                            <div style="background-color: {color}20; border: 3px solid {color}; 
                                        border-radius: 10px; padding: 20px; text-align: center; margin: 20px 0;">
                                <h2 style="color: {color}; margin: 0;">DOMINANT: {dominant.upper()}</h2>
                                <p style="margin: 10px 0;">Analyzed {summary['total_frames']} frames</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                            )
                        
                            # Interpretation
                            dist = summary['class_distribution'][dominant]
                            if dominant == 'Clean':
                                st.success(f"Water predominantly clean ({dist['percentage']:.1f}%)")
                            elif dominant == 'Muddy':
                                st.warning(f"Water shows turbidity ({dist['percentage']:.1f}%)")
                            else:
                                st.error(f"Water contaminated ({dist['percentage']:.1f}%)")
                        
                            # Show chart
                            st.markdown("#### Frame Distribution:")
                            fig = create_video_distribution_chart(summary)
                            st.pyplot(fig)
                        
                            # Detailed stats
                            with st.expander("Detailed Statistics"):
                                for class_name in CLASS_NAMES:
                                    d = summary['class_distribution'][class_name]
                                    avg = summary['average_confidences'][class_name]
                                    c1, c2, c3 = st.columns(3)
                                    c1.metric(f"{class_name} Frames", d['count'])
                                    c2.metric(f"{class_name} %", f"{d['percentage']:.1f}%")
                                    c3.metric("Avg Confidence", f"{avg:.1f}%")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                    finally:
                        if os.path.exists(video_path):
                            os.unlink(video_path)
        else:
            st.info("👆 Upload a video to analyze water quality")


if __name__ == "__main__":
    main()