import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import timm
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model as keras_load_model

# Page config
st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🧪",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .reportview-container {
            padding: 2rem;
        }
        h1, h3 {
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #4B6EA9;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #4B6EA9;
        }
    </style>
""", unsafe_allow_html=True)

# Load PyTorch ViT model
@st.cache_resource
def load_model(model_name, path, is_full_model=False):
    if is_full_model:
        model = torch.load(path, map_location=torch.device('cpu'))
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=2)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model.eval()
    return model

# Transform for Torch and Keras
def transform_image(image):
    # PyTorch (ViT)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    torch_img = transform(image).unsqueeze(0)

    # Keras CNN (trained on 224x224)
    keras_img_cnn = image.resize((224, 224))
    keras_img_cnn = np.array(keras_img_cnn) / 255.0
    keras_img_cnn = np.expand_dims(keras_img_cnn, axis=0)

    # Keras VGG (trained on 130x130)
    keras_img_vgg = image.resize((130, 130))
    keras_img_vgg = np.array(keras_img_vgg) / 255.0
    keras_img_vgg = np.expand_dims(keras_img_vgg, axis=0)

    return torch_img, keras_img_cnn, keras_img_vgg

# Predict using Torch ViT
def predict_torch(model, image_tensor):
    class_names = ['Parasitized', 'Uninfected']
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[predicted.item()], confidence.item()

# Predict using Keras CNN/VGG16
def predict_keras(model, keras_img):
    pred = model.predict(keras_img)
    confidence = float(np.max(pred))
    label = 'Parasitized' if np.argmax(pred) == 0 else 'Uninfected'
    return label, confidence

def main():
    st.title("🧪 Malaria Cell Image Classification")

    st.markdown("""
    Upload a cell image below. This app uses **five different models** to classify whether a red blood cell is:
    - **Parasitized** (infected)
    - **Uninfected** (healthy)
    """)

    uploaded_file = st.file_uploader("📁 Upload a PNG, JPG, or JPEG image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Classifying with 5 models..."):
            torch_img, keras_img_cnn, keras_img_vgg = transform_image(image)

            # Load models
            vit_full_model = load_model('vit_tiny_patch16_224.augreg_in21k', 'vit_malaria_full.pth', is_full_model=True)
            vit_head_model = load_model('vit_tiny_patch16_224.augreg_in21k', 'vit_head-only-training_malaria.pth')
            cnn_model = keras_load_model('best_model.h5', compile=False)
            vgg_model = keras_load_model('best_vgg16_model.h5')

            # Predictions
            label_vit_full, conf_vit_full = predict_torch(vit_full_model, torch_img)
            label_vit_head, conf_vit_head = predict_torch(vit_head_model, torch_img)
            label_cnn, conf_cnn = predict_keras(cnn_model, keras_img_cnn)
            label_vgg, conf_vgg = predict_keras(vgg_model, keras_img_vgg)

        # Show results
        st.markdown("## 🔬 Model Results:")

        st.subheader("🧠 Vision Transformer (ViT) Models")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Full Fine-tuning**")
            st.success(f"Prediction: {label_vit_full}")
            st.progress(conf_vit_full)
            st.markdown(f"Confidence: **{conf_vit_full * 100:.2f}%**")

        with col2:
            st.markdown("**Head-only Fine-tuning**")
            st.success(f"Prediction: {label_vit_head}")
            st.progress(conf_vit_head)
            st.markdown(f"Confidence: **{conf_vit_head * 100:.2f}%**")

        st.subheader("🧠 CNN-Based Models")
        col4, col5 = st.columns(2)

        with col4:
            st.markdown("**CNN from Scratch**")
            st.success(f"Prediction: {label_cnn}")
            st.progress(conf_cnn)
            st.markdown(f"Confidence: **{conf_cnn * 100:.2f}%**")

        with col5:
            st.markdown("**VGG16 Transfer Learning**")
            st.success(f"Prediction: {label_vgg}")
            st.progress(conf_vgg)
            st.markdown(f"Confidence: **{conf_vgg * 100:.2f}%**")

        st.caption(f"🕒 Processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
