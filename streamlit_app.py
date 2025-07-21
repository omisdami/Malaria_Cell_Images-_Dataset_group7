import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import timm
import torch.nn.functional as F
from datetime import datetime
from timm.models.vision_transformer import VisionTransformer
from torch.serialization import add_safe_globals

# Allow full model unpickling for ViT
add_safe_globals({'timm.models.vision_transformer.VisionTransformer': VisionTransformer})

st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🧪",
    layout="centered"
)

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

@st.cache_resource
def load_model(model_name, path, is_full_model=False):
    if is_full_model:
        # For full torch.save(model) checkpoint
        model = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    else:
        # For saved state_dict
        model = timm.create_model(model_name, pretrained=False, num_classes=2)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    class_names = ['Parasitized', 'Uninfected']
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[predicted.item()], confidence.item()

def main():
    st.title("🧪 Malaria Cell Image Classification")

    st.markdown("""
    Upload a cell image below. This app uses **three ViT models** to classify whether a red blood cell is:
    - **Parasitized** (infected)
    - **Uninfected** (healthy)
    """)

    uploaded_file = st.file_uploader("📁 Upload a PNG, JPG, or JPEG image:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Classifying with multiple models..."):
            image_tensor = transform_image(image)

            model_1 = load_model('vit_tiny_patch16_224.augreg_in21k', 'vit_malaria_full.pth', is_full_model=True)
            model_2 = load_model('vit_tiny_patch16_224.augreg_in21k', 'vit_head-only-training_malaria.pth', is_full_model=False)
            model_3 = load_model('vit_tiny_patch16_224.augreg_in21k', 'vit_malaria.pth', is_full_model=False)

            label_1, conf_1 = predict(model_1, image_tensor)
            label_2, conf_2 = predict(model_2, image_tensor)
            label_3, conf_3 = predict(model_3, image_tensor)

        st.markdown("## 🔬 Model Results:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🧠 Model 1: VIT Full Fine-tuning")
            st.success(f"Prediction: `{label_1}`")
            st.progress(conf_1)
            st.markdown(f"Confidence: **{conf_1 * 100:.2f}%**")

        with col2:
            st.markdown("#### 🧠 Model 2: VIT Head-only Fine-tuning")
            st.success(f"Prediction: `{label_2}`")
            st.progress(conf_2)
            st.markdown(f"Confidence: **{conf_2 * 100:.2f}%**")

        with col3:
            st.markdown("#### 🧠 Model 3: Custom vit_malaria.pth")
            st.success(f"Prediction: `{label_3}`")
            st.progress(conf_3)
            st.markdown(f"Confidence: **{conf_3 * 100:.2f}%**")

        st.caption(f"🕒 Processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
