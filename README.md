# 🧪 Malaria Cell Classification with Vision Transformers (ViT) and Streamlit

This project is part of the **Advanced Applied Mathematical Concepts for Deep Learning** course at George Brown College. It uses various deep learning models—including CNN, VGG16, and Vision Transformers—to classify red blood cell images as either **Parasitized** (infected with malaria) or **Uninfected**.

### 📌 Project Overview

The app uses a **Streamlit** frontend to showcase predictions from three trained ViT models:

1. **Full Fine-tuned ViT Model** (`vit_malaria_full.pt`)
2. **Head-only Transfer Learning ViT Model** (`vit_head-only-training_malaria.pth`)
3. **Baseline ViT Model** (`vit_malaria.pth`)

Each model takes in a microscopy cell image and predicts its malaria status. The app displays:
- The prediction (`Parasitized` or `Uninfected`)
- Confidence score
- Visual feedback through progress bars
- Processing timestamp

-------

### 📊 Dataset

**Source:** [Kaggle - Cell Images for Malaria Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

- ~27,558 labeled images
- 2 classes: `Parasitized` and `Uninfected`
- Images were preprocessed to 224x224 resolution
- Normalized to ImageNet stats (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`)

---

### 🧠 Models Used

#### 🧩 Vision Transformers (ViT)
- **Architecture:** `vit_tiny_patch16_224.augreg_in21k` from Hugging Face's `timm` library
- **Model 1:** Full fine-tuning (5.5M trainable params)
- **Model 2:** Head-only fine-tuning (386 trainable params)
- **Model 3:** Baseline fine-tuned model

#### 🧠 VGG16 (external Colab notebook)
- Transfer learning with added dense layers and dropout
- Achieved ~93% accuracy over 20 epochs

#### 🧠 CNN from Scratch
- Manually designed architecture (not included in this demo)
- Used for performance comparison

---

### 🚀 Getting Started

**Set up the environment:**
```bash
conda create -p venv python=3.12
conda activate ./venv
pip install -r requirements.txt


RUN THE APP:

streamlit run streamlit_app.py


File Structure:

.
├── streamlit_app.py                  # Main Streamlit UI
├── vit_malaria_full.pt               # Full fine-tuned ViT model
├── vit_head-only-training_malaria.pth # Head-only ViT model
├── vit_malaria.pth                   # Baseline ViT model
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview


👥 Authors (Group 7)
Abdul-Rasaq Omisesan 
Bikash Giri 
Gavriel Kirichenko 
Chia-Wei Chang 
Callum Arul 
Friba Hussainyar 
Diparshan Bhattarai 

🔗 References

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020, October 22). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv.org. https://arxiv.org/abs/2010.11929

Huggingface. (n.d.). GitHub - huggingface/pytorch-image-models: The largest collection of PyTorch image encoders / backbones. Including train, eval, inference, export scripts, and pretrained weights -- ResNet, ResNeXT, EfficientNet, NFNet, Vision Transformer (ViT), MobileNetV4, MobileNet-V3 & V2, RegNet, DPN, CSPNet, Swin Transformer, MaxViT, CoAtNet, ConvNeXt, and more. GitHub. https://github.com/huggingface/pytorch-image-models

Malaria cell Images Dataset. (2018, December 5). Kaggle. https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria


