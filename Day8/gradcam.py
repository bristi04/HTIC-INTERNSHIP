import streamlit as st
import torch
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
import numpy as np
import cv2
from PIL import Image

model = resnet18(pretrained=True)
model.eval()

target_layers = [model.layer4[-1]]

st.title("🔍 Grad-CAM Visualizer (PyTorch)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    rgb_img = np.float32(image) / 255.0

    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    st.image(visualization, caption="Grad-CAM Heatmap", use_column_width=True)

