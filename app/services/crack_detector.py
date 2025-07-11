import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2

# Initialize U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crack_model = smp.Unet(
    encoder_name="resnet34",        # Use ResNet-34 encoder
    encoder_weights="imagenet",     # Pretrained on ImageNet
    in_channels=3,                  # RGB images
    classes=1,                      # Binary segmentation (crack/no-crack)
)
crack_model.to(device)
crack_model.eval()

def preprocess_frame(frame, size=(256, 256)):
    """
    Resize and normalize frame for U-Net
    """
    resized = cv2.resize(frame, size)
    normalized = resized / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)

def predict_crack(frame):
    """
    Dummy crack detector: detects 'cracks' by looking for thin dark lines
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Assume thin edges are cracks
    crack_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    return crack_mask