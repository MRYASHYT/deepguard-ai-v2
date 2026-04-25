import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet import DeepfakeEfficientNet
from utils.dataset import get_val_transforms


def generate_gradcam(image_path, model_path='best_model.pth', output_path='gradcam_output.png'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DeepfakeEfficientNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_np = np.array(img) / 255.0

    transform = get_val_transforms()
    input_tensor = transform(image=np.array(img))['image'].unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        prob = model(input_tensor).item()
    pred_label = "FAKE" if prob > 0.5 else "REAL"
    confidence = prob if prob > 0.5 else 1 - prob

    # Grad-CAM on the last conv layer
    target_layers = [model.backbone.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [BinaryClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM: {pred_label} ({confidence*100:.1f}%)", fontsize=14)
    axes[1].axis('off')

    plt.suptitle("Deepfake Detection - Explainability Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Grad-CAM saved to {output_path}")
    plt.close()

    return pred_label, confidence, visualization


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Path to image')
    parser.add_argument('--model', type=str, default='best_model.pth')
    parser.add_argument('--output', type=str, default='gradcam_output.png')
    args = parser.parse_args()
    generate_gradcam(args.image, args.model, args.output)
