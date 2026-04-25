import torch
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet import DeepfakeEfficientNet
from utils.dataset import DeepfakeDataset, get_val_transforms


def evaluate_model(model_path='best_model.pth', test_dir='data/test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DeepfakeEfficientNet(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_dataset = DeepfakeDataset(test_dir, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    accuracy = (all_preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*40}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*40}")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Real', 'Fake'])}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake'])
    ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black', fontsize=20)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()
