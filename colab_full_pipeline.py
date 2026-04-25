"""
============================================================
  DEEPFAKE DETECTION SYSTEM - FULL PIPELINE (v4 - Kaggle)
  
  COLAB INSTRUCTIONS:
  
  CELL 1 - Install + setup Kaggle:
    !pip install -U Pillow timm facenet-pytorch albumentations grad-cam kaggle
  
  Then RESTART RUNTIME (click the button Colab shows).
  
  CELL 2 - Upload kaggle.json then run:
    import os
    os.makedirs('/root/.kaggle', exist_ok=True)
    !cp kaggle.json /root/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json
    !python colab_full_pipeline.py
============================================================
"""

import os
import sys
import subprocess
import random

# ============================================================
# STEP 1: DOWNLOAD KAGGLE DATASET
# ============================================================
print("=" * 60)
print("STEP 1/6: Downloading Kaggle dataset...")
print("=" * 60)

DATASET = "ciplab/real-and-fake-face-detection"

if not os.path.exists("real_and_fake_face_detection.zip"):
    print(f"Downloading {DATASET} from Kaggle...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", DATASET
    ])
    print("Download complete.")
else:
    print("Dataset already downloaded.")

# Extract
import zipfile

if not os.path.exists("real_and_fake_face"):
    print("Extracting...")
    with zipfile.ZipFile("real_and_fake_face_detection.zip", 'r') as z:
        z.extractall(".")
    print("Extracted.")
else:
    print("Already extracted.")

# Show what we got
print("\nDataset structure:")
for root, dirs, files in os.walk("."):
    # Only show relevant dirs
    if "real_and_fake" in root or root == ".":
        level = root.count(os.sep)
        if level < 4:
            img_count = sum(1 for f in files if f.lower().endswith(('.jpg','.png','.jpeg')))
            if img_count > 0 or dirs:
                print(f"  {root}/ -> {img_count} images, subdirs: {dirs[:5]}")


# ============================================================
# STEP 2: ORGANIZE INTO TRAIN/VAL/TEST
# ============================================================
print("\n" + "=" * 60)
print("STEP 2/6: Organizing into train/val/test splits...")
print("=" * 60)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Create output dirs
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        os.makedirs(f"data/{split}/{label}", exist_ok=True)

# Find the real and fake image directories
real_source = None
fake_source = None

for root, dirs, files in os.walk("."):
    lower = root.lower()
    imgs = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(imgs) > 10:
        if "real" in lower and "fake" not in lower:
            real_source = root
            print(f"Found REAL folder: {root} ({len(imgs)} images)")
        elif "fake" in lower:
            fake_source = root
            print(f"Found FAKE folder: {root} ({len(imgs)} images)")

if real_source is None or fake_source is None:
    print("ERROR: Could not find real/fake folders!")
    print("Listing all directories with images:")
    for root, dirs, files in os.walk("."):
        imgs = [f for f in files if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if len(imgs) > 0:
            print(f"  {root}: {len(imgs)} images")
    sys.exit(1)

# Collect image paths
real_images = sorted([os.path.join(real_source, f) for f in os.listdir(real_source)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
fake_images = sorted([os.path.join(fake_source, f) for f in os.listdir(fake_source)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

print(f"\nTotal: {len(real_images)} real, {len(fake_images)} fake")

# Shuffle
random.seed(42)
random.shuffle(real_images)
random.shuffle(fake_images)

# Copy to train/val/test with face detection
from facenet_pytorch import MTCNN

mtcnn = MTCNN(margin=20, keep_all=False, post_process=False,
              image_size=224, device=device)

def split_and_save(image_list, label, max_per_split=5000):
    """Split images into train/val/test and save face crops."""
    n = min(len(image_list), max_per_split * 2)  # Cap total for speed
    image_list = image_list[:n]

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    saved = {"train": 0, "val": 0, "test": 0}

    for i, img_path in enumerate(tqdm(image_list, desc=f"Processing {label}")):
        if i < train_end:
            split = "train"
        elif i < val_end:
            split = "val"
        else:
            split = "test"

        out_path = f"data/{split}/{label}/img_{saved[split]:05d}.jpg"

        try:
            img = Image.open(img_path).convert('RGB')

            # Try MTCNN face crop
            face = mtcnn(img, save_path=out_path)

            if face is None:
                # No face detected, resize full image
                img.resize((224, 224)).save(out_path)

            saved[split] += 1
        except Exception:
            continue

    return saved

real_saved = split_and_save(real_images, "real")
fake_saved = split_and_save(fake_images, "fake")

print(f"\nDataset prepared:")
for s in ["train", "val", "test"]:
    r = len([f for f in os.listdir(f"data/{s}/real") if f.endswith('.jpg')])
    fk = len([f for f in os.listdir(f"data/{s}/fake") if f.endswith('.jpg')])
    print(f"  {s}: {r} real, {fk} fake")


# ============================================================
# STEP 3: DEFINE MODEL + DATA LOADERS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3/6: Building EfficientNet-B4 model...")
print("=" * 60)

import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for lbl_name, lbl_val in [('real', 0), ('fake', 1)]:
            folder = os.path.join(root_dir, lbl_name)
            if os.path.exists(folder):
                for img in os.listdir(folder):
                    if img.endswith(('.jpg', '.png')):
                        self.image_paths.append(os.path.join(folder, img))
                        self.labels.append(lbl_val)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

train_tf = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        in_f = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.backbone(x)

train_ds = DeepfakeDataset('data/train', transform=train_tf)
val_ds = DeepfakeDataset('data/val', transform=val_tf)
test_ds = DeepfakeDataset('data/test', transform=val_tf)

bs = 16
train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)
test_ld = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)

print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

model = DeepfakeDetector().to(device)
criterion = nn.BCELoss()


# ============================================================
# STEP 4: TRAIN
# ============================================================
print("\n" + "=" * 60)
print("STEP 4/6: Training...")
print("=" * 60)

# Phase 1: Head only
print("\n--- Phase 1: Classifier head (5 epochs) ---")
for p in model.backbone.parameters(): p.requires_grad = False
for p in model.backbone.classifier.parameters(): p.requires_grad = True
opt = optim.Adam(model.backbone.classifier.parameters(), lr=1e-3)

for ep in range(5):
    model.train()
    ls, cor, tot = 0, 0, 0
    for imgs, lbls in tqdm(train_ld, desc=f"P1 E{ep+1}"):
        imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
        opt.zero_grad()
        out = model(imgs); loss = criterion(out, lbls)
        loss.backward(); opt.step()
        ls += loss.item()
        cor += ((out>0.5).float()==lbls).sum().item()
        tot += lbls.size(0)
    print(f"  E{ep+1}: Loss={ls/len(train_ld):.4f} Acc={cor/tot:.4f}")

# Phase 2: Full
print("\n--- Phase 2: Full fine-tune (10 epochs) ---")
for p in model.parameters(): p.requires_grad = True
opt = optim.Adam(model.parameters(), lr=1e-5)
best_val = 0

for ep in range(10):
    model.train()
    ls, cor, tot = 0, 0, 0
    for imgs, lbls in tqdm(train_ld, desc=f"P2 E{ep+1}"):
        imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
        opt.zero_grad()
        out = model(imgs); loss = criterion(out, lbls)
        loss.backward(); opt.step()
        ls += loss.item()
        cor += ((out>0.5).float()==lbls).sum().item()
        tot += lbls.size(0)
    t_acc = cor/tot

    model.eval()
    vc, vt = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_ld:
            imgs, lbls = imgs.to(device), lbls.to(device).unsqueeze(1)
            out = model(imgs)
            vc += ((out>0.5).float()==lbls).sum().item()
            vt += lbls.size(0)
    v_acc = vc/vt if vt>0 else 0
    print(f"  E{ep+1}: Loss={ls/len(train_ld):.4f} Train={t_acc:.4f} Val={v_acc:.4f}")
    if v_acc >= best_val:
        best_val = v_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  >> Saved best model")

if not os.path.exists('best_model.pth'):
    torch.save(model.state_dict(), 'best_model.pth')


# ============================================================
# STEP 5: EVALUATE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5/6: Evaluating on test set...")
print("=" * 60)

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

prds, lbls_all, prbs = [], [], []
with torch.no_grad():
    for imgs, lbls in tqdm(test_ld, desc="Testing"):
        out = model(imgs.to(device))
        prbs.extend(out.cpu().numpy().flatten())
        prds.extend((out>0.5).float().cpu().numpy().flatten())
        lbls_all.extend(lbls.numpy())

prds, lbls_all, prbs = np.array(prds), np.array(lbls_all), np.array(prbs)
acc = (prds==lbls_all).mean()
try: auc = roc_auc_score(lbls_all, prbs)
except: auc = 0
f1 = f1_score(lbls_all, prds, zero_division=0)
cm = confusion_matrix(lbls_all, prds)

print(f"\n{'='*40}")
print(f"  RESULTS")
print(f"{'='*40}")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"  AUC-ROC:  {auc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Confusion Matrix:\n  {cm}")

with open("results.txt","w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\nAUC-ROC: {auc:.4f}\nF1: {f1:.4f}\n")

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize=(6,5))
ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Real','Fake']); ax.set_yticklabels(['Real','Fake'])
for i in range(2):
    for j in range(2):
        ax.text(j,i,str(cm[i][j]),ha='center',va='center',fontsize=20,
                color='white' if cm[i][j]>cm.max()/2 else 'black')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
plt.tight_layout(); plt.savefig('confusion_matrix.png',dpi=150); plt.close()


# ============================================================
# STEP 6: GRAD-CAM
# ============================================================
print("\n" + "=" * 60)
print("STEP 6/6: Grad-CAM heatmaps...")
print("=" * 60)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

    os.makedirs("gradcam_results", exist_ok=True)
    cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])

    samples = []
    for lbl in ["real","fake"]:
        d = f"data/test/{lbl}"
        if os.path.exists(d):
            fs = [f for f in os.listdir(d) if f.endswith('.jpg')][:5]
            samples.extend([(os.path.join(d,f),lbl) for f in fs])

    for idx,(p,tl) in enumerate(samples):
        try:
            img = Image.open(p).convert('RGB').resize((224,224))
            img_np = np.array(img)/255.0
            t = val_tf(image=np.array(img))['image'].unsqueeze(0).to(device)
            with torch.no_grad(): prob = model(t).item()
            pred = "FAKE" if prob>0.5 else "REAL"
            gc = cam(input_tensor=t, targets=[BinaryClassifierOutputTarget(1)])[0]
            vis = show_cam_on_image(img_np.astype(np.float32),gc,use_rgb=True)
            fig,(a1,a2) = plt.subplots(1,2,figsize=(10,4))
            a1.imshow(img); a1.set_title(f"Original ({tl})"); a1.axis('off')
            a2.imshow(vis); a2.set_title(f"GradCAM: {pred} ({prob*100:.1f}%)"); a2.axis('off')
            plt.tight_layout()
            plt.savefig(f"gradcam_results/cam_{idx:02d}_{tl}.png",dpi=150); plt.close()
        except: continue
    print(f"Heatmaps saved to gradcam_results/")
except ImportError:
    print("Grad-CAM not available, skipping.")

print("\n" + "=" * 60)
print("  PROJECT COMPLETE!")
print("=" * 60)
print("  best_model.pth       <- DOWNLOAD THIS")
print("  results.txt          <- Metrics")
print("  confusion_matrix.png <- Plot")
print("  gradcam_results/     <- Heatmaps")
print("=" * 60)
