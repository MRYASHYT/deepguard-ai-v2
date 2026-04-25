# 🛡️ Deepfake Detection System — Complete Guide

> **Diploma Project: AI-based Deepfake Detection**
> Built with EfficientNet-B4, MTCNN, Grad-CAM, and Streamlit

---

## YOUR 3-STEP PROCESS

You only need to do **3 things**. I have built everything else.

---

### STEP 1: Run Training on Google Colab (30 min setup + 2 hrs training)

1. Go to **https://colab.research.google.com**
2. Create a **New Notebook**
3. Click **Runtime → Change runtime type → T4 GPU → Save**
4. Upload the file `colab_full_pipeline.py` to Colab:
   - Click the **📁 folder icon** on the left sidebar
   - Click the **upload button** (⬆️)
   - Select `colab_full_pipeline.py` from your computer
5. In the first cell, type this and press **Shift+Enter**:

```
!python colab_full_pipeline.py
```

6. **Wait for it to finish** (~2-3 hours). It will:
   - Install all libraries automatically
   - Download the FaceForensics++ benchmark dataset
   - Detect and crop faces using MTCNN
   - Train the EfficientNet-B4 model (Phase 1 + Phase 2)
   - Evaluate accuracy, AUC, F1 score
   - Generate Grad-CAM heatmaps
   - Save everything

---

### STEP 2: Download the Trained Model

When `colab_full_pipeline.py` finishes, it will create these files:

| File | What it is |
|---|---|
| `best_model.pth` | Your trained AI model weights |
| `results.txt` | Accuracy, AUC, F1 scores |
| `gradcam_results/` | Heatmap images showing model focus areas |
| `confusion_matrix.png` | Visual confusion matrix |

**Download `best_model.pth` from Colab:**
- Click the **📁 folder icon** on the left
- Right-click `best_model.pth` → **Download**
- Place it inside your local `deepfake-detector/` folder

---

### STEP 3: Run the Streamlit Demo Locally

Open a terminal in your `deepfake-detector` folder and run:

```
pip install streamlit torch timm facenet-pytorch albumentations pytorch-grad-cam opencv-python scikit-learn
streamlit run app.py
```

This opens a browser with the **DeepGuard AI** dashboard where you can:
- Upload any image or video
- See REAL/FAKE prediction with confidence %
- View Grad-CAM heatmap showing WHERE the model detected manipulation

---

### STEP 4: Deploy to HuggingFace (So Your Teacher Can Open a Link)

This gives you a **public URL** like `https://huggingface.co/spaces/YOUR-USERNAME/deepguard-ai`
Your teacher just clicks the link — no installation needed on their side.

1. Go to **https://huggingface.co** → Sign up (free)
2. Click your **profile icon (top right)** → **New Space**
3. Fill in:
   - **Space name**: `deepguard-ai`
   - **SDK**: `Streamlit`
   - **Hardware**: `CPU basic` (free tier works fine for inference)
   - Click **Create Space**
4. HuggingFace will give you a git repo URL. Open a terminal and run:

```
cd deepfake-detector

git init
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/deepguard-ai

copy README_HF.md README.md
copy requirements_hf.txt requirements.txt

git add app.py requirements.txt README.md models/ preprocessing/ utils/ best_model.pth gradcam.py evaluate.py
git commit -m "Deploy DeepGuard AI"
git push hf main
```

5. Wait 2-3 minutes. Your app will be live at:
   **https://huggingface.co/spaces/YOUR-USERNAME/deepguard-ai**

6. Share that link with your teacher. Done! 🎉

> **Note**: Replace `YOUR-USERNAME` with your actual HuggingFace username.
> The `best_model.pth` file must be in the folder before you push.

---

## 📂 YOUR PROJECT FILES (All Built)

```
deepfake-detector/
├── colab_full_pipeline.py     ← Upload to Colab and run (does EVERYTHING)
├── download-FaceForensics.py  ← Official dataset download script
├── app.py                     ← Streamlit demo UI (run locally)
├── train.py                   ← Training loop (used inside pipeline)
├── evaluate.py                ← Evaluation metrics script
├── gradcam.py                 ← Grad-CAM heatmap generator
├── requirements.txt           ← All dependencies
├── README.md                  ← Project documentation
├── preprocessing/
│   ├── extract_frames.py      ← Video → frames
│   └── detect_faces.py        ← MTCNN face cropping
├── models/
│   ├── efficientnet.py        ← EfficientNet-B4 model
│   └── vit.py                 ← Vision Transformer model
├── utils/
│   └── dataset.py             ← Dataset loader + augmentations
└── faceforensics_official/    ← Official FF++ repo (reference)
```

---

## 📊 WHAT TO SHOW YOUR TEACHER

1. **The Streamlit App** — Live demo uploading images and getting predictions
2. **Grad-CAM Heatmaps** — Shows the model focuses on jawline/eyes/mouth
3. **Metrics** — Accuracy, AUC-ROC, F1 Score, Confusion Matrix
4. **Code Structure** — Clean folder layout with separate modules
5. **Research Awareness** — You used FaceForensics++ (the standard academic benchmark)

---

## 🧠 HOW IT WORKS (For Your Presentation)

1. **Face Detection**: MTCNN finds faces in the image/video
2. **Feature Extraction**: EfficientNet-B4 (pretrained on ImageNet) extracts visual features
3. **Classification**: A binary classifier outputs REAL (0) or FAKE (1)
4. **Explainability**: Grad-CAM shows which pixels influenced the decision
5. **Training Strategy**: Transfer Learning (freeze backbone → fine-tune)
6. **Dataset**: FaceForensics++ with 4 manipulation types (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
