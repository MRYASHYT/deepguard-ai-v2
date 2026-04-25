# 🛡️ DeepGuard AI — Neural Forensics Engine

**DeepGuard AI** is a professional-grade deepfake detection system built for real-time analysis of synthetic media. It uses advanced computer vision to extract facial features and analyze pixel-level inconsistencies to determine if an image or video is real or manipulated.

---

## 👶 How it Works (The Simple Version)
Imagine you have a super-smart detective robot with a magnifying glass. 
1. First, the robot looks at a picture and finds exactly where the person's face is.
2. Then, it uses its magnifying glass to look incredibly closely at the face—so close that it can see things humans can't, like tiny blurry spots around the edges of the mouth or weird lighting in the eyes.
3. Finally, the robot compares what it sees to millions of other real and fake pictures it has studied in the past. If it sees those weird, unnatural spots, it flags the picture as a "Deepfake"!
4. It even draws a heatmap (a colorful map) to show you exactly *where* it thinks the image looks suspicious.

---

## 🎓 Technical Architecture (For the Professor)
DeepGuard AI is an end-to-end PyTorch-based classification pipeline designed for high-accuracy synthetic media detection.

### 1. Data Processing Pipeline
*   **Face Extraction:** Uses **MTCNN (Multi-task Cascaded Convolutional Networks)** to reliably locate and crop facial bounding boxes with a 20-pixel margin. This isolates the region of interest and removes background noise.
*   **Transformations:** Images are resized to `224x224` and normalized using standard ImageNet parameters `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` via the `Albumentations` library.

### 2. Model Architecture
*   **Backbone:** Uses **EfficientNet-B4** via the `timm` library. EfficientNet was chosen for its optimal balance of high accuracy and computational efficiency (achieved via compound scaling of depth, width, and resolution).
*   **Classification Head:** The network terminates in a custom binary classification head featuring a `Dropout(p=0.4)` layer for regularization, followed by a `Linear` layer mapping to a single output, passed through a `Sigmoid` activation function to yield a probability score `P(Fake)`.

### 3. Training Methodology
*   **Dataset:** Trained on a curated subset of the **Kaggle 'Real and Fake Face Detection'** dataset, specifically targeted at facial forgery artifacts.
*   **Two-Phase Transfer Learning:**
    1.  **Phase 1 (Feature Extraction):** The EfficientNet backbone weights were frozen, and only the classification head was trained using an `AdamW` optimizer (`lr=1e-3`) for rapid convergence on the target domain.
    2.  **Phase 2 (Fine-Tuning):** The entire network was unfrozen and trained with a highly reduced learning rate (`lr=1e-5`) to fine-tune the deep convolutional filters to recognize subtle spectral and blending inconsistencies characteristic of GANs and autoencoder-based face swaps.
*   **Loss Function:** Binary Cross Entropy (BCE) Loss.

### 4. Explainability (XAI)
*   The system implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** attached to the final convolutional layer (`conv_head`) of the EfficientNet backbone. This provides visual interpretability by highlighting the specific spatial regions (e.g., jawline blending, eye specular highlights) that maximally activated the model's decision function.

---

## 🚀 How to Run the App

This project is built using **Streamlit** for the frontend dashboard.

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the Server:**
```bash
python -m streamlit run app.py
```

**3. Access the System:**
*   Open your browser to `http://localhost:8501`
*   **Default Credentials:** 
    *   Username: `admin`
    *   Password: `deepguard`

---
*Developed for a Diploma Research Project focusing on Synthetic Media Detection and Neural Forensics.*
