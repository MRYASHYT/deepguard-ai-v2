# 🛡️ DeepGuard AI — Neural Forensics Engine

**DeepGuard AI** is a professional-grade deepfake detection system built for real-time analysis of synthetic media. It uses advanced computer vision to extract facial features and analyze pixel-level inconsistencies to determine if an image or video is real or manipulated.

---

## 👶 How it Works (The Simple Version)
Imagine you have a super-smart detective robot with a magnifying glass. 
1. First, the robot looks at a picture and finds exactly where the person's face is.
2. Then, it uses its magnifying glass to look incredibly closely at the face—so close that it can see things humans can't, like tiny blurry spots around the edges of the mouth or weird lighting in the eyes.
3. Finally, it compares what it sees to millions of other real and fake pictures it has studied in the past (Kaggle DFDC). If it sees those weird, unnatural spots, it flags it as a "Deepfake"!
4. It even draws a heatmap to show you exactly *where* it thinks the image looks suspicious.

---

## 🛠️ The Engineering Process (Technical Deep Dive)

Creating a forensic-grade detection model requires a rigorous multi-stage pipeline. Below is the full process used to build the DeepGuard engine.

### 1. Dataset Selection (Kaggle DFDC)
The model was trained exclusively on the **Deepfake Detection Challenge (DFDC)** dataset, sourced from **Kaggle**. This is currently the most significant and diverse dataset in the neural forensics field:
*   **Scale**: Over 100,000 video clips generated using multiple deepfake techniques.
*   **Diversity**: Includes a wide range of subjects, ethnicities, lighting conditions, and background environments to ensure the model generalizes to real-world scenarios.
*   **Realism**: The dataset includes diverse audio and video augmentations (compression, noise, blur) to simulate the quality of media typically found on social platforms.

### 2. Model Architecture Rationale
We selected **EfficientNet-B4** as our primary classification backbone for the following reasons:
*   **Compound Scaling**: Unlike other models that just get deeper or wider, EfficientNet scales width, depth, and resolution simultaneously. This allows it to capture fine-grained pixel artifacts (essential for deepfakes) while remaining efficient enough to run on a standard GPU.
*   **Feature Sensitivity**: Deepfakes often leave "checkerboard artifacts" or spectral inconsistencies. B4’s architecture is particularly sensitive to these high-frequency signals compared to older models like ResNet.

### 3. Training Protocol
The training was conducted in two distinct phases:
*   **Phase 1 (Feature Alignment)**: The model was initialized with ImageNet weights. The final layers were unfrozen and trained on a clean subset of the **Kaggle DFDC** dataset to establish a baseline for facial authenticity.
*   **Phase 2 (Hardening)**: The entire network was unfrozen and trained at a lower learning rate (`1e-5`) on the full **DFDC** dataset. This "hardened" the model against different lighting, compression, and resolutions.
*   **Augmentations**: To make the model robust, we applied "Forensic Augmentations" during training, including Gaussian Blur, ISO Noise, and JPEG compression. This teaches the model to see through common video "smudging" used to hide fakes.

### 4. The Forensic Pipeline
When you upload a file, the system executes the following chain:
1.  **Extraction**: MTCNN (Multi-task Cascaded Convolutional Networks) scans the frame and extracts a high-resolution face crop.
2.  **Inference**: The EfficientNet-B4 brain calculates a "Forgery Score" between 0.0 and 1.0.
3.  **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) analyzes the gradients of the final convolution layer. It identifies exactly which pixels triggered the detection and highlights them as a heatmap.

---

## 🧠 The Science Components

### 1. MTCNN (Face Detection)
MTCNN is a robust three-stage cascaded CNN that performs face detection and bounding box regression simultaneously. We use it to isolate the facial region of interest (ROI) with a 20-pixel margin, ensuring the downstream classifier isn't influenced by irrelevant background noise.

### 2. Explainability: Grad-CAM
Neural networks are often criticized as "black boxes." Grad-CAM solves this by using the gradients flowing into the final convolutional layer (`conv_head`) to produce a localization map. This allows forensic analysts to verify that the model is detecting genuine synthetic artifacts (e.g., blending boundaries) rather than overfitting to spurious correlations.

---

## 🎯 Model Capabilities & Limitations

### 1. What it Detects: Human Faces Only
The engine operates strictly on a **Facial Deepfake Detection** paradigm.
*   The system relies entirely on the MTCNN face detector to locate a region of interest. 
*   **Conclusion:** The model can **ONLY** analyze images and videos that contain a human face. It cannot analyze landscapes, AI-generated voices, or text.

### 2. Can it detect AI-Generated Cartoons or Anime?
**No.**
*   MTCNN is trained on real human faces. It will fail to detect a cartoon, anime, or animal face.
*   Even if a cartoon face is forced through the pipeline, the classifier will produce arbitrary scores because its training dataset consisted of manipulated human skin textures, not drawn or animated pixels.

### 3. Known Limitations
*   **No Face = No Detection**: If a subject's face is obscured or turned away, MTCNN will fail, halting the analysis.
*   **High Compression**: Pixel-level artifacts can be "washed out" by heavy compression (e.g., WhatsApp videos), potentially leading to False Negatives.
*   **100% Synthetic Generation**: The model is optimized to detect **facial manipulations** (blending, warping, swapping). If a video is generated 100% from scratch (like Sora or Midjourney), it may not contain the specific "blending boundaries" the model is trained to find.

---

## 🚀 How to Run the App

This project is built using **Streamlit** for the forensic dashboard.

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
*   The system initializes with a **Forensic Skeleton Loader** before providing full access to the detection engine.

---
*Developed as a high-fidelity Neural Forensics project focusing on Synthetic Media Detection and Academic Explainability.*
