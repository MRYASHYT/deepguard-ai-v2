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

### 1. Dataset Selection
The model was trained on a high-fidelity composite dataset to ensure maximum detection coverage:
*   **Kaggle Deepfake Detection Challenge (DFDC)**: Our primary source, containing over 100,000 video clips with diverse ethnicities, lighting, and environments.
*   **FaceForensics++ (FF++)**: Re-integrated to provide specialized detection of specific manipulation techniques including FaceSwap, Face2Face, and NeuralTextures.
*   **Composite Realism**: By combining these two datasets, the model is trained to recognize both seamless GAN-based blending and traditional computer-vision-based swapping artifacts.

### 2. Model Architecture Rationale
We selected **EfficientNet-B4** as our primary classification backbone for the following reasons:
*   **Compound Scaling**: Unlike other models that just get deeper or wider, EfficientNet scales width, depth, and resolution simultaneously. This allows it to capture fine-grained pixel artifacts (essential for deepfakes) while remaining efficient enough to run on a standard GPU.
*   **Feature Sensitivity**: Deepfakes often leave "checkerboard artifacts" or spectral inconsistencies. B4’s architecture is particularly sensitive to these high-frequency signals compared to older models like ResNet.

### 3. Training Protocol
The training was conducted in two distinct phases:
*   **Phase 1 (Feature Alignment)**: The model was initialized with ImageNet weights. The final layers were unfrozen and trained on a combined subset of the **Kagage DFDC** and **FaceForensics++** datasets to establish a baseline for identifying varied synthetic artifacts.
*   **Phase 2 (Hardening)**: The entire network was unfrozen and trained at a lower learning rate (`1e-5`) on the full multi-source dataset. This "hardened" the model against different lighting, compression, and resolutions.
*   **Augmentations**: To make the model robust, we applied "Forensic Augmentations" during training, including Gaussian Blur, ISO Noise, and JPEG compression. This teaches the model to see through common video "smudging" used to hide fakes.

### 4. The Forensic Pipeline
When you upload a file, the system executes the following chain:
1.  **Extraction**: MTCNN (Multi-task Cascaded Convolutional Networks) scans the frame and extracts a high-resolution face crop.
2.  **Inference**: The EfficientNet-B4 brain calculates a "Forgery Score" between 0.0 and 1.0.
3.  **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) analyzes the gradients of the final convolution layer. It identifies exactly which pixels triggered the detection and highlights them as a heatmap.

---

## 🧠 The Science Behind It (For Students & Professors)

### 1. Detection: MTCNN (Multi-task Cascaded Convolutional Networks)
*   **For the Student**: Before you can tell if a face is fake, you have to find the face! MTCNN is like a highly trained scout that scans the whole photo, ignores the background, and draws a perfect square right around the person's face.
*   **For the Professor**: MTCNN is a robust three-stage cascaded CNN that performs face detection and bounding box regression simultaneously. We use it as a preprocessing step to isolate the facial region of interest (ROI) with a 20-pixel margin, ensuring the downstream classifier isn't influenced by irrelevant background noise.

### 2. The Model: EfficientNet-B4
*   **For the Student**: This is the "brain" of the operation. EfficientNet has analyzed millions of images. It looks at the cropped face and checks for microscopic mistakes—like blurred skin boundaries or unnatural lighting—that humans can't see, but AI generators often mess up.
*   **For the Professor**: EfficientNet-B4 is our primary feature-extraction backbone. It utilizes a compound scaling method that uniformly scales network width, depth, and resolution. B4 was chosen for its optimal balance between forensic accuracy and computational efficiency. It was fine-tuned using a Two-Phase Transfer Learning approach with a custom classification head using Binary Cross Entropy (BCE) Loss.

### 3. Explainability: Grad-CAM
*   **For the Student**: If the AI says a picture is fake, we want to know *why*. Grad-CAM is like a heat-vision camera. It creates a colorful map over the face, glowing bright red over the exact spots (like a glitchy eyeball or a poorly blended chin) that proved the image was deepfaked.
*   **For the Professor**: Neural networks are often criticized as "black boxes." Grad-CAM (Gradient-weighted Class Activation Mapping) solves this by using the gradients of the target concept flowing into the final convolutional layer to produce a localization map. This provides critical visual transparency, allowing forensic analysts to verify that the model is detecting genuine synthetic artifacts rather than overfitting to spurious correlations.

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

## 🎓 Academic Defense: Proving the AI Training

When presenting this project to a professor or technical lead, use the following "Proof Points" to demonstrate that the system is a trained Neural Forensic engine and not a heuristic detection tool.

### 1. The Weight Artifact (`best_model.pth`)
*   **The Evidence**: The model's intelligence is stored in the `best_model.pth` file.
*   **The Rationale**: This file contains approximately **19 million learnable parameters** optimized through **Backpropagation**. These weights represent the "learned" patterns of synthetic skin and blending artifacts from the Kaggle DFDC and FaceForensics++ datasets.

### 2. Grad-CAM Explainability (XAI)
*   **The Evidence**: The visual Heatmap produced after every image analysis.
*   **The Rationale**: We use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency. By calculating the gradients of the target class (Fake) flowing into the final convolutional layer, we can mathematically prove the model is focusing on **manipulation artifacts** (edges, mouth boundaries, eyes) rather than random background pixels.

### 3. Architecture Selection: EfficientNet-B4
*   **The Evidence**: The use of a state-of-the-art CNN backbone.
*   **The Rationale**: Unlike simpler models, **EfficientNet-B4** uses **Compound Scaling** to balance depth, width, and resolution. This makes it specifically sensitive to high-frequency pixel inconsistencies that occur during GAN-based or FaceSwap-based generation processes.

### 4. Forensic Augmentation Protocol
*   **The Evidence**: Training-time noise and compression resilience.
*   **The Rationale**: During training, we applied **Gaussian Noise, ISO Noise, and JPEG compression** to the dataset. This forced the model to learn "Inference-invariant" features, allowing it to detect fakes even when the source media has been degraded or compressed—a key challenge in real-world forensics.

### 5. Training Metrics & Convergence
*   **The Evidence**: Use of Binary Cross Entropy (BCE) Loss.
*   **The Rationale**: The model was optimized using **BCE Loss** and validated using **AUC (Area Under Curve)** metrics on Kaggle's DFDC competition subset. We utilized a **Learning Rate Scheduler** (ReduceLROnPlateau) to ensure stable convergence during the multi-phase training protocol.

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
