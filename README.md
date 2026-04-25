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

## 🧠 The Science Behind It (For Students & Professors)

DeepGuard AI is built on three core pillars of modern deep learning. Here is how they work:

### 1. Detection: MTCNN (Multi-task Cascaded Convolutional Networks)
* **For the Student:** Before you can tell if a face is fake, you have to find the face! MTCNN is like a highly trained scout that scans the whole photo, ignores the background (like trees or rooms), and draws a perfect square right around the person's face.
* **For the Professor:** MTCNN is a robust three-stage cascaded CNN that performs face detection and bounding box regression simultaneously. We use it as a preprocessing step to isolate the facial region of interest (ROI) with a 20-pixel margin, ensuring the downstream classifier isn't influenced by irrelevant background noise.

### 2. The Model: EfficientNet-B4
* **For the Student:** This is the "brain" of the operation. EfficientNet has analyzed millions of images. It looks at the cropped face and checks for microscopic mistakes—like blurred skin boundaries or unnatural lighting—that humans can't see, but AI generators often mess up.
* **For the Professor:** EfficientNet-B4 is our primary feature-extraction backbone. It utilizes a compound scaling method that uniformly scales network width, depth, and resolution. B4 was chosen for its optimal balance between high forensic accuracy and computational efficiency. It was fine-tuned using a Two-Phase Transfer Learning approach with a custom `Dropout(p=0.4) -> Linear -> Sigmoid` classification head using Binary Cross Entropy (BCE) Loss.

### 3. Explainability: Grad-CAM
* **For the Student:** If the AI says a picture is fake, we want to know *why*. Grad-CAM is like a heat-vision camera. It creates a colorful map over the face, glowing bright red over the exact spots (like a glitchy eyeball or a poorly blended chin) that proved the image was deepfaked.
* **For the Professor:** Neural networks are often criticized as "black boxes." Grad-CAM (Gradient-weighted Class Activation Mapping) solves this by using the gradients of the target concept flowing into the final convolutional layer (`conv_head`) to produce a localization map. This provides critical visual transparency, allowing forensic analysts to verify that the model is detecting genuine synthetic artifacts (e.g., blending boundaries) rather than overfitting to spurious correlations.

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
