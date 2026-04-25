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

## 🎯 Model Capabilities & Limitations

DeepGuard AI is a highly specialized academic tool. It is critical to understand exactly what it can and cannot detect:

### 1. What it Detects: Human Faces Only
The engine operates strictly on a **Facial Deepfake Detection** paradigm.
* The system relies entirely on the MTCNN face detector to locate a region of interest. 
* **Conclusion:** The model can **ONLY** analyze images and videos that contain a human face. It cannot analyze landscapes, AI-generated voices, or text.

### 2. Can it detect AI-Generated Cartoons or Anime?
**No.**
* MTCNN is trained on real human faces. It will fail to detect a cartoon, anime, or animal face.
* Even if a cartoon face is forced through the pipeline, the EfficientNet-B4 classifier will produce arbitrary scores because its training dataset consisted of manipulated human skin textures, not drawn or animated pixels.

### 3. Known Limitations
For academic transparency, the following edge-cases and limitations apply:
* **No Face = No Detection:** If a subject's face is completely obscured, turned away from the camera, or wearing a heavy mask, MTCNN will fail to extract a face, halting the analysis.
* **Low Resolution / High Compression:** Deepfake artifacts exist at the pixel level. If a video is heavily compressed (e.g., forwarded through WhatsApp) or extremely blurry, these artifacts are smoothed over. This can result in **False Negatives** (flagging a fake video as real).
* **Extreme Profile Angles:** The MTCNN face detector struggles with extreme 90-degree profile shots. It requires a reasonable view of the eyes, nose, and mouth to establish bounding boxes.
* **100% Synthetic Generation (e.g., Sora, Midjourney):** The model is explicitly optimized to detect **facial manipulations** (e.g., FaceSwap, DeepFaceLab, lip-syncing). It looks for blending boundaries and warping. If a video is generated entirely from scratch by an AI without any face-swapping or blending, the model may struggle to classify it correctly because those specific manipulation artifacts do not exist.

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
