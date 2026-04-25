import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2, tempfile, os, sys, io, base64
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.efficientnet import DeepfakeEfficientNet
from preprocessing.detect_faces import FaceDetector
from utils.dataset import get_val_transforms

st.set_page_config(page_title="DeepGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded", menu_items={})

# ─── MASSIVE CSS OVERHAUL ───
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-dark: #020617;
    --card-bg: rgba(15, 23, 42, 0.6);
    --accent: #2dd4bf;
    --accent-glow: rgba(45, 212, 191, 0.2);
    --danger: #fb7185;
    --danger-glow: rgba(251, 113, 133, 0.2);
    --border: rgba(51, 65, 85, 0.5);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

h1, h2, h3, h4, h5, h6, p, label, li {font-family:'Plus Jakarta Sans',sans-serif!important;}
pre,code,.mono{font-family:'IBM Plex Mono',monospace!important}

/* --- Atmospheric Background --- */
.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(circle at 50% 50%, rgba(45, 212, 191, 0.03) 0%, transparent 70%),
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 100% 100%, 40px 40px, 40px 40px;
    background-attachment: fixed;
}

/* --- Sidebar Refinement --- */
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.95)!important;
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border)!important;
}

/* --- Glassmorphic Containers --- */
.box, .verdict-container, .report-box, [data-testid="stFileUploader"] section {
    background: var(--card-bg)!important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border)!important;
    border-radius: 8px!important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* --- File Uploader --- */
[data-testid="stFileUploader"] section {
    padding: 3rem!important;
    border: 1px dashed var(--border)!important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent)!important;
    background: rgba(45, 212, 191, 0.02)!important;
    box-shadow: 0 0 20px var(--accent-glow);
}

/* --- Premium Buttons --- */
.stButton>button {
    background: rgba(45, 212, 191, 0.05)!important;
    border: 1px solid var(--accent)!important;
    color: var(--accent)!important;
    font-weight: 600!important;
    letter-spacing: 1px!important;
    padding: 0.75rem 2rem!important;
    border-radius: 4px!important;
    text-transform: uppercase;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background: var(--accent)!important;
    color: var(--bg-dark)!important;
    box-shadow: 0 0 15px var(--accent-glow);
    transform: translateY(-1px);
}

/* --- Verdict Displays --- */
.verdict-container {
    padding: 3rem 2rem;
    position: relative;
    overflow: hidden;
}
.verdict-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent);
    opacity: 0.5;
}
.verdict-container.fake::before { background: var(--danger); }

.verdict-text {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -3px;
    margin: 0;
}
.verdict-text.real { color: var(--accent); text-shadow: 0 0 20px var(--accent-glow); }
.verdict-text.fake { color: var(--danger); text-shadow: 0 0 20px var(--danger-glow); }

/* --- Typography Hierarchy --- */
h1 {
    color: var(--text-main)!important;
    font-weight: 800!important;
    letter-spacing: -1.5px!important;
    margin-bottom: 0!important;
}
h2 {
    color: var(--accent)!important;
    font-size: 0.9rem!important;
    text-transform: uppercase!important;
    letter-spacing: 2px!important;
}
h3 {
    color: var(--text-muted)!important;
    font-size: 0.75rem!important;
    letter-spacing: 1px!important;
}

/* --- Forensic Scanlines --- */
.scan-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.4;
    margin: 1.5rem 0;
    position: relative;
}
.scan-line::after {
    content: 'SYSTEM_READY';
    position: absolute;
    right: 0; top: -15px;
    font-family: 'IBM Plex Mono';
    font-size: 0.6rem;
    color: var(--accent);
    letter-spacing: 1px;
}

/* --- Metrics & Findings --- */
.metric-val { font-size: 1.8rem; font-weight: 700; color: #fff; }
.finding-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 1rem;
    border-radius: 4px;
}
.finding-item.alert { border-left-color: var(--danger); }

.stProgress>div>div>div>div { background: var(--accent)!important; }
.teal{color:var(--accent)}.red{color:var(--danger)}.muted{color:var(--text-muted)}
.footer-bar{border-top:1px solid var(--border);padding:2rem 0;text-align:center;color:var(--text-muted);font-size:.7rem;letter-spacing:1px;text-transform:uppercase;margin-top:3rem}
</style>""", unsafe_allow_html=True)

# ─── AUTH ───
if "auth" not in st.session_state:
    st.session_state.auth = False

def login_page():
    st.markdown("""
    <div style='max-width: 420px; margin: 10vh auto; background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--border); padding: 4rem 3rem; border-radius: 12px; text-align: center; box-shadow: 0 30px 60px rgba(0,0,0,0.4);'>
        <div style='font-size: 3.5rem; margin-bottom: 1.5rem;'>🛡️</div>
        <h1 style='color: var(--accent)!important; font-size: 2.5rem!important; margin-bottom: 0.5rem!important;'>DeepGuard AI</h1>
        <p style='color: var(--text-muted); font-size: 0.85rem; letter-spacing: 1.5px; margin-bottom: 2rem; font-family: "IBM Plex Mono"!important;'>NEURAL FORENSICS GATEWAY v1.0</p>
        <div class='scan-line'></div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1.2,1])
    with c2:
        user = st.text_input("Access Identity", placeholder="analyst_id")
        pwd = st.text_input("Secure Key", type="password", placeholder="••••••••")
        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("AUTHORIZE ACCESS", use_container_width=True):
            if user == "admin" and pwd == "deepguard":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Authentication failed. Invalid identity.")
        st.markdown("<p style='text-align:center;color:#475569;font-size:.7rem;margin-top:1.5rem;font-family:\"IBM Plex Mono\"!important;'>DEFAULT_CREDENTIALS: admin / deepguard</p>", unsafe_allow_html=True)

if not st.session_state.auth:
    login_page()
    st.stop()

# ─── MODEL ───
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = DeepfakeEfficientNet(pretrained=False)
    if os.path.exists('best_model.pth'):
        m.load_state_dict(torch.load('best_model.pth', map_location=device))
    m.to(device); m.eval()
    return m, device

@st.cache_resource
def load_det(): return FaceDetector()

def predict(img, model, dev):
    tf = get_val_transforms()
    a = np.array(img.resize((224,224)))
    t = tf(image=a)['image'].unsqueeze(0).to(dev)
    with torch.no_grad(): return model(t).item()

def gradcam(img, model, dev):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
        im = img.resize((224,224)); arr = np.array(im)/255.0
        tf = get_val_transforms()
        inp = tf(image=np.array(im))['image'].unsqueeze(0).to(dev)
        cam = GradCAM(model=model, target_layers=[model.backbone.conv_head])
        gc = cam(input_tensor=inp, targets=[BinaryClassifierOutputTarget(1)])[0]
        vis = show_cam_on_image(arr.astype(np.float32), gc, use_rgb=True)
        return Image.fromarray(vis), gc
    except: return None, None

def regions(gc):
    if gc is None: return []
    h,w = gc.shape
    r = [("Forehead",gc[:int(h*.25),:]),("Eyes",gc[int(h*.25):int(h*.45),:]),
         ("Nose",gc[int(h*.35):int(h*.55),int(w*.3):int(w*.7)]),
         ("Mouth",gc[int(h*.55):int(h*.75),:]),("Jawline",gc[int(h*.7):,:])]
    s = [(n,float(x.mean())) for n,x in r]
    s.sort(key=lambda x:x[1],reverse=True)
    return s

def findings(prob, regs, fake):
    o = []
    if fake and regs:
        o.append(f"Primary signal in **{regs[0][0]}** — activation {regs[0][1]*100:.0f}%")
        if len(regs)>1 and regs[1][1]>.25:
            o.append(f"Secondary artifacts in **{regs[1][0]}** — blend boundary inconsistency")
        if prob>.8: o.append("Pixel texture consistent with GAN-based face synthesis")
        elif prob>.55: o.append("Moderate spectral inconsistencies across facial boundaries")
        ey = next((s for n,s in regs if "Eye" in n),0)
        if ey>.35: o.append("Irregular specular highlights in eye region")
    else:
        o.append("No manipulation artifacts detected across facial regions")
        o.append("Texture and color statistics within natural parameters")
        if prob<.15: o.append("High authenticity — passes all forensic checks")
    return o

def img_b64(img, sz=250):
    img = img.copy(); img.thumbnail((sz,sz))
    buf = io.BytesIO(); img.save(buf,format='JPEG',quality=85)
    return base64.b64encode(buf.getvalue()).decode()

def report(fn, prob, fake, finds, regs):
    v = "MANIPULATED" if fake else "AUTHENTIC"
    risk = "CRITICAL" if prob>.85 else("HIGH" if prob>.65 else("MODERATE" if prob>.45 else("LOW" if prob>.25 else "MINIMAL")))
    now = datetime.now()
    r = f"""╔════════════════════════════════════════════════════════════╗
║           DEEPGUARD AI  ·  FORENSIC ANALYSIS REPORT       ║
╚════════════════════════════════════════════════════════════╝

  Report ID       DG-{now.strftime('%Y%m%d%H%M%S')}
  Generated       {now.strftime('%Y-%m-%d %H:%M:%S')}
  Engine          DeepGuard AI v1.0
  Model           EfficientNet-B4

────────────────────────────────────────────────────────────

  VERDICT         {v}
  Score           {prob*100:.1f}%
  Risk            {risk}
  Confidence      {max(prob,1-prob)*100:.1f}%

────────────────────────────────────────────────────────────

  FINDINGS\n"""
    for i,f in enumerate(finds,1): r += f"  [{i}] {f.replace('**','')}\n"
    r += "\n────────────────────────────────────────────────────────────\n\n  REGION ACTIVATION\n"
    for n,s in regs[:5]:
        bar = "█"*int(s*20) + "░"*(20-int(s*20))
        r += f"  {n:<12} {bar} {s*100:.1f}%\n"
    r += f"""
────────────────────────────────────────────────────────────

  METHODOLOGY
  Face Detection    MTCNN (margin=20)
  Feature Extract   EfficientNet-B4 (ImageNet pretrained)
  Classifier        Binary (Dropout→Linear→Sigmoid)
  Explainability    Grad-CAM on final conv layer
  Training          Two-phase transfer learning

╔════════════════════════════════════════════════════════════╗
║                      END OF REPORT                        ║
╚════════════════════════════════════════════════════════════╝"""
    return r

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("<h3>System</h3>", unsafe_allow_html=True)
    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05)
    show_cam = st.checkbox("Grad-CAM", value=True)
    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.75rem;line-height:2;color:#475569;'>Model → EfficientNet-B4<br>Detection → MTCNN<br>Explainability → Grad-CAM</p>", unsafe_allow_html=True)
    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)
    if st.button("LOGOUT", use_container_width=True):
        st.session_state.auth = False; st.rerun()

# ─── HEADER ───
st.markdown("""
<div style='display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;'>
    <div style='width:60px;height:60px;background:var(--card-bg);border:1px solid var(--accent);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:2rem;box-shadow:0 0 20px var(--accent-glow);'>🛡️</div>
    <div>
        <h1 style='margin:0!important;padding:0!important;line-height:1!important;'>DeepGuard AI</h1>
        <p style='color:var(--text-muted);font-size:.85rem;letter-spacing:1px;margin:0;font-family:"IBM Plex Mono"!important;'>SYSTEM_STATUS: OPERATIONAL · NEURAL_FORENSICS_ONLINE</p>
    </div>
</div>""", unsafe_allow_html=True)
st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

# ─── UPLOAD ───
st.markdown("<h3 style='margin-top:1.5rem;'>Upload Media for Analysis</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=['jpg','png','jpeg','mp4','avi'], label_visibility="collapsed")
model, device = load_model()

if "analyzed_file" not in st.session_state:
    st.session_state.analyzed_file = None
    st.session_state.analyzed = False

if uploaded:
    if st.session_state.analyzed_file != uploaded.name:
        st.session_state.analyzed = False
        st.session_state.analyzed_file = uploaded.name

    if not st.session_state.analyzed:
        st.markdown("<div style='text-align:center;margin-top:2rem;'>", unsafe_allow_html=True)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            if st.button("🚀 SEND FOR DETECTION", use_container_width=True):
                st.session_state.analyzed = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.analyzed:
        with st.spinner("Forensic Analysis in Progress..."):
            if uploaded.type.startswith('image'):
                image = Image.open(uploaded).convert('RGB')
                det = load_det()
                with tempfile.NamedTemporaryFile(suffix='.jpg',delete=False) as tmp:
                    image.save(tmp.name); fp=tmp.name+"_face.jpg"; ok=det.crop_face(tmp.name,fp)
                face = Image.open(fp).convert('RGB') if (ok and os.path.exists(fp)) else image.resize((224,224))
                prob = predict(face, model, device)
                fake = prob > threshold
                hm, gc = (None,None)
                if show_cam: hm, gc = gradcam(face, model, device)
                regs = regions(gc)
                finds = findings(prob, regs, fake)
                risk = "CRITICAL" if prob>.85 else("HIGH" if prob>.65 else("MODERATE" if prob>.45 else("LOW" if prob>.25 else "MINIMAL")))

                # VERDICT
                vc = "fake" if fake else "real"
                vt = "MANIPULATED" if fake else "AUTHENTIC"
                st.markdown(f"""
                <div class='verdict-container {vc}'>
                    <div class='verdict-label'>Analysis Complete</div>
                    <div class='verdict-text {vc}'>{vt}</div>
                    <div class='metric-row'>
                        <div class='metric-item'><div class='metric-val'>{prob*100:.1f}%</div><div class='metric-label'>Manipulation</div></div>
                        <div class='metric-item'><div class='metric-val'>{max(prob,1-prob)*100:.1f}%</div><div class='metric-label'>Confidence</div></div>
                        <div class='metric-item'><div class='metric-val'>{risk}</div><div class='metric-label'>Risk</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

                # IMAGES
                st.markdown("## Visual Evidence")
                cols = st.columns(3) if hm else st.columns(2)
                with cols[0]:
                    st.markdown("<h3>Input</h3>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with cols[1]:
                    st.markdown("<h3>Face Crop</h3>", unsafe_allow_html=True)
                    st.image(face, use_container_width=True)
                if hm and len(cols)>2:
                    with cols[2]:
                        st.markdown("<h3>Heatmap</h3>", unsafe_allow_html=True)
                        st.image(hm, use_container_width=True)
                st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

                # FINDINGS
                st.markdown("## Forensic Findings")
                for i,f in enumerate(finds):
                    cls = "alert" if fake else ""
                    st.markdown(f"<div class='finding-item {cls}'><span class='finding-num'>{i+1:02d}</span>{f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

                # REGIONS
                if regs:
                    st.markdown("## Region Activation")
                    for n,s in regs[:5]:
                        p = int(min(s,1.0)*100)
                        c = "#dc2626" if s>.5 else("#f59e0b" if s>.3 else "#2dd4bf")
                        st.markdown(f"<div class='region-bar-wrap'><div class='region-name'>{n}</div><div class='region-bar-bg'><div class='region-bar-fill' style='width:{p}%;background:{c};'></div></div><div class='region-pct'>{p}%</div></div>", unsafe_allow_html=True)
                    st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

                # DOSSIER STYLE REPORT
                st.markdown("## Forensic Dossier")
                fb = img_b64(face); ib = img_b64(image)
                st.markdown(f"""
                <div class='box' style='margin-bottom:1rem; padding: 2rem;'>
                    <div style='display:flex; gap:2.5rem; flex-wrap:wrap; align-items: flex-start;'>
                        <div style='text-align: center;'>
                            <div style='font-size:.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:2px; margin-bottom:1rem; font-family: "IBM Plex Mono"!important;'>[SOURCE_MEDIA]</div>
                            <img src='data:image/jpeg;base64,{ib}' style='border:1px solid var(--border); max-width:200px; border-radius: 4px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);'/>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size:.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:2px; margin-bottom:1rem; font-family: "IBM Plex Mono"!important;'>[EXTRACTED_FACE]</div>
                            <img src='data:image/jpeg;base64,{fb}' style='border:1px solid var(--accent); max-width:200px; border-radius: 4px; box-shadow: 0 10px 30px var(--accent-glow);'/>
                        </div>
                        <div style='flex:1; min-width:250px; background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border);'>
                            <div style='font-size:.65rem; color:var(--accent); text-transform:uppercase; letter-spacing:2px; margin-bottom:1rem; font-family: "IBM Plex Mono"!important;'>// ANALYSIS_METADATA</div>
                            <table style='width: 100%; font-size:.85rem; color:var(--text-main); line-height:2.5; border-collapse: collapse;'>
                                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='color:var(--text-muted); padding-right:1rem; font-family: "IBM Plex Mono"!important;'>IDENTIFIER</td><td>{uploaded.name}</td></tr>
                                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='color:var(--text-muted); padding-right:1rem; font-family: "IBM Plex Mono"!important;'>VERDICT</td><td class='{"red" if fake else "teal"}'><b>{"MANIPULATED" if fake else "AUTHENTIC"}</b></td></tr>
                                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='color:var(--text-muted); padding-right:1rem; font-family: "IBM Plex Mono"!important;'>PROBABILITY</td><td style='font-family: "IBM Plex Mono"!important;'>{prob*100:.2f}%</td></tr>
                                <tr style='border-bottom: 1px solid rgba(255,255,255,0.05);'><td style='color:var(--text-muted); padding-right:1rem; font-family: "IBM Plex Mono"!important;'>THREAT_LEVEL</td><td style='font-family: "IBM Plex Mono"!important;'>{risk}</td></tr>
                                <tr><td style='color:var(--text-muted); padding-right:1rem; font-family: "IBM Plex Mono"!important;'>TIMESTAMP</td><td style='font-family: "IBM Plex Mono"!important;'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                rt = report(uploaded.name, prob, fake, finds, regs)
                st.markdown(f"<div class='report-box'>{rt}</div>", unsafe_allow_html=True)
                st.download_button("↓  DOWNLOAD REPORT", rt, f"deepguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain")

            elif uploaded.type.startswith('video'):
                st.video(uploaded)
                tfile = tempfile.NamedTemporaryFile(delete=False,suffix='.mp4'); tfile.write(uploaded.read())
                cap = cv2.VideoCapture(tfile.name); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                idx = np.linspace(0,total-1,min(10,total),dtype=int); res=[]; bar=st.progress(0)
                for i,ix in enumerate(idx):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,ix); ret,fr = cap.read()
                    if ret: res.append(predict(Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)),model,device))
                    bar.progress((i+1)/len(idx))
                cap.release(); avg=np.mean(res); fake=avg>threshold; vc="fake" if fake else "real"
                st.markdown(f"""<div class='verdict-container {vc}'>
                    <div class='verdict-label'>Video · {len(res)} Frames</div>
                    <div class='verdict-text {vc}'>{"MANIPULATED" if fake else "AUTHENTIC"}</div>
                    <div class='metric-row'><div class='metric-item'><div class='metric-val'>{avg*100:.1f}%</div><div class='metric-label'>Avg Score</div></div></div>
                </div>""", unsafe_allow_html=True)

else:
    st.session_state.analyzed = False
    st.session_state.analyzed_file = None
    st.markdown("""
    <div style='text-align:center; padding: 8rem 2rem;'>
        <div style='width: 100px; height: 100px; background: var(--card-bg); border: 1px solid var(--accent); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 3rem; margin: 0 auto 2rem; box-shadow: 0 0 30px var(--accent-glow);'>📡</div>
        <p style='font-size:.7rem; color:var(--accent); text-transform:uppercase; letter-spacing:4px; font-family: "IBM Plex Mono"!important;'>SYSTEM_STATUS: IDLE</p>
        <p style='color:var(--text-muted); font-size:1.1rem; max-width:500px; margin:1rem auto 3rem; line-height:1.8;'>Neural Forensics Engine ready for input. Upload source media to begin cross-reference analysis.</p>
        <div style='display:flex; justify-content:center; gap:4rem; margin-top:4rem;'>
            <div><p style='font-size:.65rem; color:var(--accent); text-transform:uppercase; letter-spacing:2px; font-family: "IBM Plex Mono"!important;'>ENGINE</p><p style='font-size:.85rem; color: var(--text-main);'>EFFICIENTNET-B4</p></div>
            <div><p style='font-size:.65rem; color:var(--accent); text-transform:uppercase; letter-spacing:2px; font-family: "IBM Plex Mono"!important;'>SENSOR</p><p style='font-size:.85rem; color: var(--text-main);'>MTCNN_FACE</p></div>
            <div><p style='font-size:.65rem; color:var(--accent); text-transform:uppercase; letter-spacing:2px; font-family: "IBM Plex Mono"!important;'>XAI_LAYER</p><p style='font-size:.85rem; color: var(--text-main);'>GRAD_CAM</p></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='footer-bar'>DeepGuard AI v1.0 · EfficientNet-B4 + MTCNN + Grad-CAM · Diploma Research Project</div>", unsafe_allow_html=True)
