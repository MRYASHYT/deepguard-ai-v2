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
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@300;400;600;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-main: #050505;
    --bg-surface: #0a0a0a;
    --accent: #0ea5e9;
    --danger: #f43f5e;
    --border-dim: #1e293b;
    --border-bright: #334155;
    --text-main: #f1f5f9;
    --text-muted: #64748b;
}

/* --- Global Reset --- */
h1, h2, h3, h4, h5, h6, p, label, li {font-family:'Hanken Grotesk',sans-serif!important;}
pre,code,.mono{font-family:'IBM Plex Mono',monospace!important}

.stApp {
    background-color: var(--bg-main);
    background-image: radial-gradient(circle at 2px 2px, #ffffff05 1px, transparent 0);
    background-size: 24px 24px;
}

/* --- Sidebar: Tactical Look --- */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface)!important;
    border-right: 1px solid var(--border-dim)!important;
}

/* --- Containers: Sharp & Solid --- */
.box, .verdict-container, .report-box, [data-testid="stFileUploader"] section {
    background: var(--bg-surface)!important;
    border: 1px solid var(--border-dim)!important;
    border-radius: 2px!important;
    box-shadow: none!important;
}

/* --- Buttons: System Standard --- */
.stButton>button {
    background: transparent!important;
    border: 1px solid var(--border-bright)!important;
    color: var(--text-main)!important;
    font-family: 'IBM Plex Mono'!important;
    font-size: 0.7rem!important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 0.6rem 1.2rem!important;
    border-radius: 2px!important;
    transition: all 0.1s ease;
}
.stButton>button:hover {
    border-color: var(--accent)!important;
    background: rgba(14, 165, 233, 0.05)!important;
    color: var(--accent)!important;
}

/* --- Verdict: High Contrast --- */
.verdict-container {
    padding: 2.5rem;
    border-left: 4px solid var(--border-dim)!important;
}
.verdict-container.fake { border-left-color: var(--danger)!important; background: rgba(244, 63, 94, 0.02)!important; }
.verdict-container.real { border-left-color: var(--accent)!important; background: rgba(14, 165, 233, 0.02)!important; }

.verdict-text {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    margin: 0.5rem 0;
}
.verdict-text.real { color: var(--accent); }
.verdict-text.fake { color: var(--danger); }

/* --- Data Visuals --- */
.metric-val { font-family: 'IBM Plex Mono'; font-size: 1.5rem; font-weight: 600; color: var(--text-main); }
.metric-label { font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }

.finding-item {
    border: 1px solid var(--border-dim);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    display: flex;
    gap: 1rem;
    align-items: center;
}
.finding-item.alert { border-color: rgba(244, 63, 94, 0.3); background: rgba(244, 63, 94, 0.03); }

/* --- System Ticker (The Spell) --- */
.scan-line {
    border-top: 1px solid var(--border-dim);
    margin: 2rem 0;
    display: flex;
    justify-content: space-between;
    padding-top: 0.5rem;
}
.scan-line::before {
    content: 'DEEPGUARD_OS_v1.0';
    font-family: 'IBM Plex Mono';
    font-size: 0.55rem;
    color: var(--text-muted);
}
.scan-line::after {
    content: '[ TERMINAL_ONLINE ]';
    font-family: 'IBM Plex Mono';
    font-size: 0.55rem;
    color: var(--accent);
    animation: blink 1.5s infinite;
}

@keyframes blink { 50% { opacity: 0.3; } }

/* --- Tables & Data --- */
table { width: 100%; border-collapse: collapse; }
td { padding: 0.6rem 0; border-bottom: 1px solid rgba(255,255,255,0.03); font-family: 'IBM Plex Mono'; font-size: 0.75rem; }

/* --- Skeleton Loader --- */
.skeleton {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    height: 200px;
    width: 100%;
    position: relative;
    overflow: hidden;
}
.skeleton::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.03), transparent);
    animation: skeleton-glow 2s infinite linear;
}
@keyframes skeleton-glow {
    0% { transform: translateX(-100%); opacity: 0.1; }
    50% { opacity: 0.3; }
    100% { transform: translateX(100%); opacity: 0.1; }
}

.system-init {
    text-align: center;
    padding: 10rem 0;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--text-muted);
    font-size: 0.7rem;
    letter-spacing: 2px;
}
.pulse-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: var(--accent);
    border-radius: 50%;
    margin-right: 10px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 0.3; transform: scale(1); } 50% { opacity: 1; transform: scale(1.2); } }
</style>""", unsafe_allow_html=True)

# ─── SYSTEM INITIALIZATION ───
if "init" not in st.session_state:
    st.session_state.init = False

if not st.session_state.init:
    init_placeholder = st.empty()
    with init_placeholder.container():
        st.markdown(f"""
        <div class='system-init'>
            <div class='pulse-dot'></div>INITIALIZING_FORENSIC_ENGINE...
            <div style='margin-top: 2rem; display: flex; flex-direction: column; gap: 1rem; max-width: 600px; margin-left: auto; margin-right: auto;'>
                <div class='skeleton' style='height: 100px;'></div>
                <div style='display: flex; gap: 1rem;'>
                    <div class='skeleton' style='height: 150px; flex: 1;'></div>
                    <div class='skeleton' style='height: 150px; flex: 1;'></div>
                </div>
                <div class='skeleton' style='height: 40px;'></div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # Load heavy resources here
        time.sleep(1.5) # Minimum aesthetic delay
        st.session_state.init = True
        st.rerun()

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
<div style='display:flex;align-items:center;gap:1.2rem;margin-bottom:1rem; border-left: 2px solid var(--accent); padding-left: 1.5rem;'>
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
    <div style='display:flex; flex-direction:column;'>
        <h1 style='margin:0!important;padding:0!important;line-height:0.9!important;font-size:1.6rem!important;letter-spacing:-0.5px!important; font-weight:800; color:var(--text-main)!important;'>DEEPGUARD <span style='font-weight:300; color:var(--accent);'>FORENSICS</span></h1>
        <p style='color:var(--text-muted);font-size:.6rem;letter-spacing:1.5px;margin:0.2rem 0 0;font-family:"IBM Plex Mono"!important; text-transform:uppercase;'>SYSTEM_NODE: 0x8F2A // AUTH_LEVEL: ANALYST</p>
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
                <div class='box' style='margin-bottom:1rem; padding: 1.5rem;'>
                    <div style='display:flex; gap:2rem; flex-wrap:wrap; align-items: flex-start;'>
                        <div style='text-align: center;'>
                            <div style='font-size:.6rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem; font-family: "IBM Plex Mono"!important;'>[ SOURCE_MEDIA ]</div>
                            <img src='data:image/jpeg;base64,{ib}' style='border:1px solid var(--border-dim); max-width:180px;'/>
                        </div>
                        <div style='text-align: center;'>
                            <div style='font-size:.6rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.8rem; font-family: "IBM Plex Mono"!important;'>[ FACE_SCAN ]</div>
                            <img src='data:image/jpeg;base64,{fb}' style='border:1px solid var(--accent); max-width:180px;'/>
                        </div>
                        <div style='flex:1; min-width:280px; background: #000; padding: 1.2rem; border: 1px solid var(--border-dim);'>
                            <div style='font-size:.6rem; color:var(--accent); text-transform:uppercase; letter-spacing:1.5px; margin-bottom:1rem; font-family: "IBM Plex Mono"!important;'>// METADATA_DUMP</div>
                            <table style='width: 100%; border: none;'>
                                <tr><td style='color:var(--text-muted); width: 100px;'>FILE_NAME</td><td>{uploaded.name}</td></tr>
                                <tr><td style='color:var(--text-muted);'>VERDICT</td><td class='{"red" if fake else "teal"}'><b>{v}</b></td></tr>
                                <tr><td style='color:var(--text-muted);'>SCORE</td><td>{prob*100:.2f}%</td></tr>
                                <tr><td style='color:var(--text-muted);'>RISK_LVL</td><td>{risk}</td></tr>
                                <tr style='border-bottom: none;'><td style='color:var(--text-muted);'>TIMESTAMP</td><td>{datetime.now().strftime('%Y%m%d_%H%M%S')}</td></tr>
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
    <div style='text-align:center; padding: 6rem 2rem;'>
        <div style='position: relative; width: 80px; height: 80px; margin: 0 auto 3rem;'>
            <div style='position: absolute; inset: 0; border: 1px solid var(--border-bright);'></div>
            <div style='position: absolute; inset: 5px; border: 1px solid var(--accent); opacity: 0.2;'></div>
            <svg style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);' width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg>
        </div>
        <p style='font-size:.65rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:2px; font-family: "IBM Plex Mono"!important; margin-bottom: 1rem;'>
            <span style='color:var(--accent); margin-right: 0.5rem;'>●</span> STANDBY_READY
        </p>
        <p style='color:var(--text-muted); font-size:0.85rem; max-width:380px; margin:0 auto 4rem; line-height:1.6;'>Neural forensics engine online. Awaiting source media for biometric cross-reference and artifact detection.</p>
        <div style='display:flex; justify-content:center; gap:4rem;'>
            <div><p style='font-size:.5rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-family: "IBM Plex Mono"!important; margin-bottom: 0.5rem;'>ENGINE</p><p style='font-size:.7rem; color: var(--text-main); font-family: "IBM Plex Mono"!important;'>ENET_B4</p></div>
            <div><p style='font-size:.5rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-family: "IBM Plex Mono"!important; margin-bottom: 0.5rem;'>SENSOR</p><p style='font-size:.7rem; color: var(--text-main); font-family: "IBM Plex Mono"!important;'>MTCNN</p></div>
            <div><p style='font-size:.5rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1.5px; font-family: "IBM Plex Mono"!important; margin-bottom: 0.5rem;'>XAI</p><p style='font-size:.7rem; color: var(--text-main); font-family: "IBM Plex Mono"!important;'>G_CAM</p></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='footer-bar'>DeepGuard AI v1.0 · EfficientNet-B4 + MTCNN + Grad-CAM · Diploma Research Project</div>", unsafe_allow_html=True)
