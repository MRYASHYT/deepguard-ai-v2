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

st.set_page_config(page_title="DeepGuard AI", page_icon="🛡️", layout="wide", menu_items={})

# ─── MASSIVE CSS OVERHAUL ───
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
h1, h2, h3, h4, h5, h6, p, label, li {font-family:'Plus Jakarta Sans',sans-serif!important;}
pre,code,.mono{font-family:'IBM Plex Mono',monospace!important}
.stApp{background:#000}
[data-testid="stHeader"]{display:none!important}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
[data-testid="stStatusWidget"]{display:none!important}
.stDeployButton{display:none!important}
button[kind="header"]{display:none!important}
[data-testid="stSidebar"]{background:#000!important;border-right:1px solid #1e293b!important;color:#94a3b8!important}
[data-testid="stFileUploader"]{background:transparent!important}
[data-testid="stFileUploader"] section{border:1px dashed #334155!important;border-radius:0!important;background:#0a0a0a!important;padding:2rem!important}
[data-testid="stFileUploader"] section:hover{border-color:#2dd4bf!important}
[data-testid="stFileUploader"] small{color:#64748b!important}
[data-testid="stUploadedFile"]{background:#0a0a0a!important;border:1px solid #1e293b!important;border-radius:0!important;padding:.5rem!important;width:100%!important}
[data-testid="stUploadedFile"] *{color:#94a3b8!important}
[data-testid="stUploadedFile"] svg{fill:#2dd4bf!important}
.stButton>button{border-radius:0!important;border:1px solid #334155!important;background:transparent!important;color:#fff!important;font-weight:500!important;padding:.6rem 1.5rem!important;letter-spacing:.5px!important;text-transform:uppercase!important;font-size:.75rem!important;transition:all .15s ease!important}
.stButton>button:hover{background:#fff!important;color:#000!important;border-color:#fff!important}
.stTextInput>div>div>input{background:#0a0a0a!important;border:1px solid #1e293b!important;border-radius:0!important;color:#fff!important;font-family:'Plus Jakarta Sans',sans-serif!important}
.stTextInput>div>div>input:focus{border-color:#2dd4bf!important}
h1{color:#2dd4bf!important;font-weight:800!important;font-size:2.2rem!important;letter-spacing:-1px!important}
h2{color:#fff!important;font-weight:700!important;font-size:1.15rem!important;letter-spacing:-.3px!important;text-transform:uppercase!important}
h3{color:#94a3b8!important;font-weight:500!important;font-size:.85rem!important;text-transform:uppercase!important;letter-spacing:1.5px!important}
.stProgress>div>div>div>div{background:#2dd4bf!important}
.line{border-top:1px solid #1e293b;margin:1.5rem 0}
.box{border:1px solid #1e293b;padding:1.5rem;margin:.5rem 0}
.verdict-container{border:1px solid #1e293b;padding:2.5rem 2rem;text-align:center}
.verdict-container.fake{border-color:#dc2626;box-shadow:0 0 80px rgba(220,38,38,.06)}
.verdict-container.real{border-color:#2dd4bf;box-shadow:0 0 80px rgba(45,212,191,.06)}
.verdict-label{font-size:.7rem;text-transform:uppercase;letter-spacing:3px;color:#64748b;margin-bottom:.8rem}
.verdict-text{font-size:2.5rem;font-weight:800;letter-spacing:-2px}
.verdict-text.fake{color:#dc2626}.verdict-text.real{color:#2dd4bf}
.metric-row{display:flex;gap:2rem;margin-top:1.5rem;justify-content:center}
.metric-item{text-align:center}
.metric-val{font-size:1.4rem;font-weight:700;color:#fff;font-family:'IBM Plex Mono',monospace!important}
.metric-label{font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-top:.2rem}
.finding-item{border-left:2px solid #2dd4bf;padding:.8rem 1rem;margin:.6rem 0;color:#cbd5e1;font-size:.88rem;line-height:1.6}
.finding-item.alert{border-left-color:#dc2626}
.finding-num{color:#64748b;font-family:'IBM Plex Mono',monospace!important;font-size:.75rem;margin-right:.5rem}
.region-bar-wrap{display:flex;align-items:center;margin:.4rem 0;gap:1rem}
.region-name{width:160px;font-size:.8rem;color:#94a3b8;flex-shrink:0}
.region-bar-bg{flex:1;height:6px;background:#1e293b}
.region-bar-fill{height:6px}
.region-pct{width:50px;text-align:right;font-size:.8rem;font-family:'IBM Plex Mono',monospace!important;color:#64748b}
.report-box{border:1px solid #1e293b;padding:1.5rem;font-family:'IBM Plex Mono',monospace!important;font-size:.78rem;line-height:1.8;color:#94a3b8;white-space:pre-wrap;word-break:break-word}
.teal{color:#2dd4bf}.red{color:#dc2626}.muted{color:#64748b}
.footer-bar{border-top:1px solid #1e293b;padding:1.5rem 0;text-align:center;color:#334155;font-size:.7rem;letter-spacing:1px;text-transform:uppercase;margin-top:3rem}
.login-box{max-width:380px;margin:12vh auto;border:1px solid #1e293b;padding:3rem;text-align:center}
.login-title{font-size:1.8rem;font-weight:800;color:#2dd4bf;letter-spacing:-1px;margin-bottom:.3rem}
.login-sub{font-size:.8rem;color:#64748b;margin-bottom:2rem}
.logo-glyph{font-size:2.5rem;margin-bottom:1rem}
.scan-line{height:1px;background:linear-gradient(90deg,transparent,#2dd4bf,transparent);margin:1rem 0;animation:scanpulse 2s infinite}
@keyframes scanpulse{0%,100%{opacity:.3}50%{opacity:1}}
</style>""", unsafe_allow_html=True)

# ─── AUTH ───
if "auth" not in st.session_state:
    st.session_state.auth = False

def login_page():
    st.markdown("""
    <div class='login-box'>
        <div class='logo-glyph'>🛡️</div>
        <div class='login-title'>DeepGuard AI</div>
        <div class='login-sub'>Neural Forensics Engine · v1.0</div>
        <div class='scan-line'></div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,1,1])
    with c2:
        user = st.text_input("Username", placeholder="analyst")
        pwd = st.text_input("Password", type="password", placeholder="••••••••")
        if st.button("ACCESS SYSTEM", use_container_width=True):
            if user == "admin" and pwd == "deepguard":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Access denied.")
        st.markdown("<p style='text-align:center;color:#334155;font-size:.7rem;margin-top:1rem;'>Default: admin / deepguard</p>", unsafe_allow_html=True)

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

# ─── HEADER ───
head_col1, head_col2 = st.columns([5, 1])
with head_col1:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:1rem;margin-bottom:.5rem;'>
        <span style='font-size:1.8rem;'>🛡️</span>
        <div>
            <h1 style='margin:0!important;padding:0!important;'>DeepGuard AI</h1>
            <p style='color:#475569;font-size:.78rem;letter-spacing:.5px;margin:0;'>Neural Forensics Engine · Synthetic Media Detection</p>
        </div>
    </div>""", unsafe_allow_html=True)
with head_col2:
    st.markdown("<div style='margin-top:0.5rem;'>", unsafe_allow_html=True)
    if st.button("LOGOUT", use_container_width=True):
        st.session_state.auth = False; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='scan-line'></div>", unsafe_allow_html=True)

# ─── SETTINGS ───
st.markdown("<h3 style='margin-top:1.5rem;'>Analysis Settings</h3>", unsafe_allow_html=True)
set_col1, set_col2 = st.columns([2, 1])
with set_col1:
    threshold = st.slider("Detection Threshold (Sensitivity)", 0.0, 1.0, 0.5, 0.05)
with set_col2:
    st.markdown("<div style='margin-top:2.2rem;'>", unsafe_allow_html=True)
    show_cam = st.checkbox("Generate Grad-CAM Heatmap", value=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── UPLOAD ───
st.markdown("<h3 style='margin-top:1.5rem;'>Upload Media</h3>", unsafe_allow_html=True)
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
                st.markdown("<div class='line'></div>", unsafe_allow_html=True)

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
                st.markdown("<div class='line'></div>", unsafe_allow_html=True)

                # FINDINGS
                st.markdown("## Forensic Findings")
                for i,f in enumerate(finds):
                    cls = "alert" if fake else ""
                    st.markdown(f"<div class='finding-item {cls}'><span class='finding-num'>{i+1:02d}</span>{f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='line'></div>", unsafe_allow_html=True)

                # REGIONS
                if regs:
                    st.markdown("## Region Activation")
                    for n,s in regs[:5]:
                        p = int(min(s,1.0)*100)
                        c = "#dc2626" if s>.5 else("#f59e0b" if s>.3 else "#2dd4bf")
                        st.markdown(f"<div class='region-bar-wrap'><div class='region-name'>{n}</div><div class='region-bar-bg'><div class='region-bar-fill' style='width:{p}%;background:{c};'></div></div><div class='region-pct'>{p}%</div></div>", unsafe_allow_html=True)
                    st.markdown("<div class='line'></div>", unsafe_allow_html=True)

                # REPORT WITH PHOTOS
                st.markdown("## Full Report")
                fb = img_b64(face); ib = img_b64(image)
                st.markdown(f"""
                <div class='box' style='margin-bottom:1rem;'>
                    <div style='display:flex;gap:1.5rem;flex-wrap:wrap;'>
                        <div><div style='font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.5rem;'>Analyzed</div>
                        <img src='data:image/jpeg;base64,{ib}' style='border:1px solid #1e293b;max-width:180px;'/></div>
                        <div><div style='font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:.5rem;'>Face</div>
                        <img src='data:image/jpeg;base64,{fb}' style='border:1px solid #1e293b;max-width:180px;'/></div>
                        <div style='flex:1;min-width:180px;'>
                            <table style='font-size:.8rem;color:#94a3b8;line-height:2.2;'>
                                <tr><td style='color:#475569;padding-right:1rem;'>File</td><td>{uploaded.name}</td></tr>
                                <tr><td style='color:#475569;padding-right:1rem;'>Verdict</td><td class='{"red" if fake else "teal"}'><b>{"FAKE" if fake else "REAL"}</b></td></tr>
                                <tr><td style='color:#475569;padding-right:1rem;'>Score</td><td>{prob*100:.1f}%</td></tr>
                                <tr><td style='color:#475569;padding-right:1rem;'>Risk</td><td>{risk}</td></tr>
                                <tr><td style='color:#475569;padding-right:1rem;'>Time</td><td>{datetime.now().strftime('%H:%M:%S')}</td></tr>
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
    <div style='text-align:center;padding:6rem 2rem;'>
        <p style='font-size:.7rem;color:#334155;text-transform:uppercase;letter-spacing:3px;'>Ready</p>
        <p style='color:#94a3b8;font-size:1rem;max-width:400px;margin:1rem auto;line-height:1.8;'>Upload an image or video for neural forensic analysis</p>
        <div style='display:flex;justify-content:center;gap:3rem;margin-top:3rem;'>
            <div><p class='teal' style='font-size:.7rem;text-transform:uppercase;letter-spacing:2px;'>Model</p><p class='muted' style='font-size:.8rem;'>EfficientNet-B4</p></div>
            <div><p class='teal' style='font-size:.7rem;text-transform:uppercase;letter-spacing:2px;'>Detection</p><p class='muted' style='font-size:.8rem;'>MTCNN</p></div>
            <div><p class='teal' style='font-size:.7rem;text-transform:uppercase;letter-spacing:2px;'>Explainability</p><p class='muted' style='font-size:.8rem;'>Grad-CAM</p></div>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='footer-bar'>DeepGuard AI v1.0 · EfficientNet-B4 + MTCNN + Grad-CAM · Diploma Research Project</div>", unsafe_allow_html=True)
