"""Streamlit UI for the skin lesion classifier (styled with custom CSS)."""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬", layout="wide")

# name, short description, colour
INFO = {
    "akiec": ("Actinic keratosis", "Rough, scaly patch caused by sun damage; can be pre-cancerous.", "#f59e0b"),
    "bcc":   ("Basal cell carcinoma", "A common, slow-growing form of skin cancer.", "#ef4444"),
    "bkl":   ("Benign keratosis", "A harmless age-spot or wart-like growth.", "#10b981"),
    "df":    ("Dermatofibroma", "A small, firm and harmless skin nodule.", "#06b6d4"),
    "mel":   ("Melanoma", "A serious skin cancer where early detection matters most.", "#dc2626"),
    "nv":    ("Melanocytic nevus", "A common mole.", "#3b82f6"),
    "vasc":  ("Vascular lesion", "A blood-vessel growth, such as an angioma.", "#8b5cf6"),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button { font-family: 'Poppins', sans-serif; }
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.stApp {background: linear-gradient(160deg, #eef5fa 0%, #f8fbfd 55%, #eaf2f7 100%);}
.block-container {max-width: 1100px; padding-top: 1.5rem; padding-bottom: 3rem;}

/* hero */
.hero {background: linear-gradient(120deg, #1f6f8b 0%, #2b7a9e 45%, #4aa3c4 100%);
       padding: 2.2rem 2.4rem; border-radius: 20px; color: #fff;
       box-shadow: 0 12px 30px rgba(43,122,158,.28); margin-bottom: 1.6rem;}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 700; color: #fff;}
.hero p {margin: .5rem 0 0; font-size: 1.02rem; opacity: .92; color: #fff;}
.chips {margin-top: 1rem;}
.chip {display: inline-block; background: rgba(255,255,255,.18); padding: .28rem .8rem;
       border-radius: 999px; font-size: .78rem; margin: 0 .4rem .3rem 0;}

/* cards */
.card {background: #fff; border-radius: 16px; padding: 1.3rem 1.5rem; color: #1c2b33;
       box-shadow: 0 4px 18px rgba(20,60,80,.08); margin-bottom: 1.1rem;}
.card h4 {margin: 0 0 .8rem; font-size: 1rem; font-weight: 600; color: #1f6f8b;}
.result {border-left: 7px solid var(--c);}
.result .label {font-size: .72rem; letter-spacing: .09em; text-transform: uppercase; color: #6b7c86;}
.result .name {font-size: 1.6rem; font-weight: 700; margin: .15rem 0; color: #1c2b33;}
.result .desc {color: #4a5b65; font-size: .9rem; margin-bottom: .9rem;}
.meter {height: 10px; background: #e6edf2; border-radius: 99px; overflow: hidden;}
.meter > div {height: 100%; border-radius: 99px; background: var(--c);}
.conf {font-size: .85rem; color: #4a5b65; margin-top: .4rem;}
.conf b {color: #1c2b33; font-size: 1.05rem;}

/* probability bars */
.row {display: flex; align-items: center; margin: .5rem 0; font-size: .85rem; color: #1c2b33;}
.row .n {width: 175px;}
.row .bar {flex: 1; height: 9px; background: #e9eff3; border-radius: 99px; overflow: hidden;}
.row .bar > div {height: 100%; border-radius: 99px;}
.row .p {width: 55px; text-align: right; font-weight: 600;}

/* notices */
.warn {background: #fff7e6; border: 1px solid #fbd38d; color: #7a4b00; border-radius: 12px;
       padding: .8rem 1rem; font-size: .85rem; margin-bottom: 1rem;}
.disclaimer {background: #fdecec; border: 1px solid #f5b5b5; color: #7a1f1f; border-radius: 12px;
             padding: .9rem 1.1rem; font-size: .82rem; margin-top: 1rem;}

/* uploader + image */
[data-testid="stFileUploader"] section {background: #fff; border: 2px dashed #8cc0d6;
                                        border-radius: 16px; padding: 1.6rem;}
[data-testid="stFileUploader"] section:hover {border-color: #2b7a9e; background: #f5fbfe;}
[data-testid="stImage"] img {border-radius: 14px; box-shadow: 0 4px 16px rgba(20,60,80,.15);}

/* sidebar */
[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e3ecf1;}
.step {display: flex; gap: .7rem; margin-bottom: .7rem; font-size: .85rem; color: #1c2b33;}
.step span {background: #2b7a9e; color: #fff; width: 24px; height: 24px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; font-size: .75rem; flex-shrink: 0;}
.dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: .5rem;}
.cls {font-size: .82rem; margin: .35rem 0; color: #1c2b33;}

/* ---- readability fixes: force light theme colours, even if the browser/Streamlit is in dark mode ---- */
:root {color-scheme: light;}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {color: #1c2b33 !important;}
[data-testid="stSidebar"] h3 {font-weight: 700; color: #1f6f8b !important;}
[data-testid="stSidebar"] .step span {color: #fff !important;}

[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label p {
    color: #1c2b33 !important; font-weight: 600; font-size: 1rem;}
[data-testid="stFileUploaderDropzoneInstructions"], [data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"] {color: #3d4f59 !important;}
[data-testid="stFileUploader"] button {background: #2b7a9e !important; color: #fff !important;
    border: none !important; border-radius: 10px; font-weight: 600;}
[data-testid="stFileUploader"] button:hover {background: #1f6f8b !important;}
[data-testid="stFileUploader"] button * {color: #fff !important;}

[data-testid="stImageCaption"], [data-testid="stImageCaption"] * {color: #3d4f59 !important;}
[data-testid="stSpinner"], [data-testid="stSpinner"] * {color: #1c2b33 !important;}
.stAppDeployButton, [data-testid="stToolbar"] {display: none !important;}

/* keep hero text white */
.hero, .hero h1, .hero p, .hero .chip {color: #fff !important;}
.card, .card .name, .card .n, .card .p {color: #1c2b33 !important;}
.card h4 {color: #1f6f8b !important;}
.result .label, .result .desc, .conf {color: #4a5b65 !important;}
.warn {color: #7a4b00 !important;} .disclaimer, .disclaimer b {color: #7a1f1f !important;}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource
def load():
    model = tf.keras.models.load_model("model/skin_model.keras")
    classes = json.load(open("model/classes.json"))
    return model, classes


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### How it works")
    st.markdown("""
<div class="step"><span>1</span><div>Upload a close-up photo of a skin lesion.</div></div>
<div class="step"><span>2</span><div>The image is resized to 224×224 pixels.</div></div>
<div class="step"><span>3</span><div>MobileNetV2 extracts visual features.</div></div>
<div class="step"><span>4</span><div>A classifier outputs a probability for each of 7 lesion types.</div></div>
""", unsafe_allow_html=True)
    st.markdown("### Lesion types")
    st.markdown("".join(
        f'<div class="cls"><span class="dot" style="background:{c}"></span>{n}</div>'
        for n, _, c in INFO.values()), unsafe_allow_html=True)
    st.caption("Trained on the HAM10000 dataset (about 10,000 dermatoscopic images).")

# ---------- hero ----------
st.markdown("""
<div class="hero">
<h1>🔬 Skin Lesion Classifier</h1>
<p>Upload an image and an AI model estimates which of 7 skin lesion types it most resembles.</p>
<div class="chips"><span class="chip">Supervised learning</span><span class="chip">Transfer learning</span>
<span class="chip">MobileNetV2</span><span class="chip">7 classes</span></div>
</div>""", unsafe_allow_html=True)

try:
    model, classes = load()
except Exception:
    st.error("No trained model found. Run `python train.py` first, and start the app from the project folder.")
    st.stop()

file = st.file_uploader("Drag and drop an image here, or browse (JPG or PNG)", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    arr = np.array(img.resize((224, 224)), dtype="float32")[None]
    with st.spinner("Analysing image..."):
        probs = model.predict(arr, verbose=0)[0]
    top = int(probs.argmax())
    name, desc, color = INFO[classes[top]]
    conf = probs[top] * 100

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with right:
        st.markdown(f"""
<div class="card result" style="--c:{color}">
<div class="label">Most likely match</div>
<div class="name">{name}</div>
<div class="desc">{desc}</div>
<div class="meter"><div style="width:{conf:.1f}%"></div></div>
<div class="conf">Confidence: <b>{conf:.1f}%</b></div>
</div>""", unsafe_allow_html=True)
        if conf < 50:
            st.markdown('<div class="warn">⚠️ The model is not very confident about this image. '
                        'Try a sharper, well-lit, close-up photo.</div>', unsafe_allow_html=True)

    order = np.argsort(probs)[::-1]
    rows = "".join(
        f'<div class="row"><div class="n">{INFO[classes[i]][0]}</div>'
        f'<div class="bar"><div style="width:{probs[i]*100:.1f}%;background:{INFO[classes[i]][2]}"></div></div>'
        f'<div class="p">{probs[i]*100:.1f}%</div></div>' for i in order)
    st.markdown(f'<div class="card"><h4>Probability for every class</h4>{rows}</div>', unsafe_allow_html=True)

st.markdown("""<div class="disclaimer"><b>Educational project only.</b> This tool is not a medical device and
cannot diagnose any condition. It can be wrong, especially on photos that differ from its training data.
Please see a dermatologist about any skin concern.</div>""", unsafe_allow_html=True)
