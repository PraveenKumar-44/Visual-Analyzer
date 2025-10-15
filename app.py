import streamlit as st
from PIL import Image
import numpy as np
import io, os
import torch
import torchvision.transforms as T
import torchvision.models as models
from model import VisualMatcher
import base64
from urllib.request import urlopen

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# --- ✨ Modern Glassmorphism + Neon Aesthetic ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

body {
  background: radial-gradient(circle at top left, #b2fefa, #0ed2f7);
  font-family: "Poppins", sans-serif;
  color: #222;
}

.main {
  padding: 2.5rem 3rem;
}

.header {
  text-align: center;
  margin-bottom: 2rem;
}

.header h1 {
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(90deg, #06beb6, #48b1bf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

.header p {
  font-size: 1.05rem;
  color: #333;
  margin-top: 0.5rem;
}

.stSidebar {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(14px);
  border-radius: 12px;
}

.stButton > button {
  background: linear-gradient(90deg, #48b1bf, #06beb6);
  color: white;
  border: none;
  border-radius: 10px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  transition: all 0.25s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
.stButton > button:hover {
  transform: translateY(-3px);
  background: linear-gradient(90deg, #06beb6, #48b1bf);
}

.card {
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.5);
  backdrop-filter: blur(15px);
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
  overflow: hidden;
  transition: all 0.3s ease;
  margin-bottom: 15px;
}
.card:hover {
  transform: translateY(-8px);
  box-shadow: 0 16px 35px rgba(0,0,0,0.15);
}

.match-img {
  width: 100%;
  height: 220px;
  object-fit: cover;
  border-radius: 14px;
}

.match-info {
  text-align: center;
  padding: 10px 6px;
}
.match-info h4 {
  font-size: 17px;
  font-weight: 700;
  margin-bottom: 3px;
  color: #222;
}
.match-info p {
  font-size: 13px;
  color: #555;
}

hr {
  border: none;
  border-top: 2px solid rgba(255,255,255,0.3);
  margin: 1.5rem 0;
}

.upload-box {
  border: 2px dashed rgba(0,0,0,0.2);
  border-radius: 14px;
  padding: 1.5rem;
  text-align: center;
  background: rgba(255,255,255,0.4);
  transition: 0.3s;
}
.upload-box:hover {
  background: rgba(255,255,255,0.6);
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
  <h1>🧠 Visual Product Matcher</h1>
  <p>Find visually similar products instantly using AI-powered image matching.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Settings")
K = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Upload or paste an image URL to search your catalog visually.")

# Helper
def b64_encode(b):
    return base64.b64encode(b).decode('utf-8')

# --- Load Matcher ---
@st.cache_resource
def load_matcher(use_gpu=False):
    device = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    matcher = VisualMatcher(embeddings_path='embeddings.npy', metadata_path='metadata.json')
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = T.Compose([
        T.Resize((224,224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return matcher, model, transform, device

try:
    matcher, model, transform, device = load_matcher(use_gpu=use_gpu)
except Exception as e:
    st.error(f"⚠️ Could not load embeddings. Please ensure `embeddings.npy` and `metadata.json` exist.\n\nError: {e}")
    st.stop()

# --- Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 Input Image")
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'webp'])
    st.markdown('</div>', unsafe_allow_html=True)
    input_image = None

    if uploaded:
        input_image = Image.open(uploaded).convert('RGB')

    url_input = st.text_input("Or paste an image URL")
    if url_input:
        try:
            img_bytes = urlopen(url_input).read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            st.error(f"❌ Could not load image: {e}")

    if input_image is not None:
        st.image(input_image, caption="Query Image", use_container_width=True)

with col2:
    st.subheader("🔍 Matching Results")
    if input_image is None:
        st.info("Upload or enter an image URL to start matching.")
    else:
        img_t = transform(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model(img_t).cpu().numpy().reshape(-1)
        results = matcher.query(q_emb, topk=K)

        cols = st.columns(3)
        for i, res in enumerate(results):
            col = cols[i % 3]
            file = res['file']
            score = res['score']
            try:
                img = Image.open(file).convert('RGB')
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                b = buf.getvalue()
                with col:
                    st.markdown(
                        f"""
                        <div class='card'>
                            <img class='match-img' src='data:image/jpeg;base64,{b64_encode(b)}'>
                            <div class='match-info'>
                                <h4>Match #{i+1}</h4>
                                <p>Similarity Score: {score:.4f}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
