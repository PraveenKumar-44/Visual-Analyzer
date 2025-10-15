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

# --- CSS for modern styling ---
st.markdown(
    """
    <style>
    /* Main header */
    .header {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(90deg,#ff7a18,#af002d 50%,#319197 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: hue 6s infinite linear;
        margin-bottom: 10px;
    }
    @keyframes hue {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    /* Subtitle */
    .subheader {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }
    /* Sidebar */
    .sidebar .stSlider>div>div>div>div {
        color: #000 !important;
    }
    /* Result cards */
    .card {
        border-radius: 12px;
        padding: 10px;
        margin: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        transition: transform .25s ease, box-shadow .25s ease;
        background-color: #fff;
    }
    .card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.18);
    }
    .match-img {
        width: 100%;
        height: 180px;
        object-fit: contain;
        border-radius: 8px;
        background: #f9f9f9;
        padding: 5px;
    }
    .title {
        font-size:18px;
        font-weight:600;
        margin-top: 6px;
        margin-bottom:4px;
    }
    .subtext {
        color: #666;
        font-size:13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<div class='header'>Visual Product Matcher</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Find visually similar products in your catalog with AI-powered embeddings.</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Settings")
K = st.sidebar.slider("Number of results", min_value=1, max_value=12, value=6)
use_gpu = st.sidebar.checkbox("Use GPU if available", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("Upload an image or enter a URL to find similar products.")

# --- helper for base64 embedding ---
def b64_encode(b):
    return base64.b64encode(b).decode('utf-8')

# Load matcher
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
    st.error(
        "Could not load embeddings. Make sure embeddings.npy and metadata.json are in the working directory. "
        f"Error: {e}"
    )
    st.stop()

# Input section
col1, col2 = st.columns([1,2])
with col1:
    st.markdown("### Input Image")
    uploaded = st.file_uploader("Upload an image", type=['jpg','jpeg','png','webp'])
    input_image = None
    if uploaded is not None:
        input_image = Image.open(uploaded).convert('RGB')

    url_input = st.text_input("Or enter image URL")
    if url_input:
        try:
            img_bytes = urlopen(url_input).read()
            input_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

    if input_image is not None:
        st.image(input_image, caption="Query Image", use_container_width=True)

# Results section
with col2:
    st.markdown("### Matches")
    if input_image is None:
        st.info("Upload an image or enter a URL to see matches.")
    else:
        img_t = transform(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            q_emb = model(img_t).cpu().numpy().reshape(-1)
        results = matcher.query(q_emb, topk=K)

        # dynamic grid layout
        ncols = 3
        for i in range(0, len(results), ncols):
            cols = st.columns(ncols)
            for j, res in enumerate(results[i:i+ncols]):
                col = cols[j]
                file = res['file']
                score = res['score']
                try:
                    img = Image.open(file).convert('RGB')
                    buf = io.BytesIO()
                    img.save(buf, format='JPEG')
                    b = buf.getvalue()
                    with col:
                        st.markdown(
                            f"<div class='card'>"
                            f"<img class='match-img' src='data:image/jpeg;base64,{b64_encode(b)}' />"
                            f"<div class='title'>Match #{i+j+1}</div>"
                            f"<div class='subtext'>Score: {score:.4f}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.write(f"Could not load {file}: {e}")
