import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from src.model import build_model
from src.config import cfg


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🫁",
    layout="wide",
)


# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_model():
    ckpt = torch.load(cfg.model_dir / "best_model.pt", map_location=cfg.device)
    classes = ckpt["classes"]
    model, target_layer_name = build_model(
        ckpt["backbone"],
        num_classes=len(classes),
        pretrained=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()
    return model, classes, ckpt["img_size"], target_layer_name


def preprocess(img: Image.Image, img_size: int):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)


def fmt_label(name: str) -> str:
    return name.replace("_", " ").strip()


def confidence_band(conf: float) -> str:
    if conf >= 0.80:
        return "High"
    if conf >= 0.60:
        return "Moderate"
    return "Low"


def overlay_heatmap(original_rgb: Image.Image, heatmap_2d: np.ndarray, alpha: float = 0.40):
    """
    original_rgb: PIL Image (RGB)
    heatmap_2d: numpy array (H, W) values in [0,1]
    """
    original = np.array(original_rgb)
    h, w = original.shape[:2]

    heatmap_resized = cv2.resize(heatmap_2d, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(original_bgr, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)


@torch.no_grad()
def predict(model, x):
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs


def make_gradcam(model, target_layer, x, class_index: int):
    """
    Returns a 2D heatmap (numpy array) normalized to [0,1]
    """
    activations = None
    gradients = None

    def forward_hook(_, __, output):
        nonlocal activations
        activations = output

    def backward_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    logits = model(x)
    score = logits[:, class_index]

    model.zero_grad(set_to_none=True)
    score.backward()

    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])  # (C,)
    act = activations.squeeze(0)  # (C, H, W)

    for i in range(act.shape[0]):
        act[i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(act, dim=0)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    fh.remove()
    bh.remove()

    return heatmap.detach().cpu().numpy()


# -----------------------------
# Load model
# -----------------------------
model, classes, img_size, target_layer_name = load_model()
target_layer = dict([*model.named_modules()])[target_layer_name]


# -----------------------------
# UI
# -----------------------------
st.title("Chest X-ray Classification")
st.caption("Multi-class deep learning model for COVID-19, Viral Pneumonia, and Normal cases.")
st.divider()

col_input, col_output = st.columns([1.1, 0.9], gap="large")

with col_input:
    st.subheader("Input")
    uploaded = st.file_uploader(
        "Upload a chest X-ray image (PNG / JPG / JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    show_cam = st.toggle("Show Grad-CAM (Explainability)", value=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(
    img,
    caption="Uploaded chest X-ray",
    width=500
)
    else:
        st.info("Upload an image to run inference.")

with col_output:
    st.subheader("Output")

    if not uploaded:
        st.info("Awaiting an input image.")
    else:
        img = Image.open(uploaded).convert("RGB")
        x = preprocess(img, img_size).to(cfg.device)

        # --- inference with spinner (simple + professional) ---
        with st.spinner("Running inference..."):
            time.sleep(0.2)
            probs = predict(model, x).squeeze(0).detach().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = fmt_label(classes[pred_idx])
        conf = float(probs[pred_idx])

        st.success("Inference completed")

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred_label)
        m2.metric("Confidence", f"{conf * 100:.2f}%")
        m3.metric("Confidence Level", confidence_band(conf))

        st.divider()
        st.markdown("### Class Probabilities")

        order = np.argsort(probs)[::-1]
        for i in order:
            cls_name = fmt_label(classes[i])
            p = float(probs[i])

            st.write(f"**{cls_name}** — {p * 100:.2f}%")
            st.progress(min(max(p, 0.0), 1.0))

        # --- Grad-CAM ---
        if show_cam:
            st.divider()
            st.markdown("### Grad-CAM (Model Attention)")

            with st.spinner("Generating Grad-CAM..."):
                # Need gradients -> temporarily enable grad
                model.zero_grad(set_to_none=True)
                for param in model.parameters():
                    param.requires_grad_(True)

                heatmap = make_gradcam(model, target_layer, x, pred_idx)
                cam_img = overlay_heatmap(img.resize((img_size, img_size)), heatmap, alpha=0.40)

                # Turn gradients back off (clean)
                for param in model.parameters():
                    param.requires_grad_(False)

            st.image(cam_img,caption=f"Grad-CAM for: {pred_label}",width=500)

            # Optional: save button
            save_col1, save_col2 = st.columns([0.5, 0.5])
            with save_col1:
                if st.button("Save Grad-CAM to outputs/gradcam"):
                    out_dir = cfg.gradcam_dir
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"streamlit_gradcam_{pred_label.replace(' ', '_')}.png"
                    cam_img.save(out_path)
                    st.success(f"Saved: {out_path}")

