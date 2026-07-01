"""Streamlit UI for interactively testing the Corn Leaf Disease classifiers.

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

import config
from models import get_model
from utils.transforms import get_eval_transforms

st.set_page_config(page_title="Corn Leaf Disease Classifier", layout="wide")

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

.app-header {
    background: linear-gradient(120deg, #1B5E20 0%, #2E7D32 55%, #558B2F 100%);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 18px rgba(27, 94, 32, 0.25);
}
.app-header h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #E8F5E9;
    margin: 0.5rem 0 0 0;
    font-size: 1.02rem;
}

section[data-testid="stSidebar"] {
    background-color: #F7FAF7;
    border-right: 1px solid #E0E8E0;
}
section[data-testid="stSidebar"] h2 {
    color: #1B5E20;
}

div[data-testid="stMetric"] {
    background-color: #F1F8F2;
    border: 1px solid #DCEDC8;
    border-left: 4px solid #2E7D32;
    border-radius: 10px;
    padding: 0.9rem 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #E8F5E9;
    color: #1B5E20 !important;
}

div[data-testid="stStatusWidget"], .stAlert {
    border-radius: 10px;
}

.class-list-item {
    padding: 0.25rem 0;
    color: #33413A;
    font-size: 0.92rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str, checkpoint_path: str):
    device = config.DEVICE
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def available_checkpoints(model_name: str) -> list[Path]:
    return sorted(config.CHECKPOINT_DIR.glob(f"{model_name}_*.pth"))


def predict(model, image: Image.Image, device: torch.device):
    transform = get_eval_transforms()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    start = time.time()
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    elapsed_ms = (time.time() - start) * 1000
    return probs, elapsed_ms


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Model Settings")
st.sidebar.markdown(f"**Device:** `{config.DEVICE}`")

model_choice = st.sidebar.selectbox("Model architecture", ["resnet18", "baseline"], index=0)
ckpts = available_checkpoints(model_choice)

if not ckpts:
    st.sidebar.error(
        f"No checkpoints found for '{model_choice}' in checkpoints/. "
        f"Train it first: `python train.py --model {model_choice}`"
    )
    model, checkpoint = None, None
else:
    ckpt_labels = [c.name for c in ckpts]
    default_idx = ckpt_labels.index(f"{model_choice}_best.pth") if f"{model_choice}_best.pth" in ckpt_labels else 0
    ckpt_choice = st.sidebar.selectbox("Checkpoint", ckpt_labels, index=default_idx)
    ckpt_path = config.CHECKPOINT_DIR / ckpt_choice
    with st.spinner("Loading model..."):
        model, checkpoint = load_model(model_choice, str(ckpt_path))
    st.sidebar.success(f"Loaded epoch {checkpoint.get('epoch')} | val_acc={checkpoint.get('val_acc'):.4f}")

st.sidebar.markdown("---")
st.sidebar.caption("CLASSES")
for short in config.CLASS_SHORT_NAMES:
    st.sidebar.markdown(f'<div class="class-list-item">— {short}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Corn Leaf Disease Classification</h1>
        <p>Deep learning pipeline for classifying maize leaves as healthy or affected by
        Gray Leaf Spot, Common Rust, or Northern Leaf Blight.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_performance, tab_curves = st.tabs(
    ["Single Image", "Batch Prediction", "Model Performance", "Training Curves"]
)

# --- Single image prediction -----------------------------------------------
with tab_single:
    st.subheader("Predict a single leaf image")
    uploaded = st.file_uploader("Upload a corn leaf image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None and model is not None:
        image = Image.open(uploaded)
        col1, col2 = st.columns([1, 1.4])
        with col1:
            st.image(image, caption=uploaded.name, use_container_width=True)

        probs, elapsed_ms = predict(model, image, config.DEVICE)
        pred_idx = int(probs.argmax())

        with col2:
            st.metric("Predicted class", config.CLASS_SHORT_NAMES[pred_idx], f"{probs[pred_idx]*100:.2f}% confidence")
            st.caption(f"Inference time: {elapsed_ms:.1f} ms on {config.DEVICE}")

            prob_df = pd.DataFrame({
                "Class": config.CLASS_SHORT_NAMES,
                "Probability": probs,
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("Class"), color="#2E7D32")
            st.dataframe(prob_df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)
    elif model is None:
        st.info("Select a model with an available checkpoint from the sidebar.")
    else:
        st.info("Upload an image to get a prediction.")

# --- Batch prediction --------------------------------------------------------
with tab_batch:
    st.subheader("Predict multiple leaf images at once")
    uploaded_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True
    )

    if uploaded_files and model is not None:
        rows = []
        progress = st.progress(0.0)
        for i, file in enumerate(uploaded_files):
            image = Image.open(file)
            probs, elapsed_ms = predict(model, image, config.DEVICE)
            pred_idx = int(probs.argmax())
            rows.append({
                "filename": file.name,
                "predicted_class": config.CLASS_SHORT_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "inference_time_ms": round(elapsed_ms, 2),
                **{f"prob_{c}": round(float(p), 4) for c, p in zip(config.CLASS_SHORT_NAMES, probs)},
            })
            progress.progress((i + 1) / len(uploaded_files))

        results_df = pd.DataFrame(rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.bar_chart(results_df["predicted_class"].value_counts(), color="#2E7D32")

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions as CSV", csv_bytes, "predictions.csv", "text/csv")
    elif model is None:
        st.info("Select a model with an available checkpoint from the sidebar.")
    else:
        st.info("Upload one or more images to run batch prediction.")

# --- Model performance --------------------------------------------------------
with tab_performance:
    st.subheader("Evaluation results (generated by evaluate.py)")

    perf_model = st.selectbox("Model", ["resnet18", "baseline"], key="perf_model")
    results_dir = config.model_results_dir(perf_model)
    plots_dir = config.model_plots_dir(perf_model)
    metrics_path = results_dir / "metrics.json"

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
        cols[1].metric("F1 (macro)", f"{metrics['f1_macro']:.2%}")
        cols[2].metric("ROC-AUC (macro)", f"{metrics.get('roc_auc_macro', float('nan')):.3f}")
        cols[3].metric("Avg. inference", f"{metrics.get('avg_inference_time_ms', 0):.1f} ms")

        col1, col2 = st.columns(2)
        cm_path = plots_dir / "confusion_matrix.png"
        roc_path = plots_dir / "roc_curve.png"
        if cm_path.exists():
            col1.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
        if roc_path.exists():
            col2.image(str(roc_path), caption="ROC Curve", use_container_width=True)

        pr_path = plots_dir / "precision_recall_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-Recall Curve", use_container_width=True)

        report_path = results_dir / "classification_report.txt"
        if report_path.exists():
            st.text("Classification report")
            st.code(report_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"No evaluation results yet for '{perf_model}'. Run: `python evaluate.py --model {perf_model}`")

    comparison_path = config.RESULTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Baseline vs. Advanced model comparison")
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)
        comparison_plot = config.PLOTS_DIR / "model_comparison.png"
        if comparison_plot.exists():
            st.image(str(comparison_plot), use_container_width=True)

# --- Training curves --------------------------------------------------------
with tab_curves:
    st.subheader("Training / validation curves")
    curves_model = st.selectbox("Model", ["resnet18", "baseline"], key="curves_model")
    curves_path = config.model_plots_dir(curves_model) / "training_curves.png"
    if curves_path.exists():
        st.image(str(curves_path), use_container_width=True)
    else:
        st.warning(f"No training curves yet for '{curves_model}'. Run: `python train.py --model {curves_model}`")

    dist_path = config.PLOTS_DIR / "class_distribution.png"
    if dist_path.exists():
        st.subheader("Dataset class distribution")
        st.image(str(dist_path), use_container_width=True)
