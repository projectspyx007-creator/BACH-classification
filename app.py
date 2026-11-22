from pathlib import Path
import io, json, os, sys
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]

st.set_page_config(page_title="BACH Research Studio", page_icon="🔬", layout="wide")

@st.cache_resource(show_spinner="Loading the research models...")
def load_models():
    extractor = tf.keras.models.load_model(ASSETS / "efficientnetb0_feature_extractor.keras", compile=False)
    classifier = tf.keras.models.load_model(ASSETS / "bach_hybrid_global_local.keras", compile=False)
    attention_model = tf.keras.Model(
        inputs=classifier.inputs,
        outputs=classifier.get_layer("hybrid_attention_weights").output
    )
    return extractor, classifier, attention_model

@st.cache_data
def load_metadata():
    manifest = pd.read_csv(ASSETS / "dataset_manifest.csv")
    with open(ASSETS / "evaluation/hybrid_internal_test_metrics.json") as f:
        metrics = json.load(f)
    return manifest, metrics

def prepare_inputs(image_bytes, extractor):
    """Recreate the verified global-JPEG and original-TIFF patch feature paths."""
    original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if original.size != (2048, 1536):
        st.warning(f"Uploaded image dimensions are {original.size}; the BACH originals are typically 2048×1536. The same crop coordinates will be applied.")

    # Global branch: resize original to 224x224, encode as JPEG, then read it back.
    global_resized = original.resize((224, 224), Image.Resampling.BILINEAR)
    jpeg_buffer = io.BytesIO()
    global_resized.save(jpeg_buffer, format="JPEG", quality=95)
    jpeg_buffer.seek(0)
    global_image = Image.open(jpeg_buffer).convert("RGB")
    global_array = np.asarray(global_image, dtype=np.float32)

    # Local branch: 88 crops from the original TIFF using the saved coordinates.
    coordinates = np.load(ASSETS / "patch_coordinates.npy")
    original_array = np.asarray(original)
    patches = []
    for x, y in coordinates:
        patch = original_array[int(y):int(y)+224, int(x):int(x)+224]
        if patch.shape[:2] != (224, 224):
            raise ValueError(f"Patch at ({x},{y}) is not 224×224. Check uploaded image dimensions.")
        patches.append(patch)
    patch_array = np.asarray(patches, dtype=np.float32)

    global_features = extractor.predict(global_array[None, ...], verbose=0)
    local_features = extractor.predict(patch_array, batch_size=16, verbose=0)
    local_features = local_features[None, ...]
    return original, global_features, local_features

def make_attention_overlay(original, weights):
    coords = np.load(ASSETS / "patch_coordinates.npy")
    weights = np.asarray(weights).reshape(-1)
    heat = np.zeros((original.height, original.width), dtype=np.float32)
    count = np.zeros_like(heat)
    for (x, y), weight in zip(coords, weights):
        x, y = int(x), int(y)
        heat[y:y+224, x:x+224] += float(weight)
        count[y:y+224, x:x+224] += 1
    valid = count > 0
    heat[valid] /= count[valid]
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    heat_img = Image.fromarray(np.uint8(heat * 255)).resize(original.size, Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
    import matplotlib.cm as cm
    colored = (cm.get_cmap("jet")(heat_arr)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(original.resize((original.width, original.height)), dtype=np.float32)
    overlay = (0.58 * base + 0.42 * colored).clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay), heat_img

manifest, metrics = load_metadata()

with st.sidebar:
    st.title("🔬 BACH Research Studio")
    st.caption("Breast Histopathology Classification")
    page = st.radio("Navigate", ["Overview", "Live prediction", "Dataset explorer", "Performance", "Experiments", "Methodology"])
    st.divider()
    st.caption("Local research demonstration · TensorFlow")

if page == "Overview":
    st.title("BACH Research Studio")
    st.write("An interactive demonstration of a hybrid global–local attention multiple-instance learning model for breast histopathology image classification.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Labeled images", "400")
    c2.metric("Classes", "4")
    c3.metric("Internal-test accuracy", f"{metrics['accuracy']*100:.2f}%")
    c4.metric("Test images", str(metrics["test_images"]))
    st.subheader("The four image categories")
    cols = st.columns(4)
    for col, cls in zip(cols, CLASSES):
        files = sorted((ASSETS / "sample_images").glob(f"{cls}_*.jpg"))
        with col:
            st.markdown(f"**{cls}**")
            if files: st.image(str(files[0]), use_container_width=True)
    st.subheader("How the model works")
    st.markdown("""
    1. **Global view:** the processed 224×224 image is encoded into a 1,280-dimensional feature vector.
    2. **Local view:** 88 overlapping 224×224 patches are extracted from the original TIFF and encoded into local feature vectors.
    3. **Hybrid attention model:** global and attention-pooled local representations are fused to predict one of four classes.
    """)
    st.info("Research prototype only. It is not a medical device and must not be used for diagnosis or treatment decisions.")

elif page == "Live prediction":
    st.title("Live image prediction")
    st.write("Upload a BACH-style histopathology TIFF image. Processing runs locally on this Mac.")
    uploaded = st.file_uploader("Choose a TIFF image", type=["tif", "tiff"])
    if uploaded:
        raw = uploaded.getvalue()
        try:
            preview = Image.open(io.BytesIO(raw)).convert("RGB")
            st.image(preview, caption=f"Uploaded image · {preview.size[0]} × {preview.size[1]}", use_container_width=True)
            if st.button("Run classification", type="primary"):
                extractor, classifier, attention_model = load_models()
                with st.spinner("Extracting global and local features and running the classifier..."):
                    original, gf, lf = prepare_inputs(raw, extractor)
                    probs = classifier.predict([gf, lf], verbose=0)[0]
                    attn = attention_model.predict([gf, lf], verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                st.success(f"Model prediction: {CLASSES[pred_idx]}")
                left,right = st.columns([1,1])
                with left:
                    st.subheader("Class probabilities")
                    prob_df = pd.DataFrame({"Class": CLASSES, "Probability": probs})
                    fig = px.bar(prob_df, x="Class", y="Probability", range_y=[0,1], text=prob_df["Probability"].map(lambda x:f"{x:.1%}"))
                    fig.update_layout(yaxis_tickformat=".0%", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    st.subheader("Patch attention")
                    overlay, heat = make_attention_overlay(original, attn)
                    st.image(overlay, caption="Attention-weight overlay (relative model attention, not a diagnostic segmentation)", use_container_width=True)
                st.subheader("Attention-ranked patches")
                coords = np.load(ASSETS / "patch_coordinates.npy")
                weights = np.asarray(attn).reshape(-1)
                top = np.argsort(weights)[::-1][:8]
                patch_cols = st.columns(4)
                arr = np.asarray(original)
                for j, idx in enumerate(top):
                    x,y = map(int, coords[idx])
                    crop = Image.fromarray(arr[y:y+224, x:x+224])
                    with patch_cols[j%4]:
                        st.image(crop, caption=f"Patch {idx+1} · attention {weights[idx]:.4f}", use_container_width=True)
                st.caption("Predicted probabilities are model outputs, not calibrated clinical probabilities. Attention highlights model weighting, not causal evidence.")
        except Exception as e:
            st.error(f"Could not process this image: {e}")

elif page == "Dataset explorer":
    st.title("Dataset explorer")
    a,b,c = st.columns(3)
    a.metric("Total labeled images", len(manifest))
    b.metric("Training", int((manifest.split=="train").sum()))
    c.metric("Validation + internal test", int(manifest.split.isin(["validation","internal_test"]).sum()))
    counts = manifest.groupby(["split","label"]).size().reset_index(name="Images")
    fig = px.bar(counts, x="split", y="Images", color="label", barmode="group", category_orders={"label":CLASSES,"split":["train","validation","internal_test"]})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Manifest")
    st.dataframe(manifest[["filename","label","patient_id","group_id","split"]], use_container_width=True, hide_index=True)
    st.caption("The original 400 labeled images are represented in this manifest. The separate 100-image unlabeled Kaggle test set is not included in labeled accuracy calculations.")

elif page == "Performance":
    st.title("Model performance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']*100:.2f}%")
    c3.metric("Macro F1", f"{metrics['macro_f1']*100:.2f}%")
    c4.metric("Weighted F1", f"{metrics['weighted_f1']*100:.2f}%")
    st.caption(f"Internal test: {metrics['correct']}/{metrics['test_images']} correct · 95% Wilson CI: {metrics['wilson_95_ci'][0]*100:.2f}%–{metrics['wilson_95_ci'][1]*100:.2f}%")
    cm = np.array(metrics["confusion_matrix"])
    fig = px.imshow(cm, x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Images"))
    fig.update_layout(title="Internal-test confusion matrix")
    st.plotly_chart(fig, use_container_width=True)
    report = metrics["classification_report"]
    rows=[]
    for cls in CLASSES:
        r=report[cls]
        rows.append({"Class":cls,"Precision":r["precision"],"Recall":r["recall"],"F1":r["f1-score"],"Support":int(r["support"])})
    st.subheader("Per-class results")
    st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1":"{:.3f}"}), use_container_width=True, hide_index=True)
    st.warning("These results are from one 60-image patient-grouped internal-test split. 255 images lacked patient IDs, so complete patient independence cannot be guaranteed. This is not evidence of clinical performance.")

elif page == "Experiments":
    st.title("Experiments & research history")
    st.write("The project retained exploratory baselines, attention MIL, the selected hybrid model, cost-sensitive loss experiments and a multi-scale attempt.")
    files = sorted((ASSETS/"training_logs").glob("*.csv"))
    if files:
        options={p.stem:p for p in files}
        chosen=st.selectbox("Choose an experiment log", list(options))
        df=pd.read_csv(options[chosen])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download this CSV", df.to_csv(index=False).encode(), file_name=options[chosen].name, mime="text/csv")
    st.subheader("Saved attention visualizations")
    images=sorted((ASSETS/"attention_visualizations").glob("*.png"))
    if images:
        cols=st.columns(3)
        for i,p in enumerate(images):
            with cols[i%3]: st.image(str(p), caption=p.stem, use_container_width=True)

elif page == "Methodology":
    st.title("Methodology & limitations")
    st.subheader("Dataset")
    st.write("400 labeled BACH microscopy images across Benign, InSitu, Invasive and Normal. The original dataset contains 100 images per class.")
    st.subheader("Patient-grouped split")
    split_counts=manifest.groupby(["split","label"]).size().unstack(fill_value=0).reindex(columns=CLASSES)
    st.dataframe(split_counts, use_container_width=True)
    st.markdown("""
    - Training: 280 images
    - Validation: 60 images
    - Internal test: 60 images
    - 145 images had patient IDs; 255 did not.
    - 29 identified patients were found, and the original random split had identified patients crossing multiple splits.
    - The final grouped split prevents overlap among known patient groups, but unique synthetic groups for missing IDs cannot guarantee that all unidentified images belong to different patients.
    """)
    st.subheader("Architecture")
    st.write("ImageNet EfficientNetB0 extracts 1,280-dimensional global and patch-level features. An attention mechanism weights the 88 local patch representations. The global and local representations are fused and classified into four categories.")
    st.subheader("Evaluation boundaries")
    st.write("The separate 100-image Kaggle test set is unlabeled and is not used to calculate accuracy. The reported internal-test results apply only to the saved 60-image split.")
    st.error("This application is an educational research prototype, not a clinical diagnostic system. Never use its output to make patient-care decisions.")

st.sidebar.divider()
st.sidebar.caption(f"TensorFlow {tf.__version__} · Runs on this computer")
