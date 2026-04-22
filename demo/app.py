"""
Wildlife Behavior Analyzer
Powered by X3D-KABR (WACV 2024) + YOLOv8
Author: rohansingh0 | HuggingFace Spaces
"""

import os
import re
import cv2
import time
import tempfile
import zipfile
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque, Counter
from huggingface_hub import hf_hub_download
from pytorchvideo.models.x3d import create_x3d
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 8
NUM_FRAMES   = 16
CROP_SIZE    = 300
MEAN         = [0.45, 0.45, 0.45]
STD          = [0.225, 0.225, 0.225]
ANIMAL_IDS   = set(range(14, 26))   # COCO animal class IDs

LABEL_NAMES  = ["Walk", "Trot", "Run", "Graze",
                "Browse", "Head Up", "Auto-Groom", "Occluded"]
LABEL_COLORS_HEX = [
    "#2196F3", "#9C27B0", "#F44336", "#4CAF50",
    "#009688", "#FF9800", "#E91E63", "#9E9E9E"
]
LABEL_COLORS_BGR = [
    (243, 150, 33), (176, 39, 156), (54, 67, 244), (80, 175, 76),
    (136, 150, 0),  (0, 152, 255),  (99, 30, 233), (158, 158, 158)
]

# ─────────────────────────────────────────────────────────────
# Model Loading (cached — runs once per session)
# ─────────────────────────────────────────────────────────────
def remap_key(k):
    """Remap SlowFast checkpoint keys → pytorchvideo format."""
    k = re.sub(r"^s1\.pathway0_stem\.conv_xy\b", "blocks.0.conv.conv_t",  k)
    k = re.sub(r"^s1\.pathway0_stem\.conv\b",    "blocks.0.conv.conv_xy", k)
    k = re.sub(r"^s1\.pathway0_stem\.bn\b",      "blocks.0.norm",         k)
    def stage(m):
        return f"blocks.{int(m.group(1))-1}.res_blocks.{int(m.group(2))}."
    k = re.sub(r"^s(\d)\.pathway0_res(\d+)\.", stage, k)
    k = re.sub(r"\.branch1\.weight$",  ".branch1_conv.weight", k)
    k = re.sub(r"\.branch1_bn\.",      ".branch1_norm.",        k)
    k = re.sub(r"\.branch2\.a\.weight$",   ".branch2.conv_a.weight",       k)
    k = re.sub(r"\.branch2\.a_bn\.",       ".branch2.norm_a.",              k)
    k = re.sub(r"\.branch2\.b\.weight$",   ".branch2.conv_b.weight",        k)
    k = re.sub(r"\.branch2\.b_bn\.",       ".branch2.norm_b.0.",            k)
    k = re.sub(r"\.branch2\.se\.fc1\.",    ".branch2.norm_b.1.block.0.",    k)
    k = re.sub(r"\.branch2\.se\.fc2\.",    ".branch2.norm_b.1.block.2.",    k)
    k = re.sub(r"\.branch2\.c\.weight$",   ".branch2.conv_c.weight",        k)
    k = re.sub(r"\.branch2\.c_bn\.",       ".branch2.norm_c.",              k)
    k = re.sub(r"^head\.conv_5\.",     "blocks.5.pool.pre_conv.",  k)
    k = re.sub(r"^head\.conv_5_bn\.", "blocks.5.pool.pre_norm.",  k)
    k = re.sub(r"^head\.lin_5\.",      "blocks.5.pool.post_conv.", k)
    k = re.sub(r"^head\.projection\.", "blocks.5.proj.",           k)
    return k


@st.cache_resource(show_spinner=False)
def load_models():
    """Download and load X3D-KABR + YOLOv8. Cached after first load."""
    # ── X3D-KABR ─────────────────────────────────────────────
    ckpt_zip = hf_hub_download(
        repo_id   = "imageomics/x3d-kabr-kinetics",
        filename  = "checkpoint_epoch_00075.pyth.zip",
        local_dir = "/tmp"
    )
    with zipfile.ZipFile(ckpt_zip, "r") as z:
        z.extractall("/tmp")

    ckpt_path = None
    for root, _, files in os.walk("/tmp"):
        for f in files:
            if f.endswith(".pyth"):
                ckpt_path = os.path.join(root, f)
                break

    model = create_x3d(
        input_clip_length=16, input_crop_size=300,
        model_num_class=NUM_CLASSES, depth_factor=5.0,
        width_factor=2.0, bottleneck_factor=2.25,
        dropout_rate=0.5, head_activation=None,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    model_sd   = model.state_dict()
    model_keys = set(model_sd.keys())

    remapped = {}
    for ck, cv in state_dict.items():
        mk = remap_key(ck)
        if mk in model_keys and model_sd[mk].shape == cv.shape:
            remapped[mk] = cv

    model.load_state_dict(remapped, strict=False)
    model = model.to(DEVICE).eval()

    # ── YOLOv8 ───────────────────────────────────────────────
    yolo = YOLO("yolov8n.pt")
    yolo.to("cpu")

    return model, yolo


# ─────────────────────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────────────────────
def preprocess_clip(frames_bgr):
    out = []
    for f in frames_bgr:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h, w = f.shape[:2]
        s = CROP_SIZE / min(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        ch = (f.shape[0] - CROP_SIZE) // 2
        cw = (f.shape[1] - CROP_SIZE) // 2
        f  = f[ch:ch + CROP_SIZE, cw:cw + CROP_SIZE]
        f  = (f.astype(np.float32) / 255. - MEAN) / STD
        out.append(f)
    clip = np.stack(out).transpose(3, 0, 1, 2)
    return torch.tensor(clip, dtype=torch.float32).unsqueeze(0)


def classify_behavior(model, crop_sequence):
    with torch.no_grad():
        x     = preprocess_clip(crop_sequence).to(DEVICE)
        out   = model(x)
        probs = torch.sigmoid(out).squeeze().cpu().numpy()
        pred  = int(np.argmax(probs[:7]))
    return pred, LABEL_NAMES[pred], probs


def detect_animals(yolo, frame_bgr, species_name):
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W    = frame_bgr.shape[:2]
    results = yolo(rgb, verbose=False, device="cpu", conf=0.15, iou=0.4)[0]
    animals = []
    for box in results.boxes:
        if int(box.cls[0]) not in ANIMAL_IDS:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        w_box, h_box  = x2 - x1, y2 - y1
        area_ratio    = (w_box * h_box) / (W * H)
        if area_ratio < 0.005 or area_ratio > 0.80:
            continue
        if w_box < 30 or h_box < 30:
            continue
        animals.append((x1, y1, x2, y2, species_name, conf))
    return animals


def smooth_predictions(timeline, window=5):
    smoothed     = []
    label_buffer = deque(maxlen=window)
    for t, label, conf in timeline:
        label_buffer.append(label)
        majority = Counter(label_buffer).most_common(1)[0][0]
        smoothed.append((t, majority, conf))
    return smoothed


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────
def make_time_budget_chart(counts, title):
    labels = [l for l in LABEL_NAMES[:7] if counts.get(l, 0) > 0]
    sizes  = [counts[l] for l in labels]
    colors = [LABEL_COLORS_HEX[LABEL_NAMES.index(l)] for l in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors,
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 11})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def make_timeline_chart(smoothed, video_name):
    label_to_idx = {l: i for i, l in enumerate(LABEL_NAMES)}
    times = [t for t, _, _ in smoothed]
    if not times:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5),
                                    gridspec_kw={"height_ratios": [3, 1]})
    for i, lbl in enumerate(LABEL_NAMES[:7]):
        pts = [(t, c) for t, l, c in smoothed if l == lbl]
        if pts:
            ts, cs = zip(*pts)
            ax1.scatter(ts, cs, label=lbl,
                       color=LABEL_COLORS_HEX[i], s=8, alpha=0.7)

    ax1.set_ylabel("Confidence", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=8, ncol=4)
    ax1.set_title(f"Behavior Timeline — {video_name}",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    dt = (times[1] - times[0]) if len(times) > 1 else 0.1
    for t, lbl, _ in smoothed:
        idx = label_to_idx.get(lbl, 7)
        ax2.axvspan(t - dt/2, t + dt/2,
                    color=LABEL_COLORS_HEX[idx], alpha=0.9)

    patches = [mpatches.Patch(color=LABEL_COLORS_HEX[i], label=l)
               for i, l in enumerate(LABEL_NAMES[:7])]
    ax2.legend(handles=patches, loc="upper right", fontsize=8, ncol=4)
    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_yticks([])
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────────────────────
def process_video(model, yolo, video_path, species_name,
                  frame_skip, progress_bar, status_text, preview_slot):
    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = os.path.join(tempfile.gettempdir(), "output_annotated.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    buffers        = {}
    all_preds      = []
    timeline       = []
    frame_idx      = 0
    last_annotated = None
    preview_every  = max(1, total_f // 20)   # update preview ~20 times

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            annotated = frame.copy()
            animals   = detect_animals(yolo, frame, species_name)
            timestamp = frame_idx / fps

            for idx, (x1, y1, x2, y2, name, _) in enumerate(animals):
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

                if idx not in buffers:
                    buffers[idx] = deque(maxlen=NUM_FRAMES)
                buffers[idx].append(crop)

                if len(buffers[idx]) == NUM_FRAMES:
                    pred_idx, pred_label, probs = classify_behavior(
                        model, list(buffers[idx]))
                    conf = float(probs[pred_idx])
                    all_preds.append(pred_label)
                    timeline.append((timestamp, pred_label, conf))
                else:
                    pred_label = "Buffering..."
                    pred_idx   = 7
                    conf       = 0.0

                color = LABEL_COLORS_BGR[pred_idx]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                txt = f"{name}: {pred_label} ({conf:.2f})"
                fs  = max(0.4, min(0.6, (x2 - x1) / 400))
                (tw, th), _ = cv2.getTextSize(
                    txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
                cv2.rectangle(annotated,
                              (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                cv2.putText(annotated, txt, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)

            cv2.putText(annotated,
                        f"t={timestamp:.1f}s | {len(animals)} detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            last_annotated = annotated

            # Live preview update
            if frame_idx % preview_every == 0:
                rgb_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_slot.image(rgb_preview, use_column_width=True,
                                   caption=f"Live preview — t={timestamp:.1f}s")

        writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1
        progress_bar.progress(min(frame_idx / total_f, 1.0))
        status_text.text(
            f"Processing frame {frame_idx}/{total_f} "
            f"({frame_idx/total_f*100:.0f}%) — "
            f"{len(all_preds)} classifications"
        )

    cap.release()
    writer.release()
    return out_path, Counter(all_preds), timeline


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Wildlife Behavior Analyzer",
        page_icon="🦒",
        layout="wide"
    )

    # ── Header ────────────────────────────────────────────────
    st.title("🦒 Wildlife Behavior Analyzer")
    st.markdown(
        "**AI-powered animal behavior recognition from drone footage**  \n"
        "Powered by [X3D-KABR](https://huggingface.co/imageomics/x3d-kabr-kinetics) "
        "(WACV 2024) + YOLOv8  \n"
        "Detects: **Walk · Trot · Run · Graze · Browse · Head Up · Auto-Groom**"
    )
    st.divider()

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        species_name = st.selectbox(
            "Animal species in your video",
            ["Giraffe", "Zebra", "Elephant", "Deer", "Horse", "Other Wildlife"],
            index=0
        )
        frame_skip = st.slider(
            "Frame skip (higher = faster, less detail)",
            min_value=1, max_value=6, value=3,
            help="Process every Nth frame. 1 = all frames (slow), 6 = every 6th frame (fast)"
        )
        st.divider()
        st.markdown("### 📊 About the Model")
        st.markdown(
            "**X3D-KABR** is trained on the KABR dataset — "
            "10+ hours of drone footage of Kenyan wildlife "
            "(giraffes, plains zebras, Grevy's zebras).  \n\n"
            "Published at **WACV 2024**.  \n\n"
            "⚠️ Best results on **drone/aerial footage** of "
            "ungulates (giraffe, zebra, deer, horse)."
        )
        st.divider()
        st.markdown("### ℹ️ Processing Time")
        st.markdown(
            "- ~2–4 min for a 30-second video  \n"
            "- ~5–8 min for a 1-minute video  \n"
            "*(Research-grade model — worth the wait!)*"
        )

    # ── Model Loading ─────────────────────────────────────────
    with st.spinner("🔄 Loading X3D-KABR model (first load takes ~1 minute)..."):
        model, yolo = load_models()
    st.success(f"✅ Model ready on **{DEVICE}**")

    # ── File Upload ───────────────────────────────────────────
    st.subheader("📤 Upload Your Drone Wildlife Video")
    uploaded = st.file_uploader(
        "Supported formats: MP4, AVI, MOV, MKV",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is None:
        st.info("👆 Upload a video to get started. Best results with drone footage of large animals.")
        # Show example output description
        with st.expander("What will you get?"):
            st.markdown(
                "**After processing you'll receive:**\n"
                "- 🎬 **Annotated video** — bounding boxes + behavior labels per animal\n"
                "- 🥧 **Time budget chart** — how much time each behavior occupied\n"
                "- 📈 **Behavior timeline** — behavior changes over the full video\n"
                "- 📋 **Detailed summary** — frame counts per behavior class"
            )
        return

    # ── Save uploaded video to temp file ─────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    video_name = uploaded.name

    # Quick video info
    cap     = cv2.VideoCapture(input_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_sec = total_f / fps
    cap.release()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration",    f"{dur_sec:.1f}s")
    col2.metric("Resolution",  f"{W}×{H}")
    col3.metric("FPS",         f"{fps:.0f}")
    col4.metric("Total Frames",f"{total_f:,}")

    # ── Run Button ────────────────────────────────────────────
    if st.button("🚀 Analyze Behavior", type="primary", use_container_width=True):

        st.subheader("🔍 Processing...")
        progress_bar  = st.progress(0.0)
        status_text   = st.empty()
        preview_slot  = st.empty()

        start_time = time.time()

        output_path, counts, timeline = process_video(
            model        = model,
            yolo         = yolo,
            video_path   = input_path,
            species_name = species_name,
            frame_skip   = frame_skip,
            progress_bar = progress_bar,
            status_text  = status_text,
            preview_slot = preview_slot
        )

        elapsed = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Done in {elapsed:.0f} seconds!")
        preview_slot.empty()

        # ── Results ───────────────────────────────────────────
        st.divider()
        st.subheader("📊 Results")

        total_preds = sum(counts.values())

        if total_preds == 0:
            st.warning(
                "⚠️ No animals detected in this video.  \n"
                "**Tips:**  \n"
                "- Try a lower frame skip setting  \n"
                "- Ensure animals are clearly visible and not too far away  \n"
                "- Best results with drone footage at 10–50m altitude"
            )
            return

        # Summary metrics
        dominant = max(counts, key=counts.get)
        dom_pct  = counts[dominant] / total_preds * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Classifications", f"{total_preds:,}")
        m2.metric("Dominant Behavior",     dominant)
        m3.metric("Dominance",             f"{dom_pct:.1f}%")

        # Behavior breakdown table
        st.markdown("**Behavior Breakdown:**")
        for label in LABEL_NAMES[:7]:
            c   = counts.get(label, 0)
            pct = c / total_preds * 100 if total_preds else 0
            col_label, col_bar, col_pct = st.columns([2, 6, 2])
            col_label.write(f"**{label}**")
            col_bar.progress(pct / 100)
            col_pct.write(f"{pct:.1f}%")

        st.divider()

        # Charts side by side
        smoothed = smooth_predictions(timeline, window=5)
        smooth_cnt = Counter(l for _, l, _ in smoothed)

        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            fig1 = make_time_budget_chart(counts, "Raw Predictions")
            st.pyplot(fig1)
            plt.close(fig1)
        with col_pie2:
            fig2 = make_time_budget_chart(smooth_cnt, "Smoothed (5-frame vote)")
            st.pyplot(fig2)
            plt.close(fig2)

        # Timeline
        fig3 = make_timeline_chart(smoothed, video_name)
        if fig3:
            st.pyplot(fig3)
            plt.close(fig3)

        st.divider()

        # ── Downloads ─────────────────────────────────────────
        st.subheader("⬇️ Download Results")

        col_dl1, col_dl2 = st.columns(2)

        with open(output_path, "rb") as f:
            col_dl1.download_button(
                label     = "🎬 Download Annotated Video",
                data      = f,
                file_name = f"behavior_{video_name}",
                mime      = "video/mp4",
                use_container_width=True
            )

        # Save charts to bytes for download
        import io
        buf = io.BytesIO()
        if fig3:
            fig_combined, axes = plt.subplots(1, 2, figsize=(14, 5))
            # Recreate pie in combined figure
            labels  = [l for l in LABEL_NAMES[:7] if smooth_cnt.get(l, 0) > 0]
            sizes   = [smooth_cnt[l] for l in labels]
            colors  = [LABEL_COLORS_HEX[LABEL_NAMES.index(l)] for l in labels]
            axes[0].pie(sizes, labels=labels, colors=colors,
                        autopct="%1.1f%%", startangle=90)
            axes[0].set_title("Time Budget (Smoothed)")
            # Timeline strip
            times = [t for t, _, _ in smoothed]
            dt    = (times[1] - times[0]) if len(times) > 1 else 0.1
            label_to_idx = {l: i for i, l in enumerate(LABEL_NAMES)}
            for t, lbl, _ in smoothed:
                idx = label_to_idx.get(lbl, 7)
                axes[1].axvspan(t - dt/2, t + dt/2,
                                color=LABEL_COLORS_HEX[idx], alpha=0.9)
            axes[1].set_title("Behavior Timeline")
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_yticks([])
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_combined)
            buf.seek(0)

        col_dl2.download_button(
            label     = "📈 Download Charts (PNG)",
            data      = buf,
            file_name = f"behavior_charts_{os.path.splitext(video_name)[0]}.png",
            mime      = "image/png",
            use_container_width=True
        )

        st.success(
            f"🎉 Analysis complete! Processed **{total_preds:,}** animal behavior "
            f"classifications from **{video_name}** in **{elapsed:.0f} seconds**."
        )

        # Cleanup
        os.unlink(input_path)


if __name__ == "__main__":
    main()
