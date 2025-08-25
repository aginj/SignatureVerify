import os
import streamlit as st
import numpy as np
import pandas as pd
import cv2

from SOURCE.utils.file_utils import clear_folders, save_uploaded_file, pdf_to_images
from SOURCE.yolo_files.detect import detect_signature
from SOURCE.vgg_model.vgg_verify import get_signature_embedding
from SOURCE.utils.similarity import calculate_multiple_similarities

st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")


def process_document(uploaded_file):
    """Save uploaded file, convert PDF → image (first page), run YOLO, return first signature crop (RGB)."""
    file_path = save_uploaded_file(uploaded_file)

    if file_path.lower().endswith(".pdf"):
        pages = pdf_to_images(file_path)
        if not pages:
            st.error(f"No pages found in PDF: {uploaded_file.name}")
            return None
        file_path = pages[0]
        st.info(f"📄 Converted {os.path.basename(uploaded_file.name)} → {len(pages)} page(s). Using first page.")

    detections = detect_signature(file_path)
    if not detections:
        st.warning(f"⚠️ No signature found in {uploaded_file.name}")
        return None

    sig_rgb, sig_path = detections[0]
    st.image(sig_rgb, caption=f"Detected signature from {uploaded_file.name}", width=280)

    return get_signature_embedding(sig_rgb)


def main():
    st.title("✍️ Signature Verification App")
    st.write("Upload **two documents** (image or PDF). The app detects signatures and compares them.")

    clear_folders()

    col1, col2 = st.columns(2)
    with col1:
        f1 = st.file_uploader("Upload Document 1", type=["jpg", "jpeg", "png", "pdf"], key="doc1")
    with col2:
        f2 = st.file_uploader("Upload Document 2", type=["jpg", "jpeg", "png", "pdf"], key="doc2")

    if f1 and f2:
        st.success("✅ Both documents uploaded")

        col1, col2 = st.columns(2)
        with col1:
            emb1 = process_document(f1)
        with col2:
            emb2 = process_document(f2)

        if emb1 is not None and emb2 is not None:
            sims = calculate_multiple_similarities(emb1, emb2)

            if sims and "error" not in sims:
                st.subheader("📊 Similarity Analysis")
                table_df = pd.DataFrame({
                    "Metric": [
                        "Cosine Similarity",
                        "Euclidean Similarity",
                        "Manhattan Similarity",
                        "Euclidean Distance",
                        "Manhattan Distance",
                    ],
                    "Value": [
                        sims["cosine_similarity"],
                        sims["euclidean_similarity"],
                        sims["manhattan_similarity"],
                        sims["euclidean_distance"],
                        sims["manhattan_distance"],
                    ],
                })
                st.dataframe(table_df, use_container_width=True)

                # Weighted score & classification
                st.subheader("✅ Classification Results")
                st.metric("Weighted Similarity Score", f"{sims['weighted_similarity']:.4f}")

                # Verdict using weighted score threshold
                threshold = 0.8
                verdict = "🔒 Likely MATCH" if sims["weighted_similarity"] >= threshold else "❌ Likely DIFFERENT"
                st.success(f"Verdict (weighted ≥ {threshold}): {verdict}")

            else:
                st.error("Error in similarity calculation")


if __name__ == "__main__":
    main()
