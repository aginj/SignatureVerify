# ✍️ Signature Verification App

A **Streamlit-based application** for verifying signatures across two documents.  
It combines **YOLOv8 (Ultralytics)** for signature detection, **VGG16 embeddings** for feature extraction, and multiple **similarity metrics** for comparison and classification.  

---

## 🚀 Features

- 📂 Upload documents in **JPG, PNG, or PDF** format  
- 🖼️ **PDF-to-image conversion** (first page only)  
- 🤖 **YOLOv8-based signature detection**  
- 🔍 **VGG16 embeddings** for signature representation  
- 📊 Multiple similarity metrics:
  - Cosine similarity  
  - Euclidean similarity & distance  
  - Manhattan similarity & distance  
- ⚖️ Weighted similarity score with classification (Match / Different)  
- 🧹 Automatic cleanup of temporary folders (`uploads/`, `crops/`)  

---

## 📂 Project Structure

```
SignatureVerificationApp/
│── app.py                        # Main Streamlit app
│
├── SOURCE/
│   ├── utils/
│   │   ├── file_utils.py          # File handling & PDF → image
│   │   └── similarity.py          # Similarity metrics
│   │
│   ├── yolo_files/
│   │   └── detect.py              # YOLOv8 signature detection
│   │
│   ├── vgg_model/
│   │   └── vgg_verify.py          # VGG16 embeddings
│
├── models/
│   └── best.pt                    # YOLO trained weights (place here)
│
├── uploads/                       # Temporary uploads
├── crops/                         # Cropped signature outputs
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/aginj/SignatureVerify.git
   cd SignatureVerify
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux / Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add YOLO model weights**
   - Place your trained YOLOv8 weights (`best.pt`) in the `models/` directory.  
   - You can train your own model or use a pretrained one.  

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the link shown in the terminal (default: `http://localhost:8501`).

---

## 🖼️ Workflow

1. Upload **two documents** (image or PDF).  
2. The app will:  
   - Convert PDFs → images (first page only)  
   - Detect signatures using YOLO  
   - Crop and extract embeddings via VGG16  
   - Compute similarities and weighted score  
3. Output:  
   - 📊 Similarity metrics table  
   - ✅ Classification result (Match / Different)  

---

## 📊 Example Output

| Metric               | Value  |
|-----------------------|--------|
| Cosine Similarity     | 0.92   |
| Euclidean Similarity  | 0.81   |
| Manhattan Similarity  | 0.79   |
| Euclidean Distance    | 0.45   |
| Manhattan Distance    | 1.12   |
| **Weighted Score**    | 0.85   |
| **Verdict**           | 🔒 Likely **MATCH** |

---

## 🔮 Future Improvements

- Train YOLO on a larger dataset for higher accuracy  
- Add support for **multiple signatures** per document  
- Use **fine-tuned CNN embeddings** instead of ImageNet VGG16  
- Deploy app on **Streamlit Cloud / Docker**  

---

## 📜 License

This project is licensed under the MIT License – feel free to use and modify.  
