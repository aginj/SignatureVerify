✍️ Signature Verification App
A Streamlit-based application that detects, extracts, and verifies handwritten signatures from documents (images or PDFs).
It uses YOLO for signature detection and a VGG-based model for feature extraction, followed by multiple similarity metrics (Cosine, Euclidean, Manhattan) for comparison.

🚀 Features
📄 Supports JPG, PNG, JPEG, and PDF documents
🤖 Detects signatures automatically using YOLOv5
🔍 Extracts feature embeddings using a VGG16-based model
📊 Computes multiple similarity metrics:
  1. Cosine Similarity
  2. Euclidean Similarity & Distance
  3. Manhattan Similarity & Distance
✅ Provides a final weighted similarity score and match verdict
🧹 Auto-cleans temporary folders each run

📂 Project Structure
signature-verification/
│── app.py                      # Main Streamlit app
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation
│
├── SOURCE/
│   ├── yolo_files/             # YOLO detection module
│   │   └── detect.py
│   │
│   ├── vgg_model/              # VGG16-based feature extractor
│   │   └── vgg_verify.py
│   │
│   ├── utils/                  # Helper utilities
│   │   ├── file_utils.py       # File saving, cleanup, PDF → image
│   │   └── similarity.py       # Similarity calculation (Cosine, Euclidean, Manhattan)
│
├── media/                      # Temporary uploaded docs
├── results/                    # YOLO detection results
└── crops/                      # Cropped signature regions

⚙️ Installation
1. Clone the repository
   git clone https://github.com/aginj/SignatureVerify.git
   cd signature-verification
2. Install dependencies
   pip install -r requirements.txt
3. Run the app
   streamlit run app.py

🖼️ Usage
1. Upload two documents (images or PDFs).
2. The app will detect and crop signatures automatically.
3. View similarity metrics and final verdict in the UI.
      
