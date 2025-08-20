import os, shutil
from pdf2image import convert_from_path

UPLOAD_FOLDER = "uploads"
CROP_FOLDER = "crops"

def clear_folders(folders=[UPLOAD_FOLDER, CROP_FOLDER]):
    """Delete all files in given folders."""
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def save_uploaded_file(uploaded_file, folder=UPLOAD_FOLDER):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def pdf_to_images(pdf_path, folder=UPLOAD_FOLDER):
    images = convert_from_path(pdf_path, dpi=200)
    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(folder, f"{os.path.basename(pdf_path)}_page{i+1}.jpg")
        img.save(img_path, "JPEG")
        image_paths.append(img_path)
    return image_paths

