import cv2, os
from ultralytics import YOLO

YOLO_MODEL_PATH = "models/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

def detect_signature(image_path, save_folder="crops"):
    os.makedirs(save_folder, exist_ok=True)
    results = yolo_model(image_path)
    crops = []

    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box[:4])
        img = cv2.imread(image_path)
        crop = img[y1:y2, x1:x2]
        crop_path = os.path.join(save_folder, f"{os.path.basename(image_path)}_sig{i+1}.jpg")
        cv2.imwrite(crop_path, crop)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append((crop_rgb, crop_path))
    return crops

