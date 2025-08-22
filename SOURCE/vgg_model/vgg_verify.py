import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Load base VGG16 model with pre-trained ImageNet weights
base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
feature_model = Model(inputs=base_model.input, outputs=base_model.output)


def get_signature_embedding(image):
    """
    Extracts a VGG16 embedding for the given signature image and applies L2 normalization.
    
    Args:
        image (np.ndarray): Input image in RGB format.
    
    Returns:
        np.ndarray: L2-normalized embedding vector.
    """
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Extract features
    embedding = feature_model.predict(img, verbose=0).flatten()

    # 🔑 Apply L2 normalization
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    embedding = embedding / norm

    return embedding


def compare_signatures(sig1, sig2):
    """
    Compare two signature embeddings using cosine similarity.
    
    Args:
        sig1 (np.ndarray): First signature embedding.
        sig2 (np.ndarray): Second signature embedding.
    
    Returns:
        float: Cosine similarity score between sig1 and sig2.
    """
    return float(cosine_similarity([sig1], [sig2])[0][0])
