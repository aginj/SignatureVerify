import cv2, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

base_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
feature_model = Model(inputs=base_model.input, outputs=base_model.output)

def get_signature_embedding(image):
    img = cv2.resize(image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    embedding = feature_model.predict(img, verbose=0)
    return embedding.flatten()

def compare_signatures(sig1, sig2):
    return float(cosine_similarity([sig1], [sig2])[0][0])

