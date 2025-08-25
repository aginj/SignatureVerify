import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def l2_normalize(vec):
    """Ensure vector is L2-normalized."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def calculate_multiple_similarities(features1, features2, weights=None):
    """
    Calculate cosine, Euclidean, Manhattan similarities and distances.
    Also compute a weighted similarity score if weights provided.
    """
    try:
        # Ensure numpy arrays
        f1 = np.array(features1).flatten()
        f2 = np.array(features2).flatten()

        # 🔑 Safeguard normalization
        f1 = l2_normalize(f1).reshape(1, -1)
        f2 = l2_normalize(f2).reshape(1, -1)

        # Cosine similarity
        cosine_sim = float(cosine_similarity(f1, f2)[0][0])

        # Euclidean
        euclidean_dist = float(np.linalg.norm(f1 - f2))
        euclidean_sim = 1 - (euclidean_dist / 2)   # better scaling for [0,1]

        # Manhattan
        manhattan_dist = float(np.sum(np.abs(f1 - f2)))
        manhattan_sim = 1 / (1 + manhattan_dist)   # keep compressed into (0,1]

        # Weighted similarity (default: cosine-heavy)
        if weights is None:
            weights = {"cosine": 0.8, "euclidean": 0.15, "manhattan": 0.05}

        weighted_score = (
            cosine_sim * weights["cosine"] +
            euclidean_sim * weights["euclidean"] +
            manhattan_sim * weights["manhattan"]
        )

        return {
            "cosine_similarity": cosine_sim,
            "euclidean_similarity": euclidean_sim,
            "manhattan_similarity": manhattan_sim,
            "euclidean_distance": euclidean_dist,
            "manhattan_distance": manhattan_dist,
            "weighted_similarity": float(weighted_score)
        }

    except Exception as e:
        print(f"[Similarity Error] {e}")  # for debugging in terminal
        return None

