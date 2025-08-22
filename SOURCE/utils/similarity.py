import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_multiple_similarities(features1, features2, weights=None):
    """
    Calculate multiple similarity/distance metrics between two embeddings.
    Assumes embeddings are already L2-normalized.
    """
    try:
        # Ensure numpy arrays
        f1 = np.array(features1).reshape(1, -1)
        f2 = np.array(features2).reshape(1, -1)

        # Cosine similarity (best for normalized embeddings)
        cosine_sim = float(cosine_similarity(f1, f2)[0][0])

        # Euclidean distance (0 = identical, 2 = opposite for unit vectors)
        euclidean_dist = float(np.linalg.norm(f1 - f2))
        euclidean_sim = 1 / (1 + euclidean_dist)

        # Manhattan distance
        manhattan_dist = float(np.sum(np.abs(f1 - f2)))
        manhattan_sim = 1 / (1 + manhattan_dist)

        # Weighted similarity (default weights: cosine strongest)
        if weights is None:
            weights = {"cosine": 0.6, "euclidean": 0.25, "manhattan": 0.15}

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
        print(f"[Similarity Error] {e}")
        return None
