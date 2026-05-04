import numpy as np


def extend_features(X_full: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Append a new feature vector to the full feature matrix."""
    return np.vstack([X_full, x_new.reshape(1, -1)])


def top_k_indices(scores: np.ndarray, k: int, exclude_indices: set = None) -> list:
    """Get top-k indices by score, optionally excluding certain indices."""
    sorted_idx = np.argsort(scores)[::-1]
    results = []
    for idx in sorted_idx:
        if exclude_indices and idx in exclude_indices:
            continue
        results.append(int(idx))
        if len(results) >= k:
            break
    return results
