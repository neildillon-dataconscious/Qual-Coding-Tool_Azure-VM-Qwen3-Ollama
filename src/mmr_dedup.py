import numpy as np

def mmr_select(text_vecs: np.ndarray, k: int, lambda_: float = 0.5):
    if len(text_vecs) == 0:
        return []
    sim = text_vecs @ text_vecs.T  # cosine if normalized
    selected = [0]
    candidates = list(range(1, len(text_vecs)))
    while len(selected) < min(k, len(text_vecs)) and candidates:
        scores = []
        for j in candidates:
            rel = sim[j, 0]              # similarity to first (proxy for query)
            div = max(sim[j, selected])  # max similarity to chosen
            scores.append((lambda_ * rel - (1 - lambda_) * div, j))
        j_best = sorted(scores, key=lambda x: x[0], reverse=True)[0][1]
        candidates.remove(j_best)
        selected.append(j_best)
    return selected

def dedup_by_threshold(text_vecs: np.ndarray, threshold: float):
    kept = []
    for i, v in enumerate(text_vecs):
        if not kept:
            kept.append(i); continue
        sims = (text_vecs[kept] @ v)
        if sims.max() >= threshold:
            continue
        kept.append(i)
    return kept
