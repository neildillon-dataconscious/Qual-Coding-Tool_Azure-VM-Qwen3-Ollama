import requests
import numpy as np

class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout=300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def encode(self, texts, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        payload = {"model": self.model, "input": texts}
        r = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        vecs = [np.array(d["embedding"], dtype="float32") for d in data["data"]]
        if normalize:
            vecs = [v/(np.linalg.norm(v)+1e-9) for v in vecs]
        return np.vstack(vecs)
