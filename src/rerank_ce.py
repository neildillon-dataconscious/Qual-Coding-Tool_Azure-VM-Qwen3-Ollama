import torch
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", batch_size=128, fp16=True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        if fp16 and device == "cuda":
            try: self.model.model.half()
            except Exception: pass
        self.batch_size = batch_size

    def score_and_sort(self, query_text: str, candidates):
        pairs = [(query_text, c["text"]) for c in candidates]
        with torch.inference_mode():
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
            )
        for c, s in zip(candidates, scores):
            c["ce_score"] = float(s)
        return sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
