from typing import List, Dict, Any
import numpy as np

def weaviate_hybrid(client, class_name: str, query_text: str, query_vec: np.ndarray, alpha: float, top_k: int) -> List[Dict[str, Any]]:
    col = client.collections.get(class_name)
    res = col.query.hybrid(
        query=query_text,
        vector=query_vec.tolist(),
        alpha=alpha,
        limit=top_k,
        properties=["doc_id","source_path","section_title","speaker","page","char_start","char_end","text"]
    )
    out = []
    for obj in res.objects:
        d = dict(obj.properties)
        d["_uuid"] = obj.uuid
        # client exposes scores differently across versions; keep placeholders
        d["_score"] = getattr(obj, "score", None)
        d["_distance"] = getattr(obj, "distance", None)
        out.append(d)
    return out
