import csv, os
from typing import List, Dict
from .utils import now_iso

def export_rows(path: str, rows: List[Dict], fields: List[str], meta: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for r in rows:
        r.setdefault("model_embed", meta.get("embed_model",""))
        r.setdefault("model_generate", meta.get("gen_model",""))
        r.setdefault("pipeline_version", meta.get("version","0.1.0"))
        r.setdefault("run_timestamp", now_iso())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
