import os, yaml, hashlib, re
from datetime import datetime

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
