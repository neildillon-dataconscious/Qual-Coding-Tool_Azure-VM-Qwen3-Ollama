import os, json
from typing import List, Dict
from pypdf import PdfReader
from docx import Document as DocxDocument
from tqdm import tqdm
from unidecode import unidecode
from .utils import sha256_file, clean_text

SUPPORTED = (".pdf", ".docx", ".txt", ".md")

def read_text_with_pages(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = []
        for i, p in enumerate(reader.pages):
            txt = p.extract_text() or ""
            pages.append({"page": i+1, "text": clean_text(txt)})
        return pages
    elif ext == ".docx":
        doc = DocxDocument(path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return [{"page": 1, "text": clean_text(text)}]
    elif ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{"page": 1, "text": clean_text(text)}]
    else:
        raise ValueError(f"Unsupported file: {path}")

def ingest_folder(raw_folder: str, processed_folder: str) -> List[Dict]:
    os.makedirs(processed_folder, exist_ok=True)
    records = []
    files = [os.path.join(raw_folder, f) for f in os.listdir(raw_folder)
             if os.path.splitext(f)[1].lower() in SUPPORTED]
    for fp in tqdm(files, desc="Ingesting"):
        doc_id = sha256_file(fp)
        pages = read_text_with_pages(fp)
        out_path = os.path.join(processed_folder, f"{doc_id}.jsonl")
        with open(out_path, "w", encoding="utf-8") as out:
            for pg in pages:
                rec = {
                    "doc_id": doc_id,
                    "source_path": fp,
                    "page": pg["page"],
                    "text": unidecode(pg["text"])
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
    return records
