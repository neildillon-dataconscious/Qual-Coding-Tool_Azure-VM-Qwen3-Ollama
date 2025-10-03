import os, uuid, argparse, pandas as pd, numpy as np
from tqdm import tqdm
from collections import defaultdict

from .utils import load_yaml
from .ingest import ingest_folder
from .chunk_llm import chunk_document_llm
from .embedder import OllamaEmbedder
from .weaviate_store import get_client, ensure_class, upsert_chunks
from .retrieve_weaviate import weaviate_hybrid
from .rerank_ce import CrossEncoderReranker
from .mmr_dedup import mmr_select, dedup_by_threshold
from .export_csv import export_rows

def read_criteria_xlsx(path: str):
    df = pd.read_excel(path)
    needed = ["criterion_id","criterion_label","subcriterion_id","subcriterion_label","guidance_prompt"]
    for n in needed:
        if n not in df.columns:
            raise ValueError(f"Missing column in criteria: {n}")
    return df

def ollama_chat_call(ollama_url, model, system, user, temperature=0.0, max_tokens=128):
    import requests
    payload = {
        "model": model,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(f"{ollama_url}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # 1) Ingest
    pages = ingest_folder(args.docs, cfg["paths"]["docs_processed"])

    # 2) Chunk (LLM semantic)
    by_doc = defaultdict(list)
    for rec in pages:
        by_doc[rec["doc_id"]].append({"page": rec["page"], "text": rec["text"], "source_path": rec["source_path"]})

    def llm_call(system, user, temperature=0.0, max_tokens=128):
        return ollama_chat_call(cfg["services"]["ollama_url"], cfg["models"]["gen_model"], system, user, temperature, max_tokens)

    all_chunks = []
    for doc_id, plist in tqdm(by_doc.items(), desc="Chunking"):
        chunks = chunk_document_llm(
            [{"page": p["page"], "text": p["text"]} for p in plist],
            target_tokens=cfg["chunking"]["target_tokens"],
            min_tokens=cfg["chunking"]["min_tokens"],
            max_tokens=cfg["chunking"]["max_tokens"],
            add_context_tokens=cfg["chunking"]["add_context_tokens"],
            boundary_llm=cfg["chunking"]["boundary_llm"]["enable"],
            llm_call=llm_call
        )
        for ch in chunks:
            ch["doc_id"] = doc_id
            ch["source_path"] = plist[0]["source_path"]
            ch["uuid"] = str(uuid.uuid4())
        all_chunks.extend(chunks)

    # 3) Embeddings
    embedder = OllamaEmbedder(cfg["services"]["ollama_url"], cfg["models"]["embed_model"])
    vecs = []
    B = 64
    for i in tqdm(range(0, len(all_chunks), B), desc="Embedding"):
        batch = [c["text"] for c in all_chunks[i:i+B]]
        v = embedder.encode(batch, normalize=True)
        vecs.append(v)
    vecs = np.vstack(vecs)
    for c, v in zip(all_chunks, vecs):
        c["embedding"] = v

    # 4) Weaviate upsert
    wclient = get_client(cfg["services"]["weaviate_url"])
    class_name = cfg["vector_store"]["class_name"]
    ensure_class(wclient, class_name)
    upsert_chunks(wclient, class_name, all_chunks)

    # 5) Retrieval + CE re-rank + MMR + dedup
    crit_df = read_criteria_xlsx(args.criteria)
    reranker = CrossEncoderReranker(cfg["models"]["reranker"], batch_size=cfg["rerank"]["batch_size"], fp16=cfg["rerank"]["fp16"])

    rows = []
    for _, row in tqdm(crit_df.iterrows(), total=len(crit_df), desc="Criteria"):
        q_text = str(row["guidance_prompt"]).strip()
        q_vec = embedder.encode(q_text, normalize=True)[0]

        cands = weaviate_hybrid(wclient, class_name, q_text, q_vec, alpha=cfg["retrieval"]["alpha"], top_k=cfg["retrieval"]["top_k_pre_rerank"])

        ranked = reranker.score_and_sort(q_text, cands)

        cand_texts = [c["text"] for c in ranked]
        cand_vecs = embedder.encode(cand_texts, normalize=True)

        if cfg["retrieval"]["use_mmr"] and len(ranked) > 1:
            select_idx = mmr_select(cand_vecs, k=cfg["retrieval"]["top_k_final"], lambda_=cfg["retrieval"]["mmr_lambda"])
        else:
            select_idx = list(range(min(cfg["retrieval"]["top_k_final"], len(ranked))))
        selected = [ranked[i] for i in select_idx]

        sel_texts = [s["text"] for s in selected]
        sel_vecs = embedder.encode(sel_texts, normalize=True)
        kept_idx = dedup_by_threshold(sel_vecs, threshold=cfg["retrieval"]["dedup_similarity"])
        final = [selected[i] for i in kept_idx]

        for c in final:
            rows.append({
                "criterion_id": row["criterion_id"],
                "criterion_label": row["criterion_label"],
                "subcriterion_id": row["subcriterion_id"],
                "subcriterion_label": row["subcriterion_label"],
                "doc_id": c.get("doc_id",""),
                "source_path": c.get("source_path",""),
                "page": c.get("page",""),
                "char_start": c.get("char_start",0),
                "char_end": c.get("char_end",0),
                "excerpt": c["text"],
                "retrieval_method": "hybrid+ce",
                "score": float(c.get("_score") or 0.0),
                "ce_score": float(c.get("ce_score") or 0.0),
            })

    meta = {
        "embed_model": cfg["models"]["embed_model"],
        "gen_model": cfg["models"]["gen_model"],
        "version": "0.1.0"
    }
    export_rows(args.out, rows, cfg["output"]["csv_fields"], meta)

if __name__ == "__main__":
    main()

