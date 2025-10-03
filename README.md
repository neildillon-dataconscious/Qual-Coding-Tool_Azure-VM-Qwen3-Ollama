# Private LLM RAG (Azure VM, Qwen3 + Weaviate)

Local-only pipeline for qualitative evidence mapping against criteria (e.g., ALNAP).  
Components (all on one Azure VM):
- Qwen3 via Ollama (LLM for chunking/optional verification)
- Embeddings via Ollama (nomic-embed-text by default)
- Weaviate (BM25 + HNSW vectors)
- Cross-encoder re-rank (BGE Reranker)

## Quick start
1. `docker compose -f docker/docker-compose.yaml up -d`
2. `python -m pip install -r requirements.txt`
3. Put transcripts in `data/docs_raw/`
4. Create `criteria/*.xlsx` with columns:
   `criterion_id, criterion_label, subcriterion_id, subcriterion_label, guidance_prompt`
5. Run:
python -m src.cli
--criteria criteria/sample_alnap_criteria.xlsx
--docs data/docs_raw
--out outputs/evidence.csv
--config config.yaml
