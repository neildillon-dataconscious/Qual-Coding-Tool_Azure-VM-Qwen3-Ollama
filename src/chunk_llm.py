import regex as re
from typing import List, Dict
from nltk.tokenize import sent_tokenize

BOUNDARY_USER = """You score whether two adjacent text segments continue the same topic.
Return JSON: {"same_topic": true|false, "confidence": 0..1}
Segment A:
---
{A}
---
Segment B:
---
{B}
---"""

def split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"(?:\n\s*\n)+", text)
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]

def chunk_document_llm(
    pages: List[Dict],
    target_tokens=1000, min_tokens=400, max_tokens=1400,
    add_context_tokens=120,
    boundary_llm=True,
    llm_call=None,   # function(system, user, temperature, max_tokens)->str
):
    """Return chunks with fields: text, page_start, page_end, char_start, char_end"""
    def tok_est(s: str): return max(1, int(len(s.split()) / 0.75))

    blocks = []
    for pg in pages:
        for pr in split_paragraphs(pg["text"]):
            blocks.append({"page": pg["page"], "text": pr})

    if not blocks:
        return []

    segments = []
    current = blocks[0]["text"]; cur_pages = [blocks[0]["page"]]
    for i in range(1, len(blocks)):
        nxt = blocks[i]["text"]
        if boundary_llm and llm_call is not None:
            user = BOUNDARY_USER.format(A=current[-1200:], B=nxt[:1200])
            resp = llm_call(
                "You evaluate topical continuity between two segments and return minimal JSON only.",
                user, temperature=0.0, max_tokens=80
            ).lower()
            same = "true" in resp
        else:
            same = not bool(re.match(r"^(?:[A-Z][\w\s]{0,40}:|#[#\s]|[A-Z][A-Za-z\s]{3,}\:)", nxt))
        if same:
            current = current + "\n" + nxt
            cur_pages.append(blocks[i]["page"])
        else:
            segments.append({"pages": cur_pages.copy(), "text": current})
            current = nxt; cur_pages = [blocks[i]["page"]]
    segments.append({"pages": cur_pages, "text": current})

    chunks = []
    for seg in segments:
        text = seg["text"]
        if tok_est(text) <= max_tokens:
            chunks.append({"text": text, "pages": (min(seg["pages"]), max(seg["pages"]))})
        else:
            sents = sent_tokenize(text)
            cur, cur_toks = [], 0
            for s in sents:
                st = tok_est(s)
                if cur_toks + st > max_tokens and cur:
                    chunks.append({"text": " ".join(cur), "pages": seg["pages"]})
                    cur, cur_toks = [], 0
                cur.append(s); cur_toks += st
            if cur: chunks.append({"text": " ".join(cur), "pages": seg["pages"]})

    final = []
    for ch in chunks:
        text = " ".join(sent_tokenize(ch["text"]))  # normalize
        final.append({
            "text": text,
            "page_start": ch["pages"][0],
            "page_end": ch["pages"][1],
            "char_start": 0,
            "char_end": len(text)
        })
    return final
