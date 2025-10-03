import requests

def verify_excerpt_ollama(ollama_url: str, model: str, subcrit: str, excerpt: str, temperature: float = 0.0):
    prompt = f"""You are verifying if the excerpt provides evidence for the sub-criterion.
Return JSON: {{"supports": true|false, "reason": "<=140 chars"}}
Sub-criterion: {subcrit}
Excerpt:
---
{excerpt}
---"""
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json().get("response","")
    supports = "true" in resp.lower()
    return supports, resp[:160]
