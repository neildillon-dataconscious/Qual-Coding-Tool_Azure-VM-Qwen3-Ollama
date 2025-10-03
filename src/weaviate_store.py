import weaviate
from weaviate.classes.config import Property, DataType, Configure
from typing import List, Dict

def get_client(url: str):
    return weaviate.Client(url)

def ensure_class(client, class_name: str):
    if not client.collections.exists(class_name):
        client.collections.create(
            class_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="source_path", data_type=DataType.TEXT),
                Property(name="section_title", data_type=DataType.TEXT),
                Property(name="speaker", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="char_start", data_type=DataType.INT),
                Property(name="char_end", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(
                max_connections=64, ef_construction=200
            ),
        )

def upsert_chunks(client, class_name: str, chunks: List[Dict]):
    col = client.collections.get(class_name)
    with col.batch.dynamic() as batch:
        for ch in chunks:
            batch.add_object(
                properties={
                    "doc_id": ch["doc_id"],
                    "source_path": ch["source_path"],
                    "section_title": ch.get("section_title",""),
                    "speaker": ch.get("speaker",""),
                    "page": ch["page_start"],
                    "char_start": ch["char_start"],
                    "char_end": ch["char_end"],
                    "text": ch["text"],
                },
                vector=ch["embedding"].tolist(),
                uuid=ch["uuid"]
            )
