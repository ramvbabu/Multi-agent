import os
from typing import Any, Dict, List

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = os.getenv("PINECONE_INDEX", "claim-policies")


def _get_pinecone_client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENV")
    if not api_key or not environment:
        raise EnvironmentError("Missing Pinecone environment variables. Set PINECONE_API_KEY and PINECONE_ENV.")
    return Pinecone(api_key=api_key)


def _get_index(pc: Pinecone, index_name: str):
    indexes = pc.list_indexes().names()
    if index_name not in indexes:
        raise ValueError(f"Pinecone index '{index_name}' does not exist")
    return pc.Index(index_name)


@tool
def query_policy_rag(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve relevant policy context for a user query from the Pinecone RAG index.

    Returns a dict with keys: 'context' (combined text), 'sources' (list of metadata),
    and 'requires_human_override' (bool) if any retrieved chunk needs human fallback.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)

    pc = _get_pinecone_client()
    index = _get_index(pc, INDEX_NAME)
    
    vector = embeddings.embed_query(query)  #raw_embeddings may be a list of Decimals or a custom type, so we convert to standard floats
    # THE CRITICAL FIX: Ensure it's a native Python list of standard floats
    clean_embeddings = [float(x) for x in vector]

    # Now hand 'clean_embeddings' to your pinecone index query
    resp = index.query(vector=clean_embeddings, top_k=3, include_metadata=True)
    
    

    # Query Pinecone; include metadata to gather requires_human_fallback flags
    # resp = index.query(queries=[vector], top_k=top_k, include_metadata=True)

    # Pinecone SDK returns matches in different shapes; be defensive
    matches = []
    try:
        matches = resp[0].matches if isinstance(resp, list) and resp else resp.matches
    except Exception:
        # fallback: try common attribute
        matches = getattr(resp, "matches", []) or []

    pieces: List[str] = []
    sources: List[Dict[str, Any]] = []
    requires_human = False

    for m in matches:
        meta = getattr(m, "metadata", None) or m.get("metadata", {})
        text = meta.get("content") or meta.get("page_content") or meta.get("text") or ""
        if not text:
            # sometimes the embedding payload stores a preview
            text = meta.get("text_preview", "")
        pieces.append(text)
        sources.append(meta)
        if meta.get("requires_human_fallback"):
            requires_human = True

    combined = "\n\n".join([p for p in pieces if p])

    return {
        "context": combined,
        "sources": sources,
        "requires_human_override": requires_human,
    }


if __name__ == "__main__":
    # quick manual test (requires env vars)
    print(query_policy_rag.invoke({"query": "What does the policy say about collision coverage?"}))
