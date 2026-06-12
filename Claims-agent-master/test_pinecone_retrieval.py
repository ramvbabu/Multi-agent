import os
import re
import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

DEFAULT_QUERY = "Does this claim cover deer collision?"
INDEX_NAME = os.getenv("PINECONE_INDEX", "claims-policies")
EMBEDDING_MODEL = "text-embedding-3-small"
SEARCH_TOP_K = 20
DISPLAY_TOP_K = 5
METADATA_FILTER = {
    "category": {
        "$in": ["coverage", "exclusions"]
    }
}

STOPWORDS = {
    "the",
    "and",
    "or",
    "for",
    "with",
    "that",
    "this",
    "does",
    "do",
    "a",
    "an",
    "in",
    "on",
    "of",
    "to",
    "it",
    "is",
    "are",
    "cover",
    "claims",
    "claim",
    "coverage",
    "insurance",
    "policy",
    "does",
    "if",
    "any",
    "the",
    "by",
    "at",
    "be",
    "from",
    "through",
}


def get_query_string():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return os.getenv("QUERY", DEFAULT_QUERY).strip()


def extract_relevant_terms(query: str) -> set[str]:
    tokens = re.findall(r"\w+", query.lower())
    return {
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2 and not token.isdigit()
    }


def extract_query_phrases(query: str, max_n: int = 3) -> list[str]:
    tokens = [
        token
        for token in re.findall(r"\w+", query.lower())
        if token not in STOPWORDS and len(token) > 2 and not token.isdigit()
    ]

    if len(tokens) < 2:
        return []

    phrases = []
    max_n = min(max_n, len(tokens))
    for n in range(2, max_n + 1):
        for i in range(len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i : i + n]))

    return phrases


def get_query_embedding(query: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
        return embeddings.embed_query(query)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate query embedding: {exc}") from exc


def normalize_chunk_text(metadata: dict) -> str:
    return (
        metadata.get("page_content")
        or metadata.get("text")
        or metadata.get("content")
        or ""
    )


def count_term_matches(text: str, terms: set[str]) -> int:
    lower_text = text.lower()
    return sum(
        1
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", lower_text)
    )


def count_phrase_matches(text: str, phrases: list[str]) -> int:
    lower_text = text.lower()
    return sum(1 for phrase in phrases if phrase in lower_text)


def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing PINECONE_API_KEY environment variable")

    return Pinecone(api_key=api_key)


def get_index(client: Pinecone):
    try:
        indexes = client.list_indexes().names()
    except Exception as exc:
        raise RuntimeError(f"Failed to list Pinecone indexes: {exc}") from exc

    if INDEX_NAME not in indexes:
        raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist")

    return client.Index(INDEX_NAME)


def print_intervention_warning(reason: str):
    print("\n" + "!" * 80)
    print("HUMAN INTERVENTION REQUIRED:", reason)
    print("!" * 80 + "\n")


def main():
    try:
        client = get_pinecone_client()
        print("Fetched Pinecone client")

        index = get_index(client)
        print(f"Using Pinecone index: {INDEX_NAME}")

        query = get_query_string()
        query_vector = get_query_embedding(query)
        print(f"Generated query embedding for: '{query}'")

        response = index.query(
            vector=query_vector,
            top_k=SEARCH_TOP_K,
            include_metadata=True,
            include_values=False,
            filter=METADATA_FILTER,
        )

        matches = response.get("matches", [])
        if not matches:
            print("No matching chunks found under metadata filter.")
            print_intervention_warning("No relevant coverage/exclusions chunks were returned.")
            return

        query_terms = extract_relevant_terms(query)
        query_phrases = extract_query_phrases(query)
        scored_matches = []

        for match in matches:
            metadata = match.get("metadata", {}) or {}
            page_content = normalize_chunk_text(metadata)
            score = float(match.get("score", 0.0))
            term_matches = count_term_matches(page_content, query_terms)
            phrase_matches = count_phrase_matches(page_content, query_phrases)
            evidence_score = term_matches + 3 * phrase_matches
            requires_human = bool(metadata.get("requires_human_fallback"))

            scored_matches.append(
                {
                    "metadata": metadata,
                    "page_content": page_content,
                    "score": score,
                    "term_matches": term_matches,
                    "phrase_matches": phrase_matches,
                    "evidence_score": evidence_score,
                    "requires_human": requires_human,
                }
            )

        scored_matches.sort(
            key=lambda item: (item["evidence_score"], item["score"]),
            reverse=True,
        )

        if query_terms and not any(item["evidence_score"] > 0 for item in scored_matches):
            print(
                "WARNING: None of the top retrieved chunks contain explicit terms or phrases from the original query. "
                "This suggests the answer may be based on broader coverage semantics rather than a direct clause."
            )

        fallback_detected = False
        for idx, item in enumerate(scored_matches[:DISPLAY_TOP_K], start=1):
            metadata = item["metadata"]
            page_content = item["page_content"]
            requires_human = item["requires_human"]

            print(f"\n--- Match {idx} ---")
            print(f"score: {item['score']:.4f}")
            print(f"evidence_score: {item['evidence_score']}")
            print(f"term_matches: {item['term_matches']}")
            print(f"phrase_matches: {item['phrase_matches']}")
            print("page_content:")
            if page_content:
                print(page_content)
            else:
                print("[No page_content available for this chunk]")

            print("metadata:")
            print(metadata)

            if requires_human:
                fallback_detected = True
                print_intervention_warning("Retrieved chunk flagged requires_human_fallback=True.")

        if fallback_detected:
            print_intervention_warning("At least one retrieved chunk requires human review.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
