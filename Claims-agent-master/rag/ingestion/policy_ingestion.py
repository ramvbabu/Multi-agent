import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from pinecone import Pinecone
from pypdf import PdfReader



load_dotenv()

DATA_DIR = Path(os.getenv("RAG_DATA_PATH", "data")).resolve()
INDEX_NAME = os.getenv("PINECONE_INDEX", "claim-policies")
EMBEDDING_MODEL = "text-embedding-3-small"
CLASSIFICATION_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


class ChunkMetadata(BaseModel):
    category: Literal["general", "coverage", "exclusions", "fallback"]
    summary: str = Field(..., description="Concise summary of the chunk contents")
    requires_human_fallback: bool


CLASSIFICATION_PROMPT = (
    "You are a policy ingestion classifier. "
    "Analyze the following insurance policy text chunk and return structured output with three fields: "
    "category, summary, and requires_human_fallback. "
    "Use category 'coverage' for policy inclusions or payouts, 'exclusions' for rules denying or limiting coverage, "
    "'general' for operational procedures or general policy context, and 'fallback' for vague, unresolvable, or ambiguous clauses. "
    "Set requires_human_fallback to true if the chunk is categorized as 'fallback' or if it implies manual review is needed. "
    "Do not invent categories outside the allowed list.\n\n"
)


def read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"Failed to decode text file: {file_path}")


def read_pdf_file(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
        texts = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(texts)
        if not text.strip():
            raise ValueError(f"No text extracted from PDF: {file_path}")
        return text
    except Exception as exc:
        raise RuntimeError(f"Failed to read PDF file {file_path}: {exc}") from exc


def get_ai_classifier():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

    return ChatOpenAI(
        model=CLASSIFICATION_MODEL,
        api_key=api_key,
        temperature=0.0,
    ).with_structured_output(ChunkMetadata)


def build_chunk_metadata(chunk_text: str, source: str, index: int, classifier) -> dict:
    prompt = (
        CLASSIFICATION_PROMPT
        + "Chunk text:\n" + chunk_text.strip() + "\n\n"
        + "Respond only with the structured output."
    )

    response = classifier.invoke(prompt)
    parsed = None
    if isinstance(response, dict):
        parsed = response.get("parsed") if "parsed" in response else response
    else:
        parsed = response

    if parsed is None:
        raise RuntimeError("AI classifier failed to return structured metadata for a chunk")

    if isinstance(parsed, dict):
        parsed = ChunkMetadata(**parsed)

    print(
        f"Classified chunk {source}:{index} -> category={parsed.category}, "
        f"requires_human_fallback={parsed.requires_human_fallback}"
    )

    preview = " ".join(chunk_text.strip().splitlines())[:200].strip()

    return {
        "source": source,
        "chunk_index": index,
        "category": parsed.category,
        "summary": parsed.summary,
        "text_preview": preview,
        "requires_human_fallback": parsed.requires_human_fallback,
        "page_content": chunk_text,
        "text": chunk_text,
        "content": chunk_text,
    }


def load_documents(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents = []
    supported_extensions = {".txt", ".md", ".pdf"}

    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                text = read_pdf_file(file_path)
            else:
                text = read_text_file(file_path)

            if not text.strip():
                print(f"Skipping empty document: {file_path}", file=sys.stderr)
                continue

            documents.append(
                {
                    "id": file_path.stem,
                    "text": text,
                    "source": str(file_path),
                }
            )
        except Exception as exc:
            print(f"Error processing {file_path}: {exc}", file=sys.stderr)

    if not documents:
        raise ValueError(f"No supported text or PDF documents found in: {data_dir}")

    return documents


def split_documents(documents, classifier):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for document in documents:
        try:
            split_texts = splitter.split_text(document["text"])
        except Exception as exc:
            raise RuntimeError(f"Failed to split document {document['source']}: {exc}") from exc

        for index, chunk_text in enumerate(split_texts):
            metadata = build_chunk_metadata(chunk_text, document["source"], index, classifier)
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

    if not chunks:
        raise ValueError("No chunks were generated from the provided documents")

    return chunks


def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENV")

    if not api_key or not environment:
        raise EnvironmentError(
            "Missing Pinecone environment variables. Set PINECONE_API_KEY and PINECONE_ENV."
        )

    try:
        return Pinecone(api_key=api_key, environment=environment)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Pinecone client: {exc}") from exc


def get_index(pc: Pinecone, index_name: str):
    try:
        indexes = pc.list_indexes().names()
    except Exception as exc:
        raise RuntimeError(f"Failed to list Pinecone indexes: {exc}") from exc

    if index_name not in indexes:
        raise ValueError(f"Pinecone index '{index_name}' does not exist")

    try:
        return pc.Index(index_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to connect to Pinecone index '{index_name}': {exc}") from exc


def embed_texts(texts):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY environment variable")

    try:
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=api_key,
        )
        return embeddings.embed_documents(texts)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embeddings: {exc}") from exc


def upsert_to_pinecone(index, chunks, vectors):
    try:
        payload = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = f"{chunk.metadata['source']}#{chunk.metadata['chunk_index']}"
            payload.append((chunk_id, vector, chunk.metadata))

        index.upsert(vectors=payload)
    except Exception as exc:
        raise RuntimeError(f"Failed to upsert vectors into Pinecone: {exc}") from exc


def main():
    try:
        documents = load_documents(DATA_DIR)
        print("Documents loaded")
        classifier = get_ai_classifier()
        chunks = split_documents(documents, classifier)
        print("Chunks created")

        texts = [chunk.page_content for chunk in chunks]
        vectors = embed_texts(texts)

        pc = get_pinecone_client()
        index = get_index(pc, INDEX_NAME)
        upsert_to_pinecone(index, chunks, vectors)

        print(
            f"Successfully ingested {len(chunks)} chunks from {len(documents)} document(s) into Pinecone index '{INDEX_NAME}'."
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
