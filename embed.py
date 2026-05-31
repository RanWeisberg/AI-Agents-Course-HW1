import argparse
import logging
import math
import os
import time
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
EMBEDDING_DIM = 1536
CHUNK_SIZE = 512
CHUNK_OVERLAP = 102  # ~20% of 512
BATCH_SIZE = 100
TIKTOKEN_ENCODING = "cl100k_base"

logging.basicConfig(
    filename="skipped.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)


def chunk_text(text: str, enc: tiktoken.Encoding) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_with_retry(client: OpenAI, texts: list[str], retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"\nEmbedding API error (attempt {attempt + 1}/{retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return []  # unreachable


def setup_pinecone_index(pc: Pinecone) -> object:
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index created.")
    else:
        print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'.")
    return pc.Index(PINECONE_INDEX_NAME)


def existing_ids(index, ids: list[str]) -> set[str]:
    try:
        result = index.fetch(ids=ids)
        return set(result.vectors.keys())
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="Chunk and embed Medium articles into Pinecone.")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max articles to process (0 = no limit, default 500).")
    parser.add_argument("--reupload", action="store_true",
                        help="Skip deduplication and overwrite all existing vectors in Pinecone.")
    args = parser.parse_args()

    # --- Load CSV ---
    csv_path = Path("medium-english-50mb.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    if args.limit > 0:
        df = df.head(args.limit)

    print(f"Articles loaded: {len(df)}")

    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)

    # --- Build chunks ---
    print("Chunking articles...")
    all_chunks: list[dict] = []
    skipped_articles = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            logging.info(f"Skipped article idx={idx} title={row.get('title', 'N/A')!r}: empty or NaN text")
            skipped_articles += 1
            continue

        article_id = str(idx)
        chunks = chunk_text(text, enc)
        for chunk_index, chunk_text_val in enumerate(chunks):
            all_chunks.append({
                "id": f"{article_id}_chunk_{chunk_index}",
                "text": chunk_text_val,
                "metadata": {
                    "article_id": article_id,
                    "title": str(row.get("title", "")),
                    "authors": str(row.get("authors", "")),
                    "url": str(row.get("url", "")),
                    "tags": str(row.get("tags", "")),
                    "chunk_index": chunk_index,
                    "text": chunk_text_val,
                },
            })

    total_chunks = len(all_chunks)
    total_tokens = sum(len(enc.encode(c["text"])) for c in all_chunks)
    cost_estimate = (total_tokens / 1_000_000) * 0.02  # $0.02 per 1M tokens

    print(f"\nArticles with text: {len(df) - skipped_articles}")
    print(f"Total chunks:       {total_chunks}")
    print(f"Estimated tokens:   {total_tokens:,}")
    print(f"Estimated cost:     ~${cost_estimate:.4f} (text-embedding-3-small @ $0.02/1M tokens)")
    confirm = input("\nProceed with embedding? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # --- Pinecone setup ---
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = setup_pinecone_index(pc)

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # --- Embed + upload in batches ---
    vectors_uploaded = 0
    vectors_skipped = 0
    num_batches = math.ceil(total_chunks / BATCH_SIZE)

    for batch_num in tqdm(range(num_batches), desc="Embedding & uploading"):
        batch = all_chunks[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        batch_ids = [c["id"] for c in batch]

        # Deduplication: skip already-uploaded vectors (bypassed with --reupload)
        if args.reupload:
            new_batch = batch
        else:
            already_exists = existing_ids(index, batch_ids)
            new_batch = [c for c in batch if c["id"] not in already_exists]
            vectors_skipped += len(already_exists)

        if not new_batch:
            continue

        texts = [c["text"] for c in new_batch]
        embeddings = embed_with_retry(openai_client, texts)

        vectors = [
            {
                "id": chunk["id"],
                "values": emb,
                "metadata": chunk["metadata"],
            }
            for chunk, emb in zip(new_batch, embeddings)
        ]

        index.upsert(vectors=vectors)
        vectors_uploaded += len(vectors)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Articles processed : {len(df) - skipped_articles}")
    print(f"Articles skipped   : {skipped_articles} (see skipped.log)")
    print(f"Chunks created     : {total_chunks}")
    print(f"Vectors uploaded   : {vectors_uploaded}")
    print(f"Vectors skipped    : {vectors_skipped} (already in Pinecone)")


if __name__ == "__main__":
    main()