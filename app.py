import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pinecone import Pinecone
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
TOP_K = 15

SYSTEM_PROMPT = (
    "You are a Medium-article assistant. You answer questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must never use external knowledge, the open internet, "
    "or any information not explicitly contained in the retrieved context. "

    "When answering, follow these rules: "

    "1. FIND A SPECIFIC ARTICLE: Identify the single most relevant article and "
    "return the requested fields (title, author, etc.) directly from the metadata. "
    "If a field like author is missing from the context, say so clearly but still "
    "provide the other fields you do have. "

    "2. LIST MULTIPLE ARTICLES: Return the requested number of DISTINCT articles "
    "with different article IDs. Never list chunks from the same article as "
    "separate results. "

    "3. SUMMARISE AN ARTICLE: Identify the most relevant article and provide a "
    "concise summary of its central argument based strictly on the retrieved chunks. "

    "4. RECOMMEND AN ARTICLE: Choose the single most relevant article and justify "
    "your recommendation by referencing specific passages from the retrieved context. "

    "IMPORTANT RESPONSE RULES: "
    "If relevant context exists, answer directly and confidently without any "
    "disclaimers. Only respond with 'I don't know based on the provided Medium "
    "articles data.' if the context contains absolutely nothing relevant to the "
    "question — and when you do, stop there without continuing to answer. "
    "Always ground your answer in the retrieved context by quoting or paraphrasing "
    "specific passages."
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


class PromptRequest(BaseModel):
    question: str


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/api/stats")
async def stats():
    return {
        "chunk_size": 512,
        "overlap_ratio": 0.2,
        "top_k": TOP_K,
    }


@app.post("/api/prompt")
async def prompt(body: PromptRequest):
    question = body.question

    try:
        embed_response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
        )
        question_vector = embed_response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        query_result = index.query(
            vector=question_vector,
            top_k=TOP_K,
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    matches = query_result.get("matches", [])

    context_items = []
    for match in matches:
        meta = match.get("metadata") or {}
        context_items.append({
            "article_id": str(meta.get("article_id", "")),
            "title": str(meta.get("title", "")),
            "chunk": str(meta.get("text", "")),
            "score": match.get("score", 0.0),
        })

    context_block = "\n\n".join(
        f"[Article ID: {c['article_id']}] {c['title']}\n{c['chunk']}"
        for c in context_items
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    try:
        chat_response = await openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = chat_response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")

    return {
        "response": answer,
        "context": context_items,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }