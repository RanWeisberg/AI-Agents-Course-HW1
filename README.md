# Medium Articles RAG Assistant

## What is this?

This is an AI assistant that can answer questions about Medium articles. It uses a technique called RAG (Retrieval-Augmented Generation) — instead of relying on the AI's own knowledge, it searches through a database of ~7,600 real Medium articles and answers based only on what it finds there.

## Live URL

https://ai-agents-khaki.vercel.app

## How it works

1. All 7,600 articles were split into chunks and stored in a Pinecone vector database
2. When you ask a question, the system finds the most relevant chunks
3. Those chunks are handed to the AI model, which answers using only that content — nothing else


## Hyperparameter choices

- **chunk_size 512** — each article is split into pieces of 512 tokens. Small enough to be precise, large enough to contain a full idea.
- **overlap_ratio 0.2** — chunks overlap by 20% so ideas that fall at the boundary between two chunks are not lost.
- **top_k 15** — the system retrieves 15 chunks per question. This gives enough variety to find 3 distinct articles when needed, without overloading the model.

## How to run locally

Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file with your keys:
```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_base_url
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=medium-articles
```

Start the server:
```bash
uvicorn app:app --reload --port 8000
```