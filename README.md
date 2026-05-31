# Medium Articles RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions strictly 
based on a dataset of ~7,600 English Medium articles. The system embeds article 
chunks into a Pinecone vector database and uses a chat model to answer questions 
using only the retrieved context.

## Live URL

https://ai-agents-khaki.vercel.app

## How It Works

1. Articles from the Medium dataset are chunked and embedded using 
   `4UHRUIN-text-embedding-3-small`
2. Embeddings are stored in a Pinecone vector index
3. When a question is received, it is embedded and used to retrieve 
   the most relevant chunks from Pinecone
4. The retrieved chunks are passed as context to `4UHRUIN-gpt-5-mini`, 
   which answers strictly based on the provided context

## API Endpoints

### POST /api/prompt
Send a natural language question and receive an answer grounded in the 
Medium articles dataset.

**Input:**
```json
{
  "question": "Your natural language question here"
}
```

**Output:**
```json
{
  "response": "Final natural language answer from the model.",
  "context": [
    {
      "article_id": "1234",
      "title": "Sample article title",
      "chunk": "The retrieved article passage",
      "score": 0.85
    }
  ],
  "Augmented_prompt": {
    "System": "The system prompt used to query the chat model",
    "User": "The user prompt with context injected"
  }
}
```

### GET /api/stats
Returns the RAG hyperparameters used by the system.

**Output:**
```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 15
}
```

### GET /
Health check endpoint.

**Output:**
```json
{"status": "ok"}
```

## RAG Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| `chunk_size` | 512 tokens | Balances context richness with retrieval precision. Large enough to capture a complete idea, small enough to stay focused |
| `overlap_ratio` | 0.2 (20%) | Prevents losing context at chunk boundaries. 102 token overlap ensures ideas that span two chunks are not missed |
| `top_k` | 15 | Retrieves enough diverse chunks to handle multi-result queries (e.g. "list 3 articles about X") without flooding the model with irrelevant context |

## Dataset

- ~7,600 English-language articles published on Medium
- CSV columns: `title`, `text`, `url`, `authors`, `timestamp`, `tags`
- Total chunks embedded: 27,426
- Embedding dimensions: 1536 (text-embedding-3-small)

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-agents.git
cd ai-agents
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a .env file
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=medium-articles
```

### 4. Run the server
```bash
uvicorn app:app --reload --port 8000
```

### 5. Test it
```bash
curl http://localhost:8000/api/stats

curl -X POST http://localhost:8000/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "List 3 articles about education"}'
```

## Embedding the Dataset (optional)
To re-embed the dataset from scratch:
```bash
python embed.py --limit 0
```
To overwrite existing vectors:
```bash
python embed.py --limit 0 --reupload
```

## Tech Stack
- **FastAPI** — API server
- **Pinecone** — Vector database
- **OpenAI-compatible API** — Embeddings and chat completion
- **Vercel** — Deployment
- **tiktoken** — Token counting for chunking