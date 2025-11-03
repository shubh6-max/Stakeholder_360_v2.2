import os

# Azure OpenAI
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4-vision-preview")

# Embeddings
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")
AZURE_EMBED_VERSION = os.getenv("AZURE_EMBED_VERSION", "2023-05-15")

# Retrieval
DEFAULT_CANDIDATES_TOP_K = int(os.getenv("RAG_TOP_K", "10"))  # ← edit here or via env
MAX_RERANK_INPUT_CHARS = 12000  # safety cap for reranker prompt
