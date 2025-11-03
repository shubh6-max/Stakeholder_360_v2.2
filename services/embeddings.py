from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from config.settings import (
    AZURE_ENDPOINT, AZURE_API_KEY, AZURE_EMBED_DEPLOYMENT, AZURE_EMBED_VERSION
)

_embedder = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBED_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=AZURE_EMBED_VERSION,
)

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _embedder.embed_documents(texts)

def embed_query(text: str) -> List[float]:
    return _embedder.embed_query(text)
