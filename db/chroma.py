"""
db/chroma.py
────────────
Shared ChromaDB client — PersistentClient with OpenAI embeddings.
Lazy-initialized so the API key is read after load_dotenv() runs.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
        _collection = _client.get_or_create_collection(
            name="nba_team_context",
            embedding_function=embedding_fn,
        )
    return _collection
