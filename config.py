"""Shared settings for the TN Schemes Knowledge-Graph RAG chatbot."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                     # put saved .html / .txt / .pdf pages here
PAGES_JSON = DATA_DIR / "pages.json"           # output of ingest.py
SCHEMES_JSON = DATA_DIR / "schemes.json"       # structured schemes (LLM extraction)
GRAPH_JSON = DATA_DIR / "graph.json"           # NetworkX node-link graph
CYPHER_FILE = DATA_DIR / "graph.cypher"        # optional Neo4j import script
FAISS_DIR = DATA_DIR / "faiss_index"

# Any LangChain provider:model string, e.g. "openai:gpt-4o-mini",
# "groq:llama-3.3-70b-versatile", "google_genai:gemini-2.0-flash"
LLM_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 3500        # characters sent to the LLM per extraction call
CHUNK_OVERLAP = 300


def get_llm(temperature: float = 0):
    from langchain.chat_models import init_chat_model
    return init_chat_model(LLM_MODEL, temperature=temperature)


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
