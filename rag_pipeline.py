# rag_pipeline.py

import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI


ROOT = Path(__file__).resolve().parent


def build_query_engine(top_k: int = 3, cutoff: float = 0.5):
    load_dotenv(ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    documents = SimpleDirectoryReader(str(ROOT / "articles")).load_data()
    index = VectorStoreIndex.from_documents(documents)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)],
    )
