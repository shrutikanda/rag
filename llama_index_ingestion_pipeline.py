import os
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

ROOT = Path(__file__).resolve().parent


def load_imdb_csv(path: str):
    df = pd.read_csv(path)

    documents = []

    for _, row in df.iterrows():
        text = f"""
        Title: {row['Series_Title']}
        Year: {row['Released_Year']}
        Genre: {row['Genre']}
        IMDB Rating: {row['IMDB_Rating']}
        Overview: {row['Overview']}
        Director: {row['Director']}
        """

        metadata = {
            "title": row["Series_Title"],
            "rating": row["IMDB_Rating"],
            "genre": row["Genre"],
            "year": row["Released_Year"],
        }

        documents.append(Document(text=text, metadata=metadata))

    return documents

def _filter_nodes_by_genre(nodes, genre: str):
    if not genre:
        return nodes
    out = []
    for n in nodes:
        md = getattr(n.node, "metadata", {}) or {}
        g = md.get("genre", "")
        # g is a string like "Crime, Drama"
        if isinstance(g, str) and genre.lower() in g.lower():
            out.append(n)
    return out


class GenreFilteringRetriever:
    """Small wrapper around any retriever that filters returned nodes by genre."""
    def __init__(self, retriever, genre: str | None):
        self.retriever = retriever
        self.genre = genre

    def retrieve(self, query):
        nodes = self.retriever.retrieve(query)
        return _filter_nodes_by_genre(nodes, self.genre) if self.genre else nodes


def build_query_engine(top_k: int = 5, cutoff: float = 0.5, genre_filter: str | None = "Crime"):
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found")

    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    docs = load_imdb_csv(ROOT / "articles" / "imdb_top_1000.csv")
    
    index = VectorStoreIndex.from_documents(docs)

    vector = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    
    print("Hybrid mode enabled (Vector + BM25)")
    bm25 = BM25Retriever.from_defaults(index=index, similarity_top_k=top_k)
    fused = QueryFusionRetriever(
          retrievers=[vector, bm25],
          similarity_top_k=top_k,
        )
    base_retriever = fused
    
    # Apply genre filter after retrieval (works across versions)
    filtered_retriever = GenreFilteringRetriever(base_retriever, genre_filter)

    return RetrieverQueryEngine(
        retriever=filtered_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=cutoff)],
    )