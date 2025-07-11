from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_index = faiss.IndexFlatL2(384)
embedding_metadata = []
