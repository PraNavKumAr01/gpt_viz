import time
import numpy as np
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Optional
import pickle

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

class SentenceTransformerHandler:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_dir: Optional[str] = None):
        """
        Handles loading and caching of a SentenceTransformer model locally.

        Args:
            model_name (str): HuggingFace model ID.
            model_dir (str | None): Local directory to save/load model.
        """
        if model_dir is None:
            model_dir = Path.cwd() / "models" / model_name
        self.model_name = model_name
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model: SentenceTransformer | None = None
        self.load_or_download_model()

    def load_or_download_model(self):
        """Load model from local storage, or download if missing."""
        try:
            if self.model_dir.exists() and any(self.model_dir.iterdir()):
                print("Loading SentenceTransformer model from local storage...")
                self.model = SentenceTransformer(str(self.model_dir))
            else:
                print("Downloading SentenceTransformer model...")
                self.model = SentenceTransformer(self.model_name)
                print("Saving SentenceTransformer model to local storage...")
                self.model.save(str(self.model_dir))
            print("✓ SentenceTransformer model ready!")
        except Exception as e:
            print(f"✗ Error loading/downloading model: {e}")
            raise

    def get_model(self) -> SentenceTransformer:
        return self.model


class FAISSTextSimilarityAnalyzer:
    def __init__(self, model_handler: SentenceTransformerHandler, similarity_threshold: float = 0.8):
        """
        Analyzer that builds FAISS index and performs similarity search.

        Args:
            model_handler (SentenceTransformerHandler): Wrapper for model loading.
            similarity_threshold (float): Minimum similarity score for results.
        """
        self.similarity_threshold = similarity_threshold
        self.model = model_handler.get_model()
        self.index: faiss.Index | None = None
        self.ids: list[str] | None = None
        self.embeddings: np.ndarray | None = None

    def create_embeddings(self, texts: list[str], ids: list[str]) -> np.ndarray:
        print(f"Creating embeddings for {len(texts)} texts...")
        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = embeddings.astype(np.float32)
        self.ids = ids
        print(f"✓ Created embeddings in {time.time() - start_time:.2f}s | Shape: {self.embeddings.shape}")
        return self.embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        print("Building FAISS index...")
        start_time = time.time()
        dim = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        if n_vectors < 10000:
            index = faiss.IndexFlatIP(dim)
            print("Using IndexFlatIP (exact search)")
        else:
            nlist = min(int(np.sqrt(n_vectors)), 1000)
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist)
            print(f"Using IndexIVFFlat with {nlist} clusters (approximate search)")
            print("Training FAISS index...")
            index.train(embeddings)

        index.add(embeddings)
        self.index = index
        print(f"✓ Built FAISS index in {time.time() - start_time:.2f}s")
        return index

    def search_similarities(self, top_k: Optional[int] = None) -> list[dict]:
        if self.index is None or self.embeddings is None:
            raise ValueError("Index not built. Call build_faiss_index first.")

        n_vectors = len(self.embeddings)
        if top_k is None:
            if n_vectors < 1000:
                top_k = n_vectors - 1
            elif n_vectors < 10000:
                top_k = min(500, n_vectors - 1)
            else:
                expected_ratio = max(0.01, (1.0 - self.similarity_threshold) * 0.1)
                top_k = min(int(n_vectors * expected_ratio), 1000)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = min(50, self.index.nlist)

        similarities, indices = self.index.search(self.embeddings, top_k + 1)
        results = []

        for i in range(n_vectors):
            for j in range(top_k + 1):
                neighbor_idx = indices[i][j]
                score = similarities[i][j]
                if neighbor_idx != i and score >= self.similarity_threshold and i < neighbor_idx:
                    results.append({
                        "source": self.ids[i],
                        "target": self.ids[neighbor_idx],
                        "weight": round(float(score), 4),
                    })
        return results

    def categorize_texts(self, category_index_path: str = 'category_index.faiss', 
                    category_metadata_path: str = 'category_metadata.pkl') -> list[dict]:
        """
        Categorize all processed texts using the pre-built category index.
        
        Args:
            category_index_path (str): Path to the saved category FAISS index
            category_metadata_path (str): Path to the saved category metadata pickle file
            
        Returns:
            list[dict]: List of dictionaries with 'id' and 'cluster' (category) for each text
        """
        if self.embeddings is None or self.ids is None:
            raise ValueError("No embeddings found. Process texts first with create_embeddings().")
        
        print("Loading category index and metadata...")
        
        try:
            category_index = faiss.read_index(category_index_path)
            with open(category_metadata_path, 'rb') as f:
                categories = pickle.load(f)
            print(f"✓ Loaded category index with {len(categories)} categories")
        except Exception as e:
            raise ValueError(f"Failed to load category files: {e}")
        
        print(f"Categorizing {len(self.embeddings)} texts...")
        
        similarities, indices = category_index.search(self.embeddings, 1)
        
        results = []
        for i in range(len(self.embeddings)):
            category_idx = indices[i][0] 
            similarity_score = similarities[i][0]
            
            if category_idx < len(categories):
                category_name = categories[category_idx]['category']
            else:
                category_name = "Unknown"
            
            results.append({
                'id': self.ids[i],
                'cluster': category_name,
                'confidence': round(float(similarity_score), 4)
            })
        
        print(f"✓ Categorized {len(results)} texts")
        return results
    
    def save_index(self, filepath: str):
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        faiss.write_index(self.index, filepath)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath: str):
        self.index = faiss.read_index(filepath)
        print(f"Index loaded from {filepath}")