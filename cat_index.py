import faiss
import numpy as np
import pickle
import time
from sentence_transformers import SentenceTransformer
from utils import extract_json

cat_data = extract_json('categories.json')

categories = cat_data.get('categories', [])

def build_faiss_index(embeddings):
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

    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"✓ Built FAISS index in {time.time() - start_time:.2f}s")
    return index

print("Loading SentenceTransformer model...")
model = SentenceTransformer("/Users/pranav/Documents/Projects/gpt_viz/models/all-MiniLM-L6-v2")

print("Creating embeddings...")
descriptions = [cat['description'] for cat in categories]
embeddings = model.encode(descriptions, show_progress_bar=True)
embeddings = embeddings.astype('float32')

index = build_faiss_index(embeddings)

print("Saving index to disk...")
faiss.write_index(index, 'category_index.faiss')

print("Saving category metadata...")
with open('category_metadata.pkl', 'wb') as f:
    pickle.dump(categories, f)

print("✓ Index and metadata saved successfully!")
print("Files created: category_index.faiss, category_metadata.pkl")

def search_categories(query, k=2):
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(categories):
            results.append((categories[idx]['category'], score))
    return results