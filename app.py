from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from utils import extract_label_and_text, build_graph_json, extract_json_from_upload
from embeddings_funcs import FAISSTextSimilarityAnalyzer, SentenceTransformerHandler

model_handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    global model_handler

    print("🚀 Loading SentenceTransformer model at startup...")
    model_handler = SentenceTransformerHandler()
    print("✓ Model loaded and ready!")
    
    yield

    print("🔄 Shutting down...")
    model_handler = None

app = FastAPI(
    title="Graph Builder API",
    lifespan=lifespan
)

@app.post("/process_json/")
async def process_json(file: UploadFile = File(...)):
    """
    Upload a JSON file with conversations,
    build embeddings, search similarities, and return graph JSON.
    """
    global model_handler
    
    if model_handler is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded. Please restart the server."}
        )

    try:
        data = extract_json_from_upload(file)
        nodes = extract_label_and_text(data)

        analyzer = FAISSTextSimilarityAnalyzer(model_handler, similarity_threshold=0.8)
        embeddings = analyzer.create_embeddings(
            [d["Text"] for d in nodes], [d["id"] for d in nodes]
        )
        analyzer.build_faiss_index(embeddings)
        edges = analyzer.search_similarities()
        categories = analyzer.categorize_texts()

        result = build_graph_json(nodes, edges, categories)
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify model status.
    """
    global model_handler
    return {
        "status": "healthy" if model_handler is not None else "model_not_loaded",
        "model_ready": model_handler is not None
    }