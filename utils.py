import re
import json
from fastapi import UploadFile, HTTPException

def extract_json(file_path: str) -> list:
    with open(file_path, "r") as f:
        return json.load(f)

def extract_json_from_upload(uploaded_file: UploadFile) -> list:
    """Extract JSON content from uploaded file instead of file path."""
    try:
        content = uploaded_file.file.read()
        data = json.loads(content.decode("utf-8"))
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"[\*#\-]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_label_and_text(conversations: list[dict]) -> list[dict]:
    output = []
    for idx, convo in enumerate(conversations, start=1):
        node_id = f"node{idx}"
        label = convo.get("label", "")
        text_parts = []
        for node in convo.get("mapping", {}).values():
            msg = node.get("message")
            if msg and "content" in msg:
                for part in msg["content"].get("parts", []):
                    if isinstance(part, str):
                        text_parts.append(part)
                    else:
                        try:
                            text_parts.append(json.dumps(part, ensure_ascii=False))
                        except Exception:
                            text_parts.append(str(part))
        raw_text = " ".join(text_parts)
        text = clean_text(raw_text)
        output.append({"id": node_id, "label": label, "Text": text})
    return output

def build_graph_json(nodes: list[dict], edges: list[dict], categories: list[dict]) -> dict:
    """
    Build a graph JSON structure from extracted nodes, similarity edges, and categories.

    Args:
        nodes (list[dict]): Output from extract_label_and_text()
            Example: [{"id": "node1", "label": "Example", "Text": "Some text"}, ...]
        edges (list[dict]): Output from FAISSTextSimilarityAnalyzer.search_similarities()
            Example: [{"source": "node1", "target": "node2", "weight": 0.87}, ...]
        categories (list[dict]): Output from categorize_texts()
            Example: [{"id": "node1", "cluster": "Coding", "confidence": 0.92}, ...]

    Returns:
        dict: Graph JSON with 'nodes' and 'edges'
    """
    
    category_lookup = {cat["id"]: cat for cat in categories}
    
    enriched_nodes = []
    for n in nodes:
        node_id = n["id"]
        node_data = {
            "id": node_id,
            "label": n.get("label", "")
        }
        
        if node_id in category_lookup:
            node_data["cluster"] = category_lookup[node_id]["cluster"]
            node_data["confidence"] = category_lookup[node_id]["confidence"]
        else:
            node_data["cluster"] = "Unknown"
            node_data["confidence"] = 0.0
        
        enriched_nodes.append(node_data)

    return {
        "nodes": enriched_nodes,
        "edges": edges
    }