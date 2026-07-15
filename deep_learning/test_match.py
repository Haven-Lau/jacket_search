import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

def main():
    # Force stdout and stderr to use UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    db_path = os.path.join("deep_learning", "embeddings.json")
    
    if not os.path.exists(query_path):
        print(f"Error: Query image not found at {query_path}")
        return
        
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    print(f"Loading query image: {query_path}")
    img_query = Image.open(query_path).convert("RGB").resize((224, 224))
    
    print("Loading database...")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = MobileNetV3SmallFeatureExtractor()
    model.eval()
    model.to(device)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Preprocess and get embedding
    tensor = preprocess(img_query).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model(tensor).squeeze(0).cpu().numpy()
        
    # Match against the 5dot variant of all database jackets
    print("Computing similarities against '5dot' database...")
    results = []
    for filename, variants in db.items():
        ref_embedding = variants.get("5dot")
        if ref_embedding is None:
            continue
            
        # Cosine similarity is dot product because both vectors are L2-normalized
        similarity = sum(q * r for q, r in zip(query_embedding, ref_embedding))
        results.append((filename, similarity))
        
    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n=== Top 10 Matches (Using raw 5-dot embeddings) ===")
    for i, (filename, score) in enumerate(results[:10]):
        status = "CORRECT" if filename == "FOR_YOUR_BRAVE!!.png" else "WRONG"
        print(f"{i+1}. {filename:<50} | Score: {score:.5f} | [{status}]")

if __name__ == "__main__":
    main()
