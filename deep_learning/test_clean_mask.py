import os
import sys
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    db_path = os.path.join("deep_learning", "embeddings.json")
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    # 1. Load query image
    img = Image.open(query_path).convert("RGB").resize((224, 224))
    query_np = np.array(img)
    
    # 2. Segment dots using distance threshold from corners
    corners = [
        query_np[0:10, 0:10],
        query_np[0:10, -10:],
        query_np[-10:, 0:10],
        query_np[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    dist = np.linalg.norm(query_np - bg_color, axis=-1)
    
    # Generate binary mask of the query dots
    query_dot_mask = (dist > 70).astype(np.uint8)
    
    # 3. Erode the mask by 2 pixels to completely remove any border background color
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(query_dot_mask, kernel, iterations=2)
    
    # 4. Mask the query onto #22008e background
    target_bg_color = [34, 0, 142]  # #22008e
    
    # Create the masked query image array
    clean_query_np = np.where(eroded_mask[:, :, np.newaxis] > 0, query_np, target_bg_color)
    clean_query_img = Image.fromarray(clean_query_np.astype(np.uint8))
    
    # Save the cleaned query
    debug_query_path = os.path.join("deep_learning", "debug_query_eroded_masked.png")
    clean_query_img.save(debug_query_path)
    print(f"Saved cleaned query image to {debug_query_path}")
    
    # 5. Load database and model
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallFeatureExtractor().eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 6. Get query embedding
    tensor = preprocess(clean_query_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).squeeze(0).cpu().numpy()
        
    # 7. Compute similarities
    results = []
    for filename, variants in db.items():
        ref_emb = variants.get("5dot")
        if ref_emb:
            sim = sum(q * r for q, r in zip(query_emb, ref_emb))
            results.append((filename, sim))
            
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 8. Print Top 10 matches
    print("\n=== Top 10 Matches with Dynamic Segmented & Eroded Mask ===")
    correct_rank = -1
    for i, (filename, score) in enumerate(results[:10]):
        status = "CORRECT" if filename == correct_song else "WRONG"
        if filename == correct_song:
            correct_rank = i + 1
        print(f"{i+1}. {filename:<50} | Score: {score:.5f} | [{status}]")
        
    if correct_rank == -1:
        # Find where it is
        for rank, (filename, score) in enumerate(results):
            if filename == correct_song:
                print(f"...\n{rank+1}. {filename:<50} | Score: {score:.5f} | [CORRECT]")
                break

if __name__ == "__main__":
    main()
