import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    db_path = os.path.join("deep_learning", "embeddings.json")
    mask_path = os.path.join("masks", "5-dot-mask.png")
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    # 1. Load images and mask
    img_query = Image.open(query_path).convert("RGB").resize((224, 224))
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    
    # 2. Mask the query: replace the background with solid #22008e
    bg_color = (34, 0, 142)  # #22008e in RGB
    bg_img = Image.new("RGB", (224, 224), bg_color)
    
    # Paste query dot regions onto the database background color
    img_query_masked = bg_img.copy()
    img_query_masked.paste(img_query, (0, 0), mask_img)
    
    # Save the cleaned query for debug visual inspection
    debug_query_path = os.path.join("deep_learning", "debug_query_masked_clean.png")
    img_query_masked.save(debug_query_path)
    print(f"Saved cleaned query image to {debug_query_path}")
    
    # 3. Load database and model
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallFeatureExtractor().eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Get query embedding
    tensor = preprocess(img_query_masked).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).squeeze(0).cpu().numpy()
        
    # 5. Compute similarities
    results = []
    for filename, variants in db.items():
        ref_emb = variants.get("5dot")
        if ref_emb:
            sim = sum(q * r for q, r in zip(query_emb, ref_emb))
            results.append((filename, sim))
            
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Print Top 10 matches
    print("\n=== Top 10 Matches after Query Masking ===")
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
