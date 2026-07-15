import os
import sys
import json
import torch
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
    
    img = Image.open(query_path).convert("RGB")
    width, height = img.size
    
    # Sample background color near corners (e.g. top-left corner)
    bg_pixels = [img.getpixel((x, y)) for x in range(10) for y in range(10)]
    avg_bg = np.mean(bg_pixels, axis=0).astype(int)
    print(f"Query top-left corner average color (RGB): {avg_bg} (Hex: #{avg_bg[0]:02x}{avg_bg[1]:02x}{avg_bg[2]:02x})")
    
    # Load database
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallFeatureExtractor().eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check what the rank of the correct song currently is
    img_resized = img.resize((224, 224))
    tensor = preprocess(img_resized).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).squeeze(0).cpu().numpy()
        
    results = []
    for filename, variants in db.items():
        ref_emb = variants.get("5dot")
        if ref_emb:
            sim = sum(q * r for q, r in zip(query_emb, ref_emb))
            results.append((filename, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Find rank of correct song
    correct_rank = -1
    correct_score = -1
    for rank, (filename, score) in enumerate(results):
        if filename == correct_song:
            correct_rank = rank + 1
            correct_score = score
            break
            
    print(f"\nBefore query masking:")
    print(f"  Correct song '{correct_song}' rank: {correct_rank}/{len(results)}")
    print(f"  Correct song similarity score: {correct_score:.5f}")
    
    # Let's try to clean the query:
    # If the background color is approximately the game's mask color, let's try replacing it with exact #22008e.
    # In the query image, the background is blue. We can do simple color thresholding to segment the dots from the blue background!
    # Let's see if we can do this. The blue background has high B channel and low R/G channels.
    # Specifically, for RGB = (r, g, b), the blue background has b > 120 and r < 80.
    # Let's print some pixel values from the center of a dot vs the background.
    print("\nSampling center pixels vs background pixels:")
    img_np = np.array(img_resized)
    # Center dot is around x=112, y=112
    print(f"  Center pixel at (112, 112): {img_np[112, 112]}")
    # Top-left dot is around x=50, y=50
    print(f"  Top-left dot pixel at (50, 50): {img_np[50, 50]}")
    # Background pixel at (10, 10)
    print(f"  Background pixel at (10, 10): {img_np[10, 10]}")

if __name__ == "__main__":
    main()
