import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

def translate_image(image_np, dx, dy, fill_value):
    """Translates a 2D or 3D numpy array by dx, dy, filling empty space with fill_value."""
    h, w = image_np.shape[:2]
    translated = np.full_like(image_np, fill_value)
    
    # Calculate source and destination slices
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    
    if src_y_start < src_y_end and src_x_start < src_x_end:
        translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_np[src_y_start:src_y_end, src_x_start:src_x_end]
        
    return translated

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    db_path = os.path.join("deep_learning", "embeddings.json")
    mask_path = os.path.join("masks", "5-dot-mask.png")
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    # 1. Load images and resize
    img_query = Image.open(query_path).convert("RGB").resize((224, 224))
    query_np = np.array(img_query)
    
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    mask_np = np.array(mask_img) > 127  # Binary boolean mask
    
    # 2. Segment the dots in the query image
    # Sample background color from corners
    corners = [
        query_np[0:10, 0:10],
        query_np[0:10, -10:],
        query_np[-10:, 0:10],
        query_np[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    print(f"Detected background color (RGB): {bg_color.astype(int)}")
    
    # Calculate Euclidean distance from background color
    dist = np.linalg.norm(query_np - bg_color, axis=-1)
    
    # Pixels far from background color are considered dots
    query_dot_mask = dist > 70
    
    # 3. Sweep translations to find the best alignment
    best_overlap = -1
    best_dx = 0
    best_dy = 0
    
    # Search range: [-30, 30] pixels
    sweep_range = 30
    for dy in range(-sweep_range, sweep_range + 1):
        for dx in range(-sweep_range, sweep_range + 1):
            # Translate query dot mask
            translated_mask = translate_image(query_dot_mask, dx, dy, False)
            
            # Compute overlap (intersection)
            overlap = np.sum(translated_mask & mask_np)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_dx = dx
                best_dy = dy
                
    print(f"Optimal alignment found: dx={best_dx}, dy={best_dy} (Overlap pixels: {best_overlap})")
    
    # 4. Apply optimal translation to query image
    # Note: Fill background with #22008e directly during translation so shifted-out borders are colored correctly
    target_bg_color = [34, 0, 142]  # #22008e
    translated_query = translate_image(query_np, best_dx, best_dy, target_bg_color)
    
    # 5. Mask the translated query
    # Keep query content where mask is True, write target_bg_color where mask is False
    aligned_masked_np = np.where(mask_np[:, :, np.newaxis], translated_query, target_bg_color)
    aligned_masked_img = Image.fromarray(aligned_masked_np.astype(np.uint8))
    
    # Save the aligned query image for debug inspection
    aligned_query_path = os.path.join("deep_learning", "aligned_query_1.png")
    aligned_masked_img.save(aligned_query_path)
    print(f"Saved aligned query image to {aligned_query_path}")
    
    # 6. Load database and model
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallFeatureExtractor().eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 7. Get query embedding
    tensor = preprocess(aligned_masked_img).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).squeeze(0).cpu().numpy()
        
    # 8. Compute similarities
    results = []
    for filename, variants in db.items():
        ref_emb = variants.get("5dot")
        if ref_emb:
            sim = sum(q * r for q, r in zip(query_emb, ref_emb))
            results.append((filename, sim))
            
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 9. Print Top 10 matches
    print("\n=== Top 10 Matches after Alignment & Masking ===")
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
