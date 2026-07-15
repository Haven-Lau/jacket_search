import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

# Centroids of the 5 dots in 224x224 coordinates
DOT_CENTROIDS = [
    (51, 53),   # Top-Left
    (169, 53),  # Top-Right
    (110, 111), # Center
    (51, 171),  # Bottom-Left
    (168, 171)  # Bottom-Right
]
PATCH_SIZE = 40

def make_circular_patch(patch):
    """Masks a 40x40 patch to be circular with a black background outside the circle."""
    h, w = patch.shape[:2]
    Y, X = np.ogrid[:h, :w]
    # Center is at (19.5, 19.5) for a 40x40 patch
    dist_from_center = np.sqrt((X - 19.5)**2 + (Y - 19.5)**2)
    # Circle radius of 17.5 pixels to strictly crop within the dot boundaries
    mask = dist_from_center <= 17.5
    
    masked_patch = np.zeros_like(patch)
    masked_patch[mask] = patch[mask]
    return masked_patch

def build_patch_grid(img_224, apply_circular_mask=True):
    """Crops 5 patches around the centroids, masks them to circular shapes, and builds a 120x80 grid."""
    img_np = np.array(img_224)
    patches = []
    
    half_size = PATCH_SIZE // 2
    for cx, cy in DOT_CENTROIDS:
        y_start = max(0, cy - half_size)
        y_end = min(224, cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(224, cx + half_size)
        
        patch = img_np[y_start:y_end, x_start:x_end]
        if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
            
        if apply_circular_mask:
            patch = make_circular_patch(patch)
            
        patches.append(patch)
        
    grid = np.zeros((PATCH_SIZE * 3, PATCH_SIZE * 2, 3), dtype=np.uint8)
    grid[0:40, 0:40] = patches[0]
    grid[0:40, 40:80] = patches[1]
    grid[40:80, 0:40] = patches[2]
    grid[40:80, 40:80] = patches[3]
    grid[80:120, 0:40] = patches[4]
    
    return Image.fromarray(grid)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    jackets_dir = "jackets"
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    img_query = Image.open(query_path).convert("RGB").resize((224, 224))
    query_np = np.array(img_query)
    
    # Apply translation dx=-1, dy=-2 based on dot analysis shifts
    h, w = query_np.shape[:2]
    translated_query = np.full_like(query_np, 0)
    dx, dy = -1, -2
    
    src_y_start = max(0, -dy)
    src_y_end = min(h, h - dy)
    src_x_start = max(0, -dx)
    src_x_end = min(w, w - dx)
    dst_y_start = max(0, dy)
    dst_y_end = min(h, h + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(w, w + dx)
    translated_query[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = query_np[src_y_start:src_y_end, src_x_start:src_x_end]
    
    # Build query grid
    query_grid = build_patch_grid(Image.fromarray(translated_query), apply_circular_mask=True)
    query_grid.save("deep_learning/debug_query_grid_circular.png")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallFeatureExtractor().eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get query embedding
    tensor = preprocess(query_grid).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model(tensor).squeeze(0).cpu().numpy()
        
    # Match against reference jackets
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Matching against {len(image_files)} reference jackets using circular masks...")
    
    results = []
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Matching jackets"):
            filepath = os.path.join(jackets_dir, filename)
            try:
                ref_img = Image.open(filepath).convert("RGB").resize((224, 224))
                ref_grid = build_patch_grid(ref_img, apply_circular_mask=True)
                
                if filename == correct_song:
                    ref_grid.save("deep_learning/debug_correct_ref_grid_circular.png")
                    
                ref_tensor = preprocess(ref_grid).unsqueeze(0).to(device)
                ref_emb = model(ref_tensor).squeeze(0).cpu().numpy()
                
                sim = sum(q * r for q, r in zip(query_emb, ref_emb))
                results.append((filename, sim))
            except Exception as e:
                print(f"Error {filename}: {e}")
                
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 10 matches
    print("\n=== Top 10 Matches using Circular Patch-Grid ===")
    correct_rank = -1
    for i, (filename, score) in enumerate(results[:10]):
        status = "CORRECT" if filename == correct_song else "WRONG"
        if filename == correct_song:
            correct_rank = i + 1
        print(f"{i+1}. {filename:<50} | Score: {score:.5f} | [{status}]")
        
    if correct_rank == -1:
        for rank, (filename, score) in enumerate(results):
            if filename == correct_song:
                print(f"...\n{rank+1}. {filename:<50} | Score: {score:.5f} | [CORRECT]")
                break

if __name__ == "__main__":
    main()
