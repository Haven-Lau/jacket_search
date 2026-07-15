import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

# Template Centroids (224x224 coordinates)
TEMPLATE_PTS = np.array([
    [51.17, 52.83],    # Top-Left
    [168.84, 52.84],   # Top-Right
    [110.03, 111.18],  # Center
    [51.18, 170.69],   # Bottom-Left
    [168.34, 170.66]   # Bottom-Right
], dtype=np.float32)

# Query Centroids detected in tests/1.png
QUERY_PTS = np.array([
    [52.49, 54.61],    # Top-Left (Dot 1 in query list)
    [170.81, 53.70],   # Top-Right (Dot 0 in query list)
    [111.45, 111.73],  # Center (Dot 2 in query list)
    [51.71, 170.58],   # Bottom-Left (Dot 3 in query list)
    [171.37, 170.83]   # Bottom-Right (Dot 4 in query list)
], dtype=np.float32)

PATCH_SIZE = 40
DOT_CENTROIDS = [(51, 53), (169, 53), (110, 111), (51, 171), (168, 171)]

def make_circular_patch(patch):
    h, w = patch.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - 19.5)**2 + (Y - 19.5)**2)
    mask = dist_from_center <= 17.5
    masked_patch = np.zeros_like(patch)
    masked_patch[mask] = patch[mask]
    return masked_patch

def build_patch_grid(img_224):
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
            
        # Apply circular mask (so corners are black in both reference and query)
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
    
    # 1. Load query image
    img_query = cv2.imread(query_path)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    img_query = cv2.resize(img_query, (224, 224))
    
    # 2. Estimate Affine Transform & Warp Query to perfectly align with Template Centroids
    # estimateAffinePartial2D solves for translation, rotation, and uniform scale
    M, inliers = cv2.estimateAffinePartial2D(QUERY_PTS, TEMPLATE_PTS)
    print(f"Estimated Affine Transform matrix:\n{M}")
    
    warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # Build query grid from warped aligned image
    query_grid = build_patch_grid(warped_query)
    query_grid.save("deep_learning/debug_query_grid_affine.png")
    
    # 3. Load model
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
        
    # 4. Match against reference jackets
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Matching query against {len(image_files)} reference jackets using Affine Alignment...")
    
    results = []
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Matching jackets"):
            filepath = os.path.join(jackets_dir, filename)
            try:
                ref_img = cv2.imread(filepath)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_img = cv2.resize(ref_img, (224, 224))
                
                ref_grid = build_patch_grid(ref_img)
                
                if filename == correct_song:
                    ref_grid.save("deep_learning/debug_correct_ref_grid_affine.png")
                    
                ref_tensor = preprocess(ref_grid).unsqueeze(0).to(device)
                ref_emb = model(ref_tensor).squeeze(0).cpu().numpy()
                
                sim = sum(q * r for q, r in zip(query_emb, ref_emb))
                results.append((filename, sim))
            except Exception as e:
                # print(f"Error {filename}: {e}")
                pass
                
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 5. Print top 10 matches
    print("\n=== Top 10 Matches using Affine Warped Patch-Grid ===")
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
