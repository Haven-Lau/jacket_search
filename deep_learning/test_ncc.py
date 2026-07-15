import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    [52.49, 54.61],    # Top-Left
    [170.81, 53.70],   # Top-Right
    [111.45, 111.73],  # Center
    [51.71, 170.58],   # Bottom-Left
    [171.37, 170.83]   # Bottom-Right
], dtype=np.float32)

def compute_ncc(img1, img2, mask):
    """Computes Normalized Cross-Correlation (NCC) between two RGB images under a boolean mask."""
    # Extract masked pixels: shape (N, 3)
    pixels1 = img1[mask].astype(np.float32)
    pixels2 = img2[mask].astype(np.float32)
    
    # Calculate NCC channel-wise and average
    ncc_channels = []
    for c in range(3):
        p1 = pixels1[:, c]
        p2 = pixels2[:, c]
        
        # Center the signals
        p1_mean = np.mean(p1)
        p2_mean = np.mean(p2)
        p1_centered = p1 - p1_mean
        p2_centered = p2 - p2_mean
        
        # Calculate correlation
        num = np.sum(p1_centered * p2_centered)
        den = np.sqrt(np.sum(p1_centered**2) * np.sum(p2_centered**2))
        
        if den < 1e-6:
            ncc = 0.0
        else:
            ncc = num / den
        ncc_channels.append(ncc)
        
    return np.mean(ncc_channels)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    mask_path = os.path.join("masks", "5-dot-mask.png")
    jackets_dir = "jackets"
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    # 1. Load images and mask
    img_query = cv2.imread(query_path)
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    img_query = cv2.resize(img_query, (224, 224))
    
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    # We want a slightly eroded mask to completely ignore border pixels where color bleed occurs!
    mask_np = np.array(mask_img)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask_np = cv2.erode(mask_np, kernel, iterations=2) > 127
    
    # 2. Warp Query to align query dots with template dots
    M, inliers = cv2.estimateAffinePartial2D(QUERY_PTS, TEMPLATE_PTS)
    warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    # 3. Match against reference jackets
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Matching query against {len(image_files)} reference jackets using NCC in pixel space...")
    
    results = []
    for filename in tqdm(image_files, desc="Matching jackets"):
        filepath = os.path.join(jackets_dir, filename)
        try:
            ref_img = cv2.imread(filepath)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = cv2.resize(ref_img, (224, 224))
            
            # Compute NCC under the eroded mask
            score = compute_ncc(warped_query, ref_img, eroded_mask_np)
            results.append((filename, score))
        except Exception as e:
            # print(f"Error {filename}: {e}")
            pass
            
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Print top 10 matches
    print("\n=== Top 10 Matches using Direct Pixel NCC ===")
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
