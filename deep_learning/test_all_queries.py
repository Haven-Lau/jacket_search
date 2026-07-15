import os
import sys
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ideal Template Centroids (224x224 coordinates)
TEMPLATE_PTS = np.array([
    [51.17, 52.83],    # Top-Left
    [168.84, 52.84],   # Top-Right
    [110.03, 111.18],  # Center
    [51.18, 170.69],   # Bottom-Left
    [168.34, 170.66]   # Bottom-Right
], dtype=np.float32)

def get_query_centroids(img_rgb):
    """Segment the dots and return their 5 centroids sorted to match TEMPLATE_PTS."""
    # Sample background color from corners
    corners = [
        img_rgb[0:10, 0:10],
        img_rgb[0:10, -10:],
        img_rgb[-10:, 0:10],
        img_rgb[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    
    # Segment dots using color distance threshold
    dist = np.linalg.norm(img_rgb - bg_color, axis=-1)
    query_dot_mask = (dist > 70).astype(np.uint8) * 255
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(query_dot_mask)
    
    # Filter components by area > 100, sort by area descending
    dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:
            dots.append((centroids[i], area))
            
    # Take top 5 components by area
    dots.sort(key=lambda x: x[1], reverse=True)
    if len(dots) < 5:
        raise ValueError(f"Found only {len(dots)} dot components, need at least 5.")
        
    top_centroids = [d[0] for d in dots[:5]]
    
    # Reorder query centroids to match TEMPLATE_PTS ordering using nearest neighbor
    ordered_centroids = []
    for tc in TEMPLATE_PTS:
        dists = [np.linalg.norm(qc - tc) for qc in top_centroids]
        closest_idx = np.argmin(dists)
        ordered_centroids.append(top_centroids[closest_idx])
        
    return np.array(ordered_centroids, dtype=np.float32)

def compute_ncc(img1, img2, mask):
    """Computes Normalized Cross-Correlation (NCC) under a boolean mask."""
    pixels1 = img1[mask].astype(np.float32)
    pixels2 = img2[mask].astype(np.float32)
    
    ncc_channels = []
    for c in range(3):
        p1 = pixels1[:, c]
        p2 = pixels2[:, c]
        
        p1_centered = p1 - np.mean(p1)
        p2_centered = p2 - np.mean(p2)
        
        num = np.sum(p1_centered * p2_centered)
        den = np.sqrt(np.sum(p1_centered**2) * np.sum(p2_centered**2))
        
        ncc = num / den if den > 1e-6 else 0.0
        ncc_channels.append(ncc)
        
    return np.mean(ncc_channels)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    jackets_dir = "jackets"
    mask_path = os.path.join("masks", "5-dot-mask.png")
    
    # Load and prepare mask
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    mask_np = np.array(mask_img)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask_np = cv2.erode(mask_np, kernel, iterations=2) > 127
    
    # Load reference images into memory for high-speed matching
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Loading {len(image_files)} reference jackets into memory...")
    
    reference_database = []
    for filename in tqdm(image_files, desc="Caching reference jackets"):
        filepath = os.path.join(jackets_dir, filename)
        try:
            ref_img = cv2.imread(filepath)
            if ref_img is None:
                continue
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = cv2.resize(ref_img, (224, 224))
            reference_database.append((filename, ref_img))
        except Exception:
            pass
            
    print(f"Successfully cached {len(reference_database)} jackets.")
    
    # Test images 1 to 4
    test_files = ["1.png", "2.png", "3.png", "4.png"]
    
    for test_file in test_files:
        query_path = os.path.join("tests", test_file)
        if not os.path.exists(query_path):
            print(f"\n[ERROR] Test image {test_file} not found.")
            continue
            
        print(f"\n==========================================")
        print(f"Processing Test Query: {test_file}")
        print(f"==========================================")
        
        # Load query
        img_query = cv2.imread(query_path)
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        img_query = cv2.resize(img_query, (224, 224))
        
        try:
            # Get centroids and warp
            query_centroids = get_query_centroids(img_query)
            M, inliers = cv2.estimateAffinePartial2D(query_centroids, TEMPLATE_PTS)
            warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            # Save aligned result for visual debugging
            debug_out_path = os.path.join("deep_learning", f"aligned_query_{test_file}")
            cv2.imwrite(debug_out_path, cv2.cvtColor(warped_query, cv2.COLOR_RGB2BGR))
            print(f"Saved aligned query to {debug_out_path}")
            
            # Perform matching against all cached references
            matches = []
            for filename, ref_img in reference_database:
                score = compute_ncc(warped_query, ref_img, eroded_mask_np)
                matches.append((filename, score))
                
            matches.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 5 Matches for {test_file}:")
            for rank, (filename, score) in enumerate(matches[:5]):
                print(f"  {rank+1}. {filename:<50} | Score: {score:.5f}")
                
        except Exception as e:
            print(f"Failed to process {test_file}: {e}")

if __name__ == "__main__":
    main()
