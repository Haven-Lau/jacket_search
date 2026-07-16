import os
import sys
import cv2
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
    corners = [
        img_rgb[0:10, 0:10],
        img_rgb[0:10, -10:],
        img_rgb[-10:, 0:10],
        img_rgb[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    
    dist = np.linalg.norm(img_rgb - bg_color, axis=-1)
    query_dot_mask = (dist > 70).astype(np.uint8) * 255
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(query_dot_mask)
    
    dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100:
            dots.append((centroids[i], area))
            
    dots.sort(key=lambda x: x[1], reverse=True)
    if len(dots) < 5:
        raise ValueError(f"Found only {len(dots)} dot components, need at least 5.")
        
    top_centroids = [d[0] for d in dots[:5]]
    
    ordered_centroids = []
    for tc in TEMPLATE_PTS:
        dists = [np.linalg.norm(qc - tc) for qc in top_centroids]
        closest_idx = np.argmin(dists)
        ordered_centroids.append(top_centroids[closest_idx])
        
    return np.array(ordered_centroids, dtype=np.float32)

def compute_ncc(img1, img2, mask):
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

def apply_contrast(img, alpha):
    # contrast mapping around center (127.5)
    # bounded to 0-255
    img_float = img.astype(np.float32)
    img_float = 127.5 + alpha * (img_float - 127.5)
    return np.clip(img_float, 0, 255).astype(np.uint8)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    jackets_dir = "jackets"
    mask_path = os.path.join("masks", "5-dot-mask.png")
    
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    mask_np = np.array(mask_img)
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask_np = cv2.erode(mask_np, kernel, iterations=2) > 127
    
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Loading {len(image_files)} reference jackets into memory...")
    
    reference_database = []
    for filename in tqdm(image_files, desc="Caching reference jackets"):
        filepath = os.path.join(jackets_dir, filename)
        try:
            ref_img = cv2.imread(filepath)
            if ref_img is None: continue
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = cv2.resize(ref_img, (224, 224))
            reference_database.append((filename, ref_img))
        except Exception:
            pass
            
    print(f"Successfully cached {len(reference_database)} jackets.")
    
    test_files = ["1.png", "2.png", "3.png", "4.png"]
    alphas = [0.5, 0.75, 1.0, 1.5, 2.0]
    
    results = {alpha: [] for alpha in alphas}
    
    # Store warped queries first to keep dot alignment invariant to contrast (which could mess up connected components)
    queries = {}
    for test_file in test_files:
        query_path = os.path.join("tests", test_file)
        if not os.path.exists(query_path):
            continue
        img_query = cv2.imread(query_path)
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        img_query = cv2.resize(img_query, (224, 224))
        try:
            query_centroids = get_query_centroids(img_query)
            M, inliers = cv2.estimateAffinePartial2D(query_centroids, TEMPLATE_PTS)
            warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            queries[test_file] = warped_query
        except Exception as e:
            print(f"Failed to align {test_file}: {e}")
            continue

    for alpha in alphas:
        print(f"\n==========================================")
        print(f"Testing Contrast Alpha = {alpha}")
        print(f"==========================================")
        
        # Apply contrast to reference DB
        alpha_db = [(name, apply_contrast(img, alpha)) for name, img in reference_database]
        
        for test_file, warped_query in queries.items():
            alpha_query = apply_contrast(warped_query, alpha)
            
            matches = []
            for filename, ref_img in alpha_db:
                score = compute_ncc(alpha_query, ref_img, eroded_mask_np)
                matches.append((filename, score))
                
            matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = matches[:3]
            
            # Record result
            results[alpha].append((test_file, top_matches))
            print(f"{test_file} | Rank 1: {top_matches[0][0]} ({top_matches[0][1]:.5f}) | Rank 2: {top_matches[1][0]} ({top_matches[1][1]:.5f}) | Margin: {top_matches[0][1] - top_matches[1][1]:.5f}")
            
if __name__ == "__main__":
    main()
