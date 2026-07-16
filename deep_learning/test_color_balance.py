import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_mask_centroids(mask_np):
    """Finds centroids of dots in a binary template mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np.astype(np.uint8) * 255)
    dots = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            dots.append(centroids[i])
    # Sort top-to-bottom, left-to-right
    dots.sort(key=lambda d: (d[1], d[0]))
    return np.array(dots, dtype=np.float32)

def get_query_centroids_for_mask(img_rgb, template_centroids):
    """Finds centroids in the query image and matches them to the template centroids."""
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
    
    query_centroids = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            query_centroids.append(centroids[i])
            
    if len(query_centroids) == 0:
        raise ValueError("No dots detected in query image.")
        
    # Match query centroids to template centroids using nearest neighbors
    matched_query_centroids = []
    for tc in template_centroids:
        dists = [np.linalg.norm(qc - tc) for qc in query_centroids]
        closest_idx = np.argmin(dists)
        matched_query_centroids.append(query_centroids[closest_idx])
        
    return np.array(matched_query_centroids, dtype=np.float32)

def apply_color_balance(pixels):
    """
    Applies Gray World assumption.
    pixels: (N, 3) float32 array
    """
    mean_color = np.mean(pixels, axis=0) # [R_mean, G_mean, B_mean]
    avg_gray = np.mean(mean_color)
    
    scale = avg_gray / (mean_color + 1e-6)
    balanced_pixels = pixels * scale
    return np.clip(balanced_pixels, 0, 255)

def compute_ncc(img1, img2, mask, apply_wb=False):
    """Computes Normalized Cross-Correlation (NCC) under a boolean mask."""
    pixels1 = img1[mask].astype(np.float32)
    pixels2 = img2[mask].astype(np.float32)
    
    if apply_wb:
        pixels1 = apply_color_balance(pixels1)
    
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
    test_files = [
        ("1.png", "FOR_YOUR_BRAVE!!.png"),
        ("2.png", "YARA_TUM_KAHAN.png"),
        ("3.png", "ROZA_DE_ANDALUCIA.png"),
        ("4.png", "ReGENERATION.png")
    ]
    
    masks_config = {
        "5-dot": "5-dot-mask.png",
        "3-dot": "3-dot-mask.png",
        "1-dot": "1-dot-mask.png"
    }
    
    # 1. Cache reference jackets
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Caching {len(image_files)} reference jackets...")
    
    reference_database = []
    for filename in tqdm(image_files, desc="Caching jackets"):
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
            
    # 2. Iterate over masks and queries
    for mask_name, mask_filename in masks_config.items():
        print(f"\n==========================================")
        print(f"TESTING WITH CONFIG: {mask_name.upper()} MASK")
        print(f"==========================================")
        
        # Load mask and template centroids
        mask_path = os.path.join("masks", mask_filename)
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_filename} not found.")
            continue
            
        mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
        mask_np = np.array(mask_img)
        
        # Eroded mask for NCC matching
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask_np = cv2.erode(mask_np, kernel, iterations=2) > 127
        
        template_centroids = get_mask_centroids(mask_np > 127)
        print(f"Template centroids for {mask_name}: {template_centroids.tolist()}")
        
        # Run matching for each query
        for test_file, correct_song in test_files:
            query_path = os.path.join("tests", test_file)
            if not os.path.exists(query_path):
                continue
                
            img_query = cv2.imread(query_path)
            img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
            img_query = cv2.resize(img_query, (224, 224))
            
            try:
                # Align query to template centroids
                query_centroids = get_query_centroids_for_mask(img_query, template_centroids)
                
                if len(template_centroids) >= 2:
                    # Affine alignment for 3 or 5 points
                    M, inliers = cv2.estimateAffinePartial2D(query_centroids, template_centroids)
                    warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                else:
                    # Simple translation for 1 point
                    tc = template_centroids[0]
                    qc = query_centroids[0]
                    dx = tc[0] - qc[0]
                    dy = tc[1] - qc[1]
                    
                    # Warp using translation matrix
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    warped_query = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                
                # Perform matching WITHOUT White Balance
                matches = []
                for filename, ref_img in reference_database:
                    score = compute_ncc(warped_query, ref_img, eroded_mask_np, apply_wb=False)
                    matches.append((filename, score))
                matches.sort(key=lambda x: x[1], reverse=True)
                
                correct_rank_no_wb = -1
                correct_score_no_wb = -1
                for rank, (filename, score) in enumerate(matches):
                    if filename == correct_song:
                        correct_rank_no_wb = rank + 1
                        correct_score_no_wb = score
                        break

                # Perform matching WITH White Balance
                matches_wb = []
                for filename, ref_img in reference_database:
                    score = compute_ncc(warped_query, ref_img, eroded_mask_np, apply_wb=True)
                    matches_wb.append((filename, score))
                matches_wb.sort(key=lambda x: x[1], reverse=True)
                
                correct_rank_wb = -1
                correct_score_wb = -1
                for rank, (filename, score) in enumerate(matches_wb):
                    if filename == correct_song:
                        correct_rank_wb = rank + 1
                        correct_score_wb = score
                        break
                        
                print(f"Query: {test_file:<7} | Match: {correct_song:<25}")
                print(f"  -> NO WB : Rank {correct_rank_no_wb:<4} | Score: {correct_score_no_wb:.5f}")
                print(f"  -> WITH WB: Rank {correct_rank_wb:<4} | Score: {correct_score_wb:.5f}")
                
            except Exception as e:
                print(f"Query: {test_file:<7} | Failed to process: {e}")

if __name__ == "__main__":
    main()
