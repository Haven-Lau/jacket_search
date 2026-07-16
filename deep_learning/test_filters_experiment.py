import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

TEMPLATE_PTS = np.array([
    [51.17, 52.83],    # Top-Left
    [168.84, 52.84],   # Top-Right
    [110.03, 111.18],  # Center
    [51.18, 170.69],   # Bottom-Left
    [168.34, 170.66]   # Bottom-Right
], dtype=np.float32)

def get_query_centroids(img_rgb):
    corners = [img_rgb[0:10, 0:10], img_rgb[0:10, -10:], img_rgb[-10:, 0:10], img_rgb[-10:, -10:]]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    dist = np.linalg.norm(img_rgb - bg_color, axis=-1)
    query_dot_mask = (dist > 70).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(query_dot_mask)
    dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 100: dots.append((centroids[i], area))
    dots.sort(key=lambda x: x[1], reverse=True)
    if len(dots) < 5: raise ValueError("Not enough dots")
    top_centroids = [d[0] for d in dots[:5]]
    ordered_centroids = []
    for tc in TEMPLATE_PTS:
        dists = [np.linalg.norm(qc - tc) for qc in top_centroids]
        ordered_centroids.append(top_centroids[np.argmin(dists)])
    return np.array(ordered_centroids, dtype=np.float32)

def compute_ncc(img1, img2, mask):
    p1 = img1[mask].astype(np.float32)
    p2 = img2[mask].astype(np.float32)
    ncc_channels = []
    for c in range(3):
        c1 = p1[:, c]
        c2 = p2[:, c]
        c1_c = c1 - np.mean(c1)
        c2_c = c2 - np.mean(c2)
        den = np.sqrt(np.sum(c1_c**2) * np.sum(c2_c**2))
        ncc = np.sum(c1_c * c2_c) / den if den > 1e-6 else 0.0
        ncc_channels.append(ncc)
    return np.mean(ncc_channels)

def compute_mae(img1, img2, mask):
    p1 = img1[mask].astype(np.float32)
    p2 = img2[mask].astype(np.float32)
    return np.mean(np.abs(p1 - p2))

def compute_hist_sim(img1, img2, mask_cv2):
    # mask_cv2 should be uint8
    h1 = cv2.calcHist([img1], [0,1,2], mask_cv2, [8,8,8], [0,256,0,256,0,256])
    h2 = cv2.calcHist([img2], [0,1,2], mask_cv2, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    jackets_dir = "jackets"
    
    masks = {}
    kernel = np.ones((3, 3), np.uint8)
    for mask_name in ["1-dot", "3-dot", "5-dot"]:
        m_img = Image.open(f"masks/{mask_name}-mask.png").convert("L").resize((224, 224), Image.Resampling.NEAREST)
        m_np = cv2.erode(np.array(m_img), kernel, iterations=2) > 127
        masks[mask_name] = m_np
        
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Loading {len(image_files)} reference jackets...")
    
    reference_database = []
    for filename in tqdm(image_files, desc="Caching references"):
        filepath = os.path.join(jackets_dir, filename)
        ref_img = cv2.imread(filepath)
        if ref_img is None: continue
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = cv2.resize(ref_img, (224, 224))
        reference_database.append((filename, ref_img))
        
    test_files = ["1.png", "2.png", "3.png", "4.png"]
    correct_answers = {
        "1.png": "FOR_YOUR_BRAVE!!.png",
        "2.png": "YARA_TUM_KAHAN.png",
        "3.png": "ROZA_DE_ANDALUCIA.png",
        "4.png": "ReGENERATION.png"
    }
    
    queries = {}
    for tf in test_files:
        query_path = os.path.join("tests", tf)
        if not os.path.exists(query_path): continue
        img_query = cv2.resize(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB), (224, 224))
        try:
            M, _ = cv2.estimateAffinePartial2D(get_query_centroids(img_query), TEMPLATE_PTS)
            warped = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            queries[tf] = warped
        except Exception as e:
            print(f"Failed to align {tf}: {e}")
            
    for mask_name, mask_np in masks.items():
        mask_cv2 = mask_np.astype(np.uint8) * 255
        print(f"\n==========================================")
        print(f"Testing {mask_name}")
        print(f"==========================================")
        
        for test_file, warped_query in queries.items():
            correct_name = correct_answers[test_file]
            std_dev = np.std(warped_query[mask_np])
            
            matches = []
            for filename, ref_img in reference_database:
                score = compute_ncc(warped_query, ref_img, mask_np)
                mae = compute_mae(warped_query, ref_img, mask_np)
                hist_dist = compute_hist_sim(warped_query, ref_img, mask_cv2)
                matches.append((filename, score, mae, hist_dist))
                
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Find stats for correct match and top wrong match
            correct_stats = next(x for x in matches if x[0] == correct_name)
            wrong_stats = next(x for x in matches if x[0] != correct_name)
            
            print(f"\n[ {test_file} ] Query StdDev: {std_dev:.2f}")
            print(f"  CORRECT ({correct_stats[0]}):")
            print(f"    NCC: {correct_stats[1]:.4f} | MAE: {correct_stats[2]:.2f} | HistDist: {correct_stats[3]:.4f}")
            print(f"  TOP WRONG ({wrong_stats[0]}):")
            print(f"    NCC: {wrong_stats[1]:.4f} | MAE: {wrong_stats[2]:.2f} | HistDist: {wrong_stats[3]:.4f}")

if __name__ == "__main__":
    main()
