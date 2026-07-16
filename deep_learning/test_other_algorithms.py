import os
import sys
import cv2
import numpy as np
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

def compute_dhash(img_gray):
    # Resize to 9x8
    resized = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)
    # Compare adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # Convert to 64-bit integer
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

def compute_orb_matches(orb, bf, query_desc, ref_desc):
    if query_desc is None or ref_desc is None:
        return 0
    # Match descriptors
    matches = bf.match(query_desc, ref_desc)
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # Take top matches and calculate a score (e.g., number of matches with distance < 50)
    good_matches = [m for m in matches if m.distance < 50]
    return len(good_matches)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    jackets_dir = "jackets"
    
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Loading {len(image_files)} reference jackets and computing hashes/features...")
    
    reference_database = []
    for filename in tqdm(image_files, desc="Caching references"):
        filepath = os.path.join(jackets_dir, filename)
        ref_img = cv2.imread(filepath)
        if ref_img is None: continue
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.resize(ref_gray, (224, 224))
        
        # dHash
        dhash = compute_dhash(ref_gray)
        
        # ORB
        kp, desc = orb.detectAndCompute(ref_gray, None)
        
        reference_database.append((filename, dhash, desc))
        
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
        img_query = cv2.imread(query_path)
        img_query_rgb = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        img_query_rgb = cv2.resize(img_query_rgb, (224, 224))
        try:
            M, _ = cv2.estimateAffinePartial2D(get_query_centroids(img_query_rgb), TEMPLATE_PTS)
            warped = cv2.warpAffine(img_query_rgb, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            queries[tf] = warped_gray
        except Exception as e:
            print(f"Failed to align {tf}: {e}")
            
    print(f"\n==========================================")
    print(f"Testing dHash (Lower distance is better)")
    print(f"==========================================")
    for test_file, warped_gray in queries.items():
        q_hash = compute_dhash(warped_gray)
        matches = []
        for filename, r_hash, r_desc in reference_database:
            dist = hamming_distance(q_hash, r_hash)
            matches.append((filename, dist))
        matches.sort(key=lambda x: x[1]) # lower is better
        correct_name = correct_answers[test_file]
        correct_stats = next(x for x in matches if x[0] == correct_name)
        wrong_stats = next(x for x in matches if x[0] != correct_name)
        print(f"[{test_file}] CORRECT: {correct_stats[1]} dist | TOP WRONG: {wrong_stats[0]} ({wrong_stats[1]} dist)")

    print(f"\n==========================================")
    print(f"Testing ORB Feature Matching (Higher matches is better)")
    print(f"==========================================")
    for test_file, warped_gray in queries.items():
        kp, q_desc = orb.detectAndCompute(warped_gray, None)
        matches = []
        for filename, r_hash, r_desc in reference_database:
            score = compute_orb_matches(orb, bf, q_desc, r_desc)
            matches.append((filename, score))
        matches.sort(key=lambda x: x[1], reverse=True) # higher is better
        correct_name = correct_answers[test_file]
        correct_stats = next(x for x in matches if x[0] == correct_name)
        wrong_stats = next(x for x in matches if x[0] != correct_name)
        print(f"[{test_file}] CORRECT: {correct_stats[1]} matches | TOP WRONG: {wrong_stats[0]} ({wrong_stats[1]} matches)")

if __name__ == "__main__":
    main()
