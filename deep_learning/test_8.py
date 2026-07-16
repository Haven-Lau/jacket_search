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
        if area > 50: dots.append((centroids[i], area))
    dots.sort(key=lambda x: x[1], reverse=True)
    
    # Just take however many dots we have up to 5, to avoid errors on 1-dot or 3-dot tests
    num_to_use = min(len(dots), 5)
    if num_to_use == 0:
        return None, None
        
    top_centroids = [d[0] for d in dots[:num_to_use]]
    ordered_centroids = []
    ordered_template = []
    
    # Match each found dot to the closest template dot
    for qc in top_centroids:
        dists = [np.linalg.norm(qc - tc) for tc in TEMPLATE_PTS]
        best_idx = np.argmin(dists)
        ordered_centroids.append(qc)
        ordered_template.append(TEMPLATE_PTS[best_idx])
        
    return np.array(ordered_centroids, dtype=np.float32), np.array(ordered_template, dtype=np.float32)

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

def compute_color_distance(img1, img2, mask):
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
    
    p1 = hsv1[mask].astype(np.float32)
    p2 = hsv2[mask].astype(np.float32)
    
    hue_diff = np.abs(p1[:, 0] - p2[:, 0])
    hue_dist = np.minimum(hue_diff, 180 - hue_diff) / 90.0
    sat_dist = np.abs(p1[:, 1] - p2[:, 1]) / 255.0
    
    return np.mean(hue_dist + sat_dist)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
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
        
    query_path = os.path.join("tests", "8.png")
    if not os.path.exists(query_path): 
        print(f"Missing 8.png")
        return
        
    img_query = cv2.resize(cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB), (224, 224))
    
    # Try to align
    q_pts, t_pts = get_query_centroids(img_query)
    
    if q_pts is None or len(q_pts) < 2:
        print("Not enough dots found to auto-align. Skipping warping and running raw center-crop.")
        warped = img_query
    else:
        M, _ = cv2.estimateAffinePartial2D(q_pts, t_pts)
        if M is None:
            print("Affine estimation failed. Using raw image.")
            warped = img_query
        else:
            warped = cv2.warpAffine(img_query, M, (224, 224), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
    for mask_name in ["1-dot", "3-dot", "5-dot"]:
        mask_np = masks[mask_name]
        
        matches = []
        for filename, ref_img in reference_database:
            ncc_score = compute_ncc(warped, ref_img, mask_np)
            color_dist = compute_color_distance(warped, ref_img, mask_np)
            penalized = ncc_score - (color_dist * 0.2)
            matches.append((filename, penalized, ncc_score, color_dist))
            
        matches.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n==========================================")
        print(f"Test File: 8.png ({mask_name})")
        print(f"==========================================")
        for i in range(10):
            print(f"  {i+1}. {matches[i][0]:<40} | Final: {matches[i][1]:.4f} | NCC: {matches[i][2]:.4f} | HSV Dist: {matches[i][3]:.4f}")

if __name__ == "__main__":
    main()
