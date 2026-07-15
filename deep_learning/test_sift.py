import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    query_path = os.path.join("tests", "1.png")
    jackets_dir = "jackets"
    correct_song = "FOR_YOUR_BRAVE!!.png"
    
    # 1. Load query image
    img_query = cv2.imread(query_path)
    if img_query is None:
        print(f"Error: Could not load query image {query_path}")
        return
    gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    
    # Segment query dots to find background mask (we only want keypoints inside the dots!)
    # We sample the corner background color
    corners = [
        img_query[0:10, 0:10],
        img_query[0:10, -10:],
        img_query[-10:, 0:10],
        img_query[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    dist = np.linalg.norm(img_query - bg_color, axis=-1)
    query_dot_mask = (dist > 70).astype(np.uint8) * 255
    
    # 2. Detect SIFT keypoints and descriptors in the query image inside the dots mask
    sift = cv2.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(gray_query, mask=query_dot_mask)
    print(f"Detected {len(kp_query)} SIFT keypoints in query image (inside dots).")
    
    if len(kp_query) == 0:
        print("Error: No keypoints found in query image.")
        return
        
    # Matcher settings
    # For SIFT, use FLANN or BFMatcher with L2 distance
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    # 3. Match against reference jackets
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Matching query keypoints against {len(image_files)} jackets using SIFT + RANSAC...")
    
    results = []
    
    # For speed, let's limit reference image processing to grayscale
    for filename in tqdm(image_files, desc="Matching jackets"):
        filepath = os.path.join(jackets_dir, filename)
        try:
            ref_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                continue
                
            # Resize reference to 224x224 to keep resolution consistent
            ref_img = cv2.resize(ref_img, (224, 224))
            
            # Detect keypoints in full reference image
            kp_ref, des_ref = sift.detectAndCompute(ref_img, mask=None)
            if kp_ref is None or len(kp_ref) < 4:
                results.append((filename, 0))
                continue
                
            # Match descriptors using KNN (k=2) for Lowe's ratio test
            matches = bf.knnMatch(des_query, des_ref, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
            # Filter matches using RANSAC to find inliers
            inlier_count = 0
            if len(good_matches) >= 4:
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find Homography using RANSAC
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inlier_count = int(np.sum(mask))
                    
            results.append((filename, inlier_count))
        except Exception as e:
            print(f"Error matching {filename}: {e}")
            sys.exit(1)  # Crash immediately if there's a logic error so we don't spin silently
            
    # Sort results by inlier count descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Print top 10 matches
    print("\n=== Top 10 Matches using SIFT + RANSAC ===")
    correct_rank = -1
    for i, (filename, inliers) in enumerate(results[:10]):
        status = "CORRECT" if filename == correct_song else "WRONG"
        if filename == correct_song:
            correct_rank = i + 1
        print(f"{i+1}. {filename:<50} | Inliers: {inliers:<4} | [{status}]")
        
    if correct_rank == -1:
        for rank, (filename, inliers) in enumerate(results):
            if filename == correct_song:
                print(f"...\n{rank+1}. {filename:<50} | Inliers: {inliers:<4} | [CORRECT]")
                break

if __name__ == "__main__":
    main()
