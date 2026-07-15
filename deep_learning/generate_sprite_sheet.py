import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm

# Centroids in 224x224 coordinates
DOT_CENTROIDS = [
    (51, 53),   # Top-Left
    (169, 53),  # Top-Right
    (110, 111), # Center
    (51, 171),  # Bottom-Left
    (168, 171)  # Bottom-Right
]
PATCH_SIZE = 40

def make_circular_patch(patch):
    """Applies a circular mask of radius 17.5 in the center of a 40x40 patch, blacking out corners."""
    h, w = patch.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - 19.5)**2 + (Y - 19.5)**2)
    mask = dist_from_center <= 17.5
    
    masked_patch = np.zeros_like(patch)
    masked_patch[mask] = patch[mask]
    return masked_patch

def extract_patch_grid(img_224):
    """Crops 5 patches, masks them to circles, and builds a 120x80 grid."""
    patches = []
    half_size = PATCH_SIZE // 2
    for cx, cy in DOT_CENTROIDS:
        y_start = max(0, cy - half_size)
        y_end = min(224, cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(224, cx + half_size)
        
        patch = img_224[y_start:y_end, x_start:x_end]
        if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
            
        patch = make_circular_patch(patch)
        patches.append(patch)
        
    grid = np.zeros((PATCH_SIZE * 3, PATCH_SIZE * 2, 3), dtype=np.uint8)
    grid[0:40, 0:40] = patches[0]
    grid[0:40, 40:80] = patches[1]
    grid[40:80, 0:40] = patches[2]
    grid[40:80, 40:80] = patches[3]
    grid[80:120, 0:40] = patches[4]
    
    return grid

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    jackets_dir = "jackets"
    web_dir = "web"
    os.makedirs(web_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    # Sort files to ensure stable indexing
    image_files.sort()
    
    total_jackets = len(image_files)
    print(f"Generating sprite sheet for {total_jackets} reference jackets...")
    
    # 32 columns, 69 rows
    cols = 32
    rows = (total_jackets + cols - 1) // cols
    
    grid_w = 80
    grid_h = 120
    
    sprite_w = cols * grid_w
    sprite_h = rows * grid_h
    
    print(f"Sprite sheet dimensions: {sprite_w}x{sprite_h} px ({cols} cols, {rows} rows)")
    sprite_image = np.zeros((sprite_h, sprite_w, 3), dtype=np.uint8)
    
    valid_filenames = []
    
    for i, filename in enumerate(tqdm(image_files, desc="Processing jackets")):
        filepath = os.path.join(jackets_dir, filename)
        try:
            img = cv2.imread(filepath)
            if img is None:
                continue
            # Keep RGB color space consistency (OpenCV reads BGR, but we want RGB for saving)
            # Actually, cv2.imwrite expects BGR, so we can keep it as BGR!
            # Since both the reference and query will be processed in BGR/RGB consistently,
            # let's save the sprite sheet using cv2.imwrite in BGR format.
            img_resized = cv2.resize(img, (224, 224))
            grid = extract_patch_grid(img_resized)
            
            # Position in sprite sheet
            r = i // cols
            c = i % cols
            y = r * grid_h
            x = c * grid_w
            
            sprite_image[y:y+grid_h, x:x+grid_w] = grid
            valid_filenames.append(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Save the sprite sheet as WebP with lossless compression or high quality
    sprite_path = os.path.join(web_dir, "jackets_sprite.webp")
    print(f"Saving sprite sheet to {sprite_path}...")
    # cv2.IMWRITE_WEBP_QUALITY = 101 represents lossless compression in OpenCV
    cv2.imwrite(sprite_path, sprite_image, [cv2.IMWRITE_WEBP_QUALITY, 101])
    
    # Save the filename index mapping list
    index_path = os.path.join(web_dir, "jacket_index.json")
    print(f"Saving mapping index to {index_path}...")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(valid_filenames, f, indent=None)
        
    print("Database sprite sheet generation complete!")

if __name__ == "__main__":
    main()
