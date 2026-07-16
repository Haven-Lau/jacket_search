import os
import json
import numpy as np
from PIL import Image

def make_circular_patch(patch, patch_size=40, radius=17.5):
    h, w = patch.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center = (patch_size - 1) / 2.0
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = dist_from_center <= radius
    
    masked_patch = np.zeros_like(patch)
    masked_patch[mask] = patch[mask]
    return masked_patch

def build_patch_grid(img_224, patch_size=40, radius=17.5):
    DOT_CENTROIDS = [
        (51, 53),   (169, 53),  (110, 111), (51, 171),  (168, 171)
    ]
    img_np = np.array(img_224)
    patches = []
    half_size = patch_size // 2
    for cx, cy in DOT_CENTROIDS:
        y_start = max(0, cy - half_size)
        y_end = min(224, cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(224, cx + half_size)
        
        patch = img_np[y_start:y_end, x_start:x_end]
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
            
        patch = make_circular_patch(patch, patch_size, radius)
        patches.append(patch)
        
    grid = np.zeros((120, 80, 3), dtype=np.uint8)
    grid[0:40, 0:40] = patches[0]
    grid[0:40, 40:80] = patches[1]
    grid[40:80, 0:40] = patches[2]
    grid[40:80, 40:80] = patches[3]
    grid[80:120, 0:40] = patches[4]
    
    return grid

def main():
    print("Loading song metadata...")
    with open("song_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    jacket_index = list(metadata.keys())
    jacket_index.sort()
    
    print(f"Total jackets to process: {len(jacket_index)}")
    
    with open("jacket_index.json", "w", encoding="utf-8") as f:
        json.dump(jacket_index, f, ensure_ascii=False, indent=2)
    print("Saved jacket_index.json")
    
    cols = 32
    rows = (len(jacket_index) + cols - 1) // cols
    
    sprite_w = cols * 80
    sprite_h = rows * 120
    
    print(f"Sprite dimensions: {sprite_w}x{sprite_h}")
    sprite_img = np.zeros((sprite_h, sprite_w, 3), dtype=np.uint8)
    
    for i, jacket_file in enumerate(jacket_index):
        if i % 100 == 0:
            print(f"Processing {i}/{len(jacket_index)}...")
            
        path = os.path.join("jackets", jacket_file)
        if not os.path.exists(path):
            thumb_path = os.path.join("thumbnails", jacket_file.replace(".png", ".webp"))
            if os.path.exists(thumb_path):
                img = Image.open(thumb_path).convert("RGB").resize((224, 224))
            else:
                img = Image.new("RGB", (224, 224), (0,0,0))
        else:
            try:
                img = Image.open(path).convert("RGB").resize((224, 224))
            except Exception as e:
                img = Image.new("RGB", (224, 224), (0,0,0))
                
        grid = build_patch_grid(img)
        row = i // cols
        col = i % cols
        y = row * 120
        x = col * 80
        sprite_img[y:y+120, x:x+80] = grid

    print("Saving jackets_sprite.webp...")
    Image.fromarray(sprite_img).save("jackets_sprite.webp", "WEBP", quality=80)
    print("Done!")

if __name__ == "__main__":
    main()
