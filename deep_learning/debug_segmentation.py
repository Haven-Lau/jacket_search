import os
import numpy as np
from PIL import Image

def main():
    query_path = os.path.join("tests", "1.png")
    img = Image.open(query_path).convert("RGB").resize((224, 224))
    query_np = np.array(img)
    
    # Calculate background color from corners
    corners = [
        query_np[0:10, 0:10],
        query_np[0:10, -10:],
        query_np[-10:, 0:10],
        query_np[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    print("Background color:", bg_color)
    
    # Segment dots using distance threshold
    dist = np.linalg.norm(query_np - bg_color, axis=-1)
    
    # Save distance map and mask
    dist_normalized = (dist / dist.max() * 255).astype(np.uint8)
    Image.fromarray(dist_normalized).save("deep_learning/debug_query_dist.png")
    
    for threshold in [40, 70, 100, 130]:
        mask = (dist > threshold).astype(np.uint8) * 255
        Image.fromarray(mask).save(f"deep_learning/debug_query_mask_thresh_{threshold}.png")
        print(f"Saved threshold {threshold} mask.")

if __name__ == "__main__":
    main()
