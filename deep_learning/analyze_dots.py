import os
import cv2
import numpy as np
from PIL import Image

def get_dot_properties(binary_mask):
    # Convert to uint8 for cv2
    mask_uint8 = (binary_mask.astype(np.uint8)) * 255
    # Find connected components (centroids and stats)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
    
    # Filter out background (label 0)
    dots = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Ignore very small noise components
        if area > 100:
            dots.append({
                "centroid": centroids[i],
                "area": area,
                "bbox": (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], 
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
            
    # Sort dots by position: top-to-bottom, then left-to-right to order them consistently
    # For a 5-dot dice:
    # 0: Top-Left, 1: Top-Right, 2: Center, 3: Bottom-Left, 4: Bottom-Right
    # We can sort them by distance to corners/center or simple coordinate sorting.
    dots.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    return dots

def main():
    query_path = os.path.join("tests", "1.png")
    mask_path = os.path.join("masks", "5-dot-mask.png")
    
    # Load and resize
    img_query = Image.open(query_path).convert("RGB").resize((224, 224))
    query_np = np.array(img_query)
    
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    mask_np = np.array(mask_img) > 127
    
    # Segment query dots
    corners = [
        query_np[0:10, 0:10],
        query_np[0:10, -10:],
        query_np[-10:, 0:10],
        query_np[-10:, -10:]
    ]
    bg_color = np.mean([np.mean(c, axis=(0, 1)) for c in corners], axis=0)
    dist = np.linalg.norm(query_np - bg_color, axis=-1)
    query_dot_mask = dist > 70
    
    # Get properties
    template_dots = get_dot_properties(mask_np)
    query_dots = get_dot_properties(query_dot_mask)
    
    print(f"Template dots found: {len(template_dots)}")
    for i, dot in enumerate(template_dots):
        print(f"  Dot {i}: Centroid={dot['centroid']}, Area={dot['area']}, BBox={dot['bbox']}")
        
    print(f"\nQuery dots found: {len(query_dots)}")
    for i, dot in enumerate(query_dots):
        print(f"  Dot {i}: Centroid={dot['centroid']}, Area={dot['area']}, BBox={dot['bbox']}")
        
    if len(template_dots) == 5 and len(query_dots) == 5:
        print("\nDistance comparison (Query vs Template):")
        for i in range(5):
            tc = template_dots[i]["centroid"]
            qc = query_dots[i]["centroid"]
            diff = qc - tc
            print(f"  Dot {i}: Shift=(dx={diff[0]:.2f}, dy={diff[1]:.2f}) | Area ratio (Query/Template)={query_dots[i]['area']/template_dots[i]['area']:.2f}")

if __name__ == "__main__":
    main()
