import os
from PIL import Image

def main():
    jacket_path = os.path.join("jackets", "FOR_YOUR_BRAVE!!.png")
    mask_path = os.path.join("masks", "5-dot-mask.png")
    
    if not os.path.exists(jacket_path):
        print(f"Error: {jacket_path} not found.")
        return
        
    img = Image.open(jacket_path).convert("RGB").resize((224, 224))
    mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
    
    bg_color = (34, 0, 142)  # #22008e
    bg_img = Image.new("RGB", (224, 224), bg_color)
    
    masked = bg_img.copy()
    masked.paste(img, (0, 0), mask_img)
    
    output_path = os.path.join("deep_learning", "debug_db_for_your_brave_5dot.png")
    masked.save(output_path)
    print(f"Saved database masked version to {output_path}")

if __name__ == "__main__":
    main()
