import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from prepare_model import MobileNetV3SmallFeatureExtractor

def main():
    # Force stdout and stderr to use UTF-8 on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    print("Loading feature extractor model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = MobileNetV3SmallFeatureExtractor()
    model.eval()
    model.to(device)
    
    # Define ImageNet normalization pipeline (expects PyTorch tensor input)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load masks
    masks_dir = "masks"
    mask_filenames = {
        "1dot": "1-dot-mask.png",
        "3dot": "3-dot-mask.png",
        "5dot": "5-dot-mask.png"
    }
    
    masks = {}
    for name, filename in mask_filenames.items():
        mask_path = os.path.join(masks_dir, filename)
        if not os.path.exists(mask_path):
            print(f"Error: Mask file '{mask_path}' not found.")
            return
        # Load mask, convert to grayscale ('L'), and resize to 224x224 to match input size
        mask_img = Image.open(mask_path).convert("L").resize((224, 224), Image.Resampling.NEAREST)
        masks[name] = mask_img
        print(f"Loaded and prepared mask: {filename}")
        
    # Setup background image with solid color #22008e
    bg_color = (34, 0, 142)  # #22008e in RGB
    bg_img = Image.new("RGB", (224, 224), bg_color)
    
    jackets_dir = "jackets"
    if not os.path.exists(jackets_dir):
        print(f"Error: Directory '{jackets_dir}' not found.")
        return
        
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Found {len(image_files)} images in '{jackets_dir}'.")
    
    # Setup debug folder for visual verification of masked outputs
    debug_dir = os.path.join("deep_learning", "debug_masked")
    os.makedirs(debug_dir, exist_ok=True)
    saved_debug_samples = 0
    
    embeddings_db = {}
    
    # Run feature extraction with PyTorch autograd disabled
    with torch.no_grad():
        for filename in tqdm(image_files, desc="Extracting embeddings"):
            filepath = os.path.join(jackets_dir, filename)
            try:
                # 1. Load original image and convert to RGB, resize to 224x224
                img_org = Image.open(filepath).convert("RGB").resize((224, 224))
                
                # 2. Construct the variants
                variants = {
                    "full": img_org
                }
                
                for mask_name, mask_img in masks.items():
                    # Paste original image onto the #22008e background using the mask
                    masked_img = bg_img.copy()
                    masked_img.paste(img_org, (0, 0), mask_img)
                    variants[mask_name] = masked_img
                    
                # Save visual debug samples for the first 5 images
                if saved_debug_samples < 5:
                    base_name = os.path.splitext(filename)[0]
                    # Save the full image and all masked versions
                    img_org.save(os.path.join(debug_dir, f"{base_name}_full.png"))
                    for mask_name, masked_img in variants.items():
                        if mask_name != "full":
                            masked_img.save(os.path.join(debug_dir, f"{base_name}_{mask_name}.png"))
                    saved_debug_samples += 1
                
                # 3. Compute embedding for each variant
                img_embeddings = {}
                for variant_name, img_variant in variants.items():
                    # Normalize and prepare input tensor
                    tensor = preprocess(img_variant).unsqueeze(0).to(device)  # shape: (1, 3, 224, 224)
                    
                    # Forward pass
                    embedding = model(tensor).squeeze(0).cpu().numpy()  # shape: (576,)
                    
                    # Round floats to 5 decimal places to reduce JSON file size significantly
                    rounded_embedding = [round(float(val), 5) for val in embedding]
                    img_embeddings[variant_name] = rounded_embedding
                    
                # Store in database
                embeddings_db[filename] = img_embeddings
                
            except Exception as e:
                print(f"\nError processing {filename}: {e}")
                
    output_path = os.path.join("deep_learning", "embeddings.json")
    print(f"Saving {len(embeddings_db)} sets of embeddings to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embeddings_db, f, indent=None)  # Compact format without spaces/newlines
        
    print(f"Embedding generation completed successfully! Debug samples saved in: {debug_dir}")

if __name__ == "__main__":
    main()
