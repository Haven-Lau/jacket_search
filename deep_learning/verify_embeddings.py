import os
import json
import math

def check_l2_norm(vec):
    sq_sum = sum(x * x for x in vec)
    norm = math.sqrt(sq_sum)
    return abs(norm - 1.0) < 1e-3  # Allow slightly higher tolerance due to 5-decimal rounding

def main():
    print("=== Verification Script (Masked Embeddings) ===")
    
    onnx_path = os.path.join("deep_learning", "model.onnx")
    db_path = os.path.join("deep_learning", "embeddings.json")
    jackets_dir = "jackets"
    
    # 1. Verify ONNX Model
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[OK] ONNX model exists: {onnx_path}")
        print(f"     ONNX model size: {size_mb:.2f} MB")
    else:
        print(f"[FAIL] ONNX model not found at: {onnx_path}")
        return
        
    # 2. Verify Embeddings Database
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"[OK] Embeddings database exists: {db_path}")
        print(f"     Database file size: {size_mb:.2f} MB")
    else:
        print(f"[FAIL] Embeddings database not found at: {db_path}")
        return
        
    # Load and validate embeddings
    print("Loading and parsing database...")
    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)
        
    print(f"Loaded {len(db)} items from the database.")
    
    # Check directory contents
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Images in jackets directory: {len(image_files)}")
    
    # 3. Check for mismatches
    missing_in_db = [f for f in image_files if f not in db]
    if missing_in_db:
        print(f"[WARNING] {len(missing_in_db)} images in the directory are missing from the database.")
        if len(missing_in_db) <= 5:
            print("Missing files:", missing_in_db)
    else:
        print("[OK] All local images have precomputed entries in the database.")
        
    # 4. Check embedding properties
    valid_keys = True
    valid_dim = True
    valid_norm = True
    expected_dim = 576
    required_variants = {"full", "1dot", "3dot", "5dot"}
    
    for filename, variants in db.items():
        # Check keys
        actual_variants = set(variants.keys())
        if actual_variants != required_variants:
            print(f"[FAIL] Keys mismatch for {filename}: got {actual_variants}, expected {required_variants}")
            valid_keys = False
            break
            
        # Check dimensions and norms for each variant
        for name, embedding in variants.items():
            if len(embedding) != expected_dim:
                print(f"[FAIL] Dimension mismatch for {filename} ({name}): got {len(embedding)}, expected {expected_dim}")
                valid_dim = False
                break
            if not check_l2_norm(embedding):
                norm_val = math.sqrt(sum(x*x for x in embedding))
                print(f"[FAIL] Embedding for {filename} ({name}) is not L2-normalized: norm={norm_val}")
                valid_norm = False
                break
        if not (valid_dim and valid_norm):
            break
            
    if valid_keys:
        print(f"[OK] All database entries contain exactly the required variants: {required_variants}")
    if valid_dim:
        print(f"[OK] All embeddings have consistent dimension: {expected_dim}")
    if valid_norm:
        print("[OK] All embeddings are properly L2-normalized (with tolerance for rounding).")
        
    # Check visual debug folder
    debug_dir = os.path.join("deep_learning", "debug_masked")
    if os.path.exists(debug_dir):
        debug_files = os.listdir(debug_dir)
        print(f"[OK] Debug visual samples generated: {len(debug_files)} files in '{debug_dir}'")
    else:
        print("[WARNING] Debug visual samples directory not found.")
        
    if len(db) == len(image_files) and valid_keys and valid_dim and valid_norm:
        print("\nVerification PASSED! The database is successfully prepared for all quiz formats.")
    else:
        print("\nVerification FAILED. Please review the issues noted above.")

if __name__ == "__main__":
    main()
