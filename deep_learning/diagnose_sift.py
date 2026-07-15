import os
import cv2

def main():
    jackets_dir = "jackets"
    image_files = [f for f in os.listdir(jackets_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Total files in directory: {len(image_files)}")
    
    success_count = 0
    fail_count = 0
    first_fails = []
    
    for filename in image_files[:100]:
        filepath = os.path.abspath(os.path.join(jackets_dir, filename))
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            success_count += 1
        else:
            fail_count += 1
            first_fails.append(filename)
            
    print(f"First 100 files: {success_count} loaded successfully, {fail_count} failed.")
    if first_fails:
        print("Sample failures:", first_fails[:5])

if __name__ == "__main__":
    main()
