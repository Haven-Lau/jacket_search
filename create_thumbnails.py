import os
from PIL import Image

def create_thumbnails(source_dir='jackets', target_dir='thumbnails', size=(128, 128), quality=80):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    print(f"Found {len(files)} images to process.")

    saved_size_total = 0
    original_size_total = 0
    processed_count = 0

    for i, file_name in enumerate(files):
        source_path = os.path.join(source_dir, file_name)
        base_name, _ = os.path.splitext(file_name)
        target_name = f"{base_name}.webp"
        target_path = os.path.join(target_dir, target_name)

        try:
            original_size_total += os.path.getsize(source_path)
            
            with Image.open(source_path) as img:
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGBA')
                
                img_resized = img.resize(size, Image.Resampling.LANCZOS)
                img_resized.save(target_path, 'WEBP', quality=quality, method=6)
                
            saved_size_total += os.path.getsize(target_path)
            processed_count += 1
            if processed_count % 100 == 0 or processed_count == len(files):
                print(f"Processed {processed_count}/{len(files)} images...")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    orig_mb = original_size_total / (1024 * 1024) if original_size_total > 0 else 0
    new_mb = saved_size_total / (1024 * 1024) if saved_size_total > 0 else 0
    print(f"\nDone! Processed {processed_count} images successfully.")
    print(f"Original size: {orig_mb:.2f} MB")
    print(f"Thumbnail size: {new_mb:.2f} MB")
    if orig_mb > 0:
        print(f"Reduction: {(1 - new_mb/orig_mb)*100:.1f}%")

if __name__ == '__main__':
    create_thumbnails()
