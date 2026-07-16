import os
import sys
import time
import urllib.parse
import threading
import concurrent.futures
import requests

URLS_FILE = "missing_jacket_links.txt"
OUTPUT_DIR = "jackets"
ERROR_LOG = "download_errors.log"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_WORKERS = 5

def sanitize_filename(filename: str) -> str:
    """Decodes URL-encoded filenames and replaces any invalid Windows filesystem characters."""
    decoded = urllib.parse.unquote(filename)
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        decoded = decoded.replace(char, '_')
    return decoded

def download_worker(url: str, stop_event: threading.Event, error_lock: threading.Lock, log_file_path: str) -> str:
    """Worker function to download a single image."""
    if stop_event.is_set():
        return None
        
    raw_filename = url.split('/')[-1]
    filename = sanitize_filename(raw_filename)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        headers = {"User-Agent": "RemyWikiJacketExtractor/1.0 (Contact: user@example.com)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        with open(filepath, "wb") as img_file:
            img_file.write(response.content)
            
        return filename
        
    except Exception as e:
        # Signal other threads to stop initiating new requests
        stop_event.set()
        
        error_msg = f"Failed to download {filename} from {url}: {e}"
        with error_lock:
            # Print error output cleanly without overlapping
            sys.stderr.write(f"\n[ERROR] {error_msg}\n")
            try:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            except Exception as write_err:
                sys.stderr.write(f"Failed to write to error log: {write_err}\n")
                
        raise e

def download_jackets(batch_size: int, max_workers: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(URLS_FILE):
        print(f"Error: {URLS_FILE} not found. Please run extract_jacket_urls.py first.")
        sys.exit(1)
        
    with open(URLS_FILE, "r", encoding="utf-8") as f:
        all_urls = [line.strip() for line in f if line.strip()]
        
    total_urls = len(all_urls)
    
    # Pre-filter pending URLs
    pending_urls = []
    skipped_count = 0
    
    for url in all_urls:
        raw_filename = url.split('/')[-1]
        filename = sanitize_filename(raw_filename)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            skipped_count += 1
        else:
            pending_urls.append(url)
            
    # Apply batch size limit to pending URLs
    if batch_size > 0:
        targets = pending_urls[:batch_size]
    else:
        targets = pending_urls
        
    to_download_count = len(targets)
    
    print(f"Total URLs in file:      {total_urls}")
    print(f"Already Downloaded:      {skipped_count}")
    print(f"Remaining to Download:   {len(pending_urls)}")
    print(f"Batch Size for this run: {batch_size if batch_size > 0 else 'Unlimited'}")
    print(f"Parallel Workers (Threads): {max_workers}")
    print(f"Output Directory:        {OUTPUT_DIR}/")
    print("-" * 50)
    
    if to_download_count == 0:
        print("All matching files are already downloaded. Nothing to do!")
        return
        
    stop_event = threading.Event()
    error_lock = threading.Lock()
    
    downloaded_count = 0
    failed = False
    
    print(f"Launching parallel downloads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(download_worker, url, stop_event, error_lock, ERROR_LOG): url 
            for url in targets
        }
        
        try:
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    filename = future.result()
                    if filename:
                        downloaded_count += 1
                        print(f"[{downloaded_count}/{to_download_count}] Downloaded: {filename}")
                except Exception:
                    # Thread raised exception; stop_event has been set by the worker
                    failed = True
                    # Attempt to cancel any non-started tasks
                    for f in future_to_url:
                        f.cancel()
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Cancelling pending downloads...")
            stop_event.set()
            for f in future_to_url:
                f.cancel()
            failed = True
            
    print("-" * 50)
    print("Run Summary:")
    print(f"  Successfully Downloaded: {downloaded_count}")
    print(f"  Skipped (Already Done):  {skipped_count}")
    print(f"  Total Local Images:      {skipped_count + downloaded_count}")
    print("-" * 50)
    
    if failed:
        print("Script halted early due to an error or manual cancellation.")
        print(f"Check {ERROR_LOG} for error details if a network issue occurred.")
        sys.exit(1)

if __name__ == "__main__":
    batch_size = DEFAULT_BATCH_SIZE
    max_workers = DEFAULT_MAX_WORKERS
    
    # Parse CLI arguments:
    # Usage: python download_jackets.py [batch_size] [max_workers]
    if len(sys.argv) > 1:
        try:
            batch_size = int(sys.argv[1])
        except ValueError:
            print(f"Invalid batch size parameter. Using default: {DEFAULT_BATCH_SIZE}")
            
    if len(sys.argv) > 2:
        try:
            max_workers = int(sys.argv[2])
        except ValueError:
            print(f"Invalid workers parameter. Using default: {DEFAULT_MAX_WORKERS}")
            
    download_jackets(batch_size, max_workers)
