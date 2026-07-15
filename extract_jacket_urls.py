import requests
import json
import time

def extract_jacket_urls():
    api_url = "https://remywiki.com/api.php"
    
    # Base query parameters for MediaWiki API
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": "Category:GuitarFreaks_&_DrumMania_Jackets",
        "gcmlimit": 500,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    
    urls = []
    page_count = 1
    
    print("Starting extraction of jacket image URLs from RemyWiki...")
    
    while True:
        print(f"Fetching batch {page_count}...")
        try:
            response = requests.get(api_url, params=params, headers={"User-Agent": "RemyWikiJacketExtractor/1.0 (Contact: user@example.com)"})
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching batch {page_count}: {e}")
            break
        
        # Check if query and pages exist in the response
        query_data = data.get("query", {})
        pages = query_data.get("pages", {})
        
        batch_urls_count = 0
        for page_id, page_info in pages.items():
            image_info_list = page_info.get("imageinfo", [])
            if image_info_list:
                img_url = image_info_list[0].get("url")
                if img_url:
                    urls.append(img_url)
                    batch_urls_count += 1
        
        print(f"Batch {page_count} processed. Found {batch_urls_count} image URLs in this batch.")
        
        # Check for continuation tokens
        continue_info = data.get("continue")
        if continue_info:
            # Update params with continuation data (merges continue and gcmcontinue parameters)
            params.update(continue_info)
            page_count += 1
            # Add a small delay between requests to be polite to the server
            time.sleep(0.5)
        else:
            print("No more pages to fetch. Extraction complete.")
            break
            
    # Write the collected URLs to jacket_urls.txt
    output_filename = "jacket_urls.txt"
    try:
        # Sort URLs for consistent output order
        urls.sort()
        with open(output_filename, "w", encoding="utf-8") as f:
            for url in urls:
                f.write(url + "\n")
        print(f"\nSuccessfully wrote {len(urls)} URLs to {output_filename}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    start_time = time.time()
    extract_jacket_urls()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
