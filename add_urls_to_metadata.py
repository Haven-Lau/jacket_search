import json
import urllib.parse

def main():
    # Load urls
    urls = {}
    with open("jacket_urls.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            # Extract filename from URL
            filename = urllib.parse.unquote(line.split("/")[-1])
            urls[filename] = line
            
    # Load metadata
    with open("song_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    for filename, info in metadata.items():
        if filename in urls:
            info["url"] = urls[filename]
        else:
            # Try replacing space with underscore
            alt = filename.replace(" ", "_")
            if alt in urls:
                info["url"] = urls[alt]
            else:
                info["url"] = f"https://remywiki.com/images/default.png"
                
    with open("song_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Updated song_metadata.json with {len(urls)} URLs")

if __name__ == "__main__":
    main()
