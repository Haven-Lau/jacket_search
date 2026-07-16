import os
import json
import urllib.request
import urllib.parse
import re
from bs4 import BeautifulSoup
import time

def extract_songs_from_html():
    with open('songlist.html', 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    songs = {}
    for li in soup.find_all('li'):
        a_tag = li.find('a')
        if a_tag and 'href' in a_tag.attrs:
            href = a_tag['href']
            if '#' in href:
                continue
            if href.startswith('https://remywiki.com/'):
                url_part = urllib.parse.unquote(href.split('/')[-1])
                song_name = a_tag.get_text().strip()
                
                li_text = li.get_text()
                if ' / ' in li_text:
                    artist_name = li_text.split(' / ', 1)[1].strip()
                else:
                    artist_name = 'Unknown Artist'
                
                songs[url_part] = {
                    'title': song_name,
                    'artist': artist_name
                }
    return songs

def fetch_jackets_from_wiki(song_keys, start_offset=0):
    base_url = 'https://remywiki.com/api.php?action=query&prop=revisions&rvprop=content&format=json&titles='
    
    jacket_map = {}
    
    # Process in batches of 50
    batch_size = 50
    keys_list = list(song_keys)
    
    for i in range(start_offset, len(keys_list), batch_size):
        batch = keys_list[i:i+batch_size]
        titles = '|'.join(batch)
        url = base_url + urllib.parse.quote(titles)
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode('utf-8'))
                pages = data.get('query', {}).get('pages', {})
                
                for page_id, page_info in pages.items():
                    if 'revisions' not in page_info:
                        continue
                        
                    title = page_info['title'].replace(' ', '_')
                    content = page_info['revisions'][0]['*']
                    
                    # Extract all images
                    images = []
                    lines = content.split('\n')
                    for line in lines:
                        match = re.search(r'(?:\[\[(?:Image|File):|\|\s*jacket\s*=\s*)([^|\]\n]+\.(?:png|jpg|jpeg|gif|webp))', line, re.IGNORECASE)
                        if match:
                            img = match.group(1).strip()
                            images.append((img, line))
                            
                    best_img = None
                    if len(images) == 1:
                        best_img = images[0][0]
                    elif len(images) > 1:
                        # Find the best match
                        for img, line in images:
                            if 'jacket' in line.lower() or 'gitadora' in img.lower():
                                best_img = img
                                break
                        if not best_img:
                            best_img = images[0][0] # Fallback to first
                            
                    if best_img:
                        # Standardize space to underscore if wiki returns spaces
                        best_img = best_img.replace(' ', '_')
                        jacket_map[title] = best_img
        except Exception as e:
            print(f"Error fetching batch {i}: {e}")
            
        time.sleep(0.5) # Be nice to the API
        
    return jacket_map

def main():
    import sys
    start_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("Extracting songs from songlist.html...")
    songs = extract_songs_from_html()
    print(f"Found {len(songs)} active songs.")
    
    print("Fetching jacket filenames from RemyWiki API (this may take a couple minutes)...")
    jacket_map = fetch_jackets_from_wiki(songs.keys(), start_offset)
    print(f"Successfully mapped {len(jacket_map)} jackets.")
    
    # We want the key to be the actual jacket filename (e.g. 'Any_Percent.png')
    metadata = {}
    if start_offset > 0:
        try:
            with open('song_metadata.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception:
            pass
            
    # We only want to process the songs that were in our batches if we are using an offset
    keys_list = list(songs.keys())
    songs_to_process = keys_list[start_offset:] if start_offset > 0 else keys_list
    
    for song_key in songs_to_process:
        info = songs[song_key]
        jacket_file = jacket_map.get(song_key)
        
        if jacket_file:
            # Add to metadata using the exact jacket filename
            metadata[jacket_file] = info
        else:
            # Fallback if no image was found on the wiki, use song_key + .png
            metadata[f"{song_key}.png"] = info
            
    with open('song_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print("Updated song_metadata.json!")

if __name__ == '__main__':
    main()
