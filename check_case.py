import json
import os

with open('song_metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

thumbnail_files = os.listdir('thumbnails') if os.path.exists('thumbnails') else []
thumbnail_basenames = {os.path.splitext(f)[0] for f in thumbnail_files}
lower_thumbnail_basenames = {k.lower(): k for k in thumbnail_basenames}

missing_thumbnails = []
case_mismatches = []

for key in metadata:
    basename = os.path.splitext(key)[0]
    if basename not in thumbnail_basenames:
        if basename.lower() in lower_thumbnail_basenames:
            case_mismatches.append((basename, lower_thumbnail_basenames[basename.lower()]))
        else:
            missing_thumbnails.append(basename)

print(f'Total active songs in metadata: {len(metadata)}')
print(f'Missing thumbnails: {len(missing_thumbnails)}')
print(f'Case mismatches (Metadata vs Thumbnail): {len(case_mismatches)}')

for m in case_mismatches[:10]:
    print(f'Case Mismatch: Metadata has "{m[0]}" but thumbnail is "{m[1]}.webp"')

if missing_thumbnails:
    print('--- Missing Thumbnails ---')
    for m in missing_thumbnails[:10]:
        print(m)
