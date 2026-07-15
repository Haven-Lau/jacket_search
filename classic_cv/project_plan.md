# Classic Computer Vision Image Retrieval (ORB/SIFT)

## Objective
Build a lightweight, serverless web app running in the mobile browser that matches video frames from the camera against a reference database of images using ORB (Oriented FAST and Rotated BRIEF) or SIFT features, filtered via RANSAC homography check.

## Roadmap / Tasks

### Phase 1: Database Generation (Python)
- [ ] Set up Python environment.
- [ ] Write `extract_features.py` using OpenCV (`opencv-python`).
- [ ] Read reference images and compute ORB/SIFT keypoints and descriptors.
- [ ] Export keypoints and descriptors to `db.json` (or a compact binary format).

### Phase 2: Web App Shell
- [ ] Create HTML structure (`index.html`) with target alignment box.
- [ ] Add basic CSS (`style.css`) with premium dark mode and responsive layout.
- [ ] Implement camera stream capturing using `getUserMedia` to HTML5 Canvas.

### Phase 3: OpenCV.js Integration
- [ ] Download and integrate `opencv.js` (WebAssembly version).
- [ ] Load the pre-calculated descriptors database (`db.json`) on start.
- [ ] Implement frame processing loop:
  - Crop target area from canvas.
  - Run ORB feature detector on the cropped frame.
  - Match keypoints against database descriptors.
  - Run RANSAC (Homography) to filter outliers.
  - Rank matches by inlier count.

### Phase 4: UI/UX & Feedback
- [ ] Highlight the matched image name and attributes on screen.
- [ ] (Optional) Draw green matching keypoint lines onto the video canvas overlay.
- [ ] Test performance on mobile browsers (Safari, Chrome) to hit >=15 FPS.
