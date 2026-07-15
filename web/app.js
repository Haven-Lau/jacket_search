// Core constants
const SPRITE_COLS = 32;
const GRID_W = 80;
const GRID_H = 120;
const PATCH_SIZE = 40;

// Ideal Template Centroids (224x224 coordinates)
const TEMPLATE_DOTS_ALL = [
    { cx: 51.17, cy: 52.83, slot: 0 },   // 0: Top-Left
    { cx: 168.84, cy: 52.84, slot: 1 },  // 1: Top-Right
    { cx: 110.03, cy: 111.18, slot: 2 }, // 2: Center
    { cx: 51.18, cy: 170.69, slot: 3 },  // 3: Bottom-Left
    { cx: 168.34, cy: 170.66, slot: 4 }  // 4: Bottom-Right
];

// Grid slots in the 120x80 compiled patch grid
const GRID_SLOTS = [
    { x: 20, y: 20 },  // Slot 0 (Top-Left)
    { x: 60, y: 20 },  // Slot 1 (Top-Right)
    { x: 20, y: 60 },  // Slot 2 (Center)
    { x: 60, y: 60 },  // Slot 3 (Bottom-Left)
    { x: 20, y: 100 }  // Slot 4 (Bottom-Right)
];

// Global State
let jacketNames = [];
let databaseLoaded = false;
let databaseNormalizedPixels = null; // Float32Array of pre-normalized pixels
let activeMaskName = "5-dot";
let videoStream = null;
let processingInterval = null;

// DOM Elements
const video = document.getElementById("webcam-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas.getContext("2d");
const maskSelect = document.getElementById("mask-select");
const cameraSelect = document.getElementById("camera-select");
const fileInput = document.getElementById("file-input");
const matchesList = document.getElementById("matches-list");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");

const debugQueryCanvas = document.getElementById("debug-query-canvas");
const dqCtx = debugQueryCanvas.getContext("2d");
const debugRefCanvas = document.getElementById("debug-ref-canvas");
const drCtx = debugRefCanvas.getContext("2d");

// Create offscreen canvas for resizing
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = 224;
offscreenCanvas.height = 224;
const offCtx = offscreenCanvas.getContext("2d");

// Precompute the circular eroded mask
const erodedMask = new Uint8Array(GRID_W * GRID_H);
const erodedRadius = 15.5; // matching Python's eroded mask (iterations=2)
let maskPixelCount = 0;

for (let y = 0; y < GRID_H; y++) {
    for (let x = 0; x < GRID_W; x++) {
        let inside = false;
        // Check distance to closest slot center
        for (const slot of GRID_SLOTS) {
            const dist = Math.sqrt((x - slot.x)**2 + (y - slot.y)**2);
            if (dist <= erodedRadius) {
                inside = true;
                break;
            }
        }
        erodedMask[y * GRID_W + x] = inside ? 1 : 0;
        if (inside) maskPixelCount++;
    }
}

// 1. Initial Database Loading
async function initDatabase() {
    try {
        loadingOverlay.style.opacity = "1";
        loadingOverlay.style.display = "flex";
        
        loadingText.textContent = "Fetching index mapping...";
        const indexRes = await fetch("jacket_index.json");
        if (!indexRes.ok) throw new Error("Failed to load jacket_index.json");
        jacketNames = await indexRes.ok ? await indexRes.json() : [];
        console.log(`Loaded ${jacketNames.length} song names.`);
        
        loadingText.textContent = "Downloading sprite sheet (~3.5 MB)...";
        const spriteImg = new Image();
        spriteImg.src = "jackets_sprite.webp";
        
        await new Promise((resolve, reject) => {
            spriteImg.onload = resolve;
            spriteImg.onerror = reject;
        });
        
        loadingText.textContent = "Pre-processing database pixels...";
        
        // Draw sprite to an offscreen canvas to get pixels
        const canvas = document.createElement("canvas");
        canvas.width = spriteImg.width;
        canvas.height = spriteImg.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(spriteImg, 0, 0);
        
        const numJackets = jacketNames.length;
        // Each jacket will have pre-normalized RGB values (3 floats per pixel)
        databaseNormalizedPixels = new Float32Array(numJackets * maskPixelCount * 3);
        
        for (let i = 0; i < numJackets; i++) {
            const col = i % SPRITE_COLS;
            const row = Math.floor(i / SPRITE_COLS);
            const sx = col * GRID_W;
            const sy = row * GRID_H;
            
            // Extract grid pixel data
            const imgData = ctx.getImageData(sx, sy, GRID_W, GRID_H).data;
            
            // Filter pixels under the eroded mask
            const rChannel = [];
            const gChannel = [];
            const bChannel = [];
            
            for (let idx = 0; idx < GRID_W * GRID_H; idx++) {
                if (erodedMask[idx] === 1) {
                    rChannel.push(imgData[idx * 4]);
                    gChannel.push(imgData[idx * 4 + 1]);
                    bChannel.push(imgData[idx * 4 + 2]);
                }
            }
            
            // Calculate mean and std for each channel
            const getStats = (channel) => {
                const sum = channel.reduce((a, b) => a + b, 0);
                const mean = sum / channel.length;
                const variance = channel.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / channel.length;
                const std = Math.sqrt(variance) + 1e-6;
                return { mean, std };
            };
            
            const rStats = getStats(rChannel);
            const gStats = getStats(gChannel);
            const bStats = getStats(bChannel);
            
            // Write normalized pixels to global database buffer
            const offset = i * maskPixelCount * 3;
            for (let idx = 0; idx < maskPixelCount; idx++) {
                databaseNormalizedPixels[offset + idx * 3]     = (rChannel[idx] - rStats.mean) / rStats.std;
                databaseNormalizedPixels[offset + idx * 3 + 1] = (gChannel[idx] - gStats.mean) / gStats.std;
                databaseNormalizedPixels[offset + idx * 3 + 2] = (bChannel[idx] - bStats.mean) / bStats.std;
            }
        }
        
        console.log("Database pre-processing complete!");
        databaseLoaded = true;
        
        // Hide loading overlay
        loadingOverlay.style.opacity = "0";
        setTimeout(() => loadingOverlay.style.display = "none", 500);
        
    } catch (err) {
        console.error(err);
        loadingText.textContent = "Initialization failed. Check console errors.";
    }
}

// 2. Camera Access Configuration
async function setupCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === "videoinput");
        
        cameraSelect.innerHTML = "";
        if (videoDevices.length === 0) {
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            return;
        }
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            // Prioritize back camera on mobile
            if (device.label.toLowerCase().includes("back") || device.label.toLowerCase().includes("rear")) {
                option.selected = true;
            }
            cameraSelect.appendChild(option);
        });
        
        // Start streaming
        await startCamera(cameraSelect.value);
        
        cameraSelect.onchange = async () => {
            await startCamera(cameraSelect.value);
        };
        
    } catch (err) {
        console.error("Camera detection error:", err);
    }
}

async function startCamera(deviceId) {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
    
    const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId }, width: 480, height: 480 } : { facingMode: "environment", width: 480, height: 480 }
    };
    
    try {
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        video.onloadedmetadata = () => {
            video.play();
            resizeCanvas();
        };
    } catch (err) {
        console.error("Error accessing camera:", err);
    }
}

function resizeCanvas() {
    overlayCanvas.width = video.clientWidth || 480;
    overlayCanvas.height = video.clientHeight || 480;
    drawOverlay();
}

// 3. Draw Target Overlay Box and Dots
function drawOverlay() {
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, w, h);
    
    // Draw target bounding box (centered, 80% of width)
    const boxSize = Math.min(w, h) * 0.85;
    const bx = (w - boxSize) / 2;
    const by = (h - boxSize) / 2;
    
    overlayCtx.strokeStyle = "rgba(0, 210, 255, 0.4)";
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(bx, by, boxSize, boxSize);
    
    // Highlight corners
    overlayCtx.fillStyle = "var(--primary)";
    const cornerSize = 15;
    // Top-Left
    overlayCtx.fillRect(bx, by, cornerSize, 4);
    overlayCtx.fillRect(bx, by, 4, cornerSize);
    // Top-Right
    overlayCtx.fillRect(bx + boxSize - cornerSize, by, cornerSize, 4);
    overlayCtx.fillRect(bx + boxSize - 4, by, 4, cornerSize);
    // Bottom-Left
    overlayCtx.fillRect(bx, by + boxSize - 4, cornerSize, 4);
    overlayCtx.fillRect(bx, by + boxSize - cornerSize, 4, cornerSize);
    // Bottom-Right
    overlayCtx.fillRect(bx + boxSize - cornerSize, by + boxSize - 4, cornerSize, 4);
    overlayCtx.fillRect(bx + boxSize - 4, by + boxSize - cornerSize, 4, cornerSize);
    
    // Draw alignment circle overlays
    const dots = getActiveTemplateDots();
    overlayCtx.fillStyle = "rgba(0, 255, 170, 0.3)";
    overlayCtx.strokeStyle = "rgba(0, 255, 170, 0.8)";
    overlayCtx.lineWidth = 1.5;
    
    for (const dot of dots) {
        // Map 224x224 template coordinates to screen target box coordinates
        const dx = bx + (dot.cx / 224) * boxSize;
        const dy = by + (dot.cy / 224) * boxSize;
        const r = (19 / 224) * boxSize; // Radius corresponding to diameter ~38 px
        
        overlayCtx.beginPath();
        overlayCtx.arc(dx, dy, r, 0, 2 * Math.PI);
        overlayCtx.fill();
        overlayCtx.stroke();
    }
}

function getActiveTemplateDots() {
    if (activeMaskName === "5-dot") {
        return TEMPLATE_DOTS_ALL;
    } else if (activeMaskName === "3-dot") {
        // Top-Left, Center, Bottom-Right
        return [TEMPLATE_DOTS_ALL[0], TEMPLATE_DOTS_ALL[2], TEMPLATE_DOTS_ALL[4]];
    } else {
        // Center only
        return [TEMPLATE_DOTS_ALL[2]];
    }
}

// 4. Connected Components Labeling (CCL)
function findQueryCentroids(binaryMask, width, height) {
    const visited = new Uint8Array(width * height);
    const components = [];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (binaryMask[idx] === 1 && visited[idx] === 0) {
                // BFS to label component
                const queue = [idx];
                visited[idx] = 1;
                let sumX = 0;
                let sumY = 0;
                let count = 0;
                
                let head = 0;
                while (head < queue.length) {
                    const curr = queue[head++];
                    const px = curr % width;
                    const py = Math.floor(curr / width);
                    sumX += px;
                    sumY += py;
                    count++;
                    
                    // Check 4 neighbors
                    if (px + 1 < width) {
                        const nidx = curr + 1;
                        if (binaryMask[nidx] === 1 && visited[nidx] === 0) { visited[nidx] = 1; queue.push(nidx); }
                    }
                    if (px - 1 >= 0) {
                        const nidx = curr - 1;
                        if (binaryMask[nidx] === 1 && visited[nidx] === 0) { visited[nidx] = 1; queue.push(nidx); }
                    }
                    if (py + 1 < height) {
                        const nidx = curr + width;
                        if (binaryMask[nidx] === 1 && visited[nidx] === 0) { visited[nidx] = 1; queue.push(nidx); }
                    }
                    if (py - 1 >= 0) {
                        const nidx = curr - width;
                        if (binaryMask[nidx] === 1 && visited[nidx] === 0) { visited[nidx] = 1; queue.push(nidx); }
                    }
                }
                
                if (count > 70) { // Keep components above area threshold
                    components.push({
                        cx: sumX / count,
                        cy: sumY / count,
                        area: count
                    });
                }
            }
        }
    }
    
    // Sort by area descending
    components.sort((a, b) => b.area - a.area);
    return components.slice(0, 5); // return top 5
}

// 5. Query Preprocessing, Alignment, and Matching
function processQueryFrame() {
    if (!databaseLoaded) return;
    
    // 1. Capture current frame from webcam and draw/resize to offscreen 224x224 canvas
    // We crop the centered target box area
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) return;
    
    const boxSize = Math.min(vw, vh) * 0.85;
    const sx = (vw - boxSize) / 2;
    const sy = (vh - boxSize) / 2;
    
    offCtx.drawImage(video, sx, sy, boxSize, boxSize, 0, 0, 224, 224);
    
    // 2. Segment the dots dynamically
    // Sample background color near corners (mean of four 10x10 corners)
    const imgData = offCtx.getImageData(0, 0, 224, 224);
    const pixels = imgData.data;
    
    let sumR = 0, sumG = 0, sumB = 0, count = 0;
    const sampleCorner = (cx, cy) => {
        for (let dy = 0; dy < 10; dy++) {
            for (let dx = 0; dx < 10; dx++) {
                const idx = ((cy + dy) * 224 + (cx + dx)) * 4;
                sumR += pixels[idx];
                sumG += pixels[idx + 1];
                sumB += pixels[idx + 2];
                count++;
            }
        }
    };
    sampleCorner(0, 0);
    sampleCorner(224 - 10, 0);
    sampleCorner(0, 224 - 10);
    sampleCorner(224 - 10, 224 - 10);
    
    const bgR = sumR / count;
    const bgG = sumG / count;
    const bgB = sumB / count;
    
    // Threshold color distance from background
    const binaryMask = new Uint8Array(224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
        const r = pixels[i * 4];
        const g = pixels[i * 4 + 1];
        const b = pixels[i * 4 + 2];
        const dist = Math.sqrt((r - bgR)**2 + (g - bgG)**2 + (b - bgB)**2);
        binaryMask[i] = dist > 70 ? 1 : 0;
    }
    
    // 3. Find Centroids & Align
    const queryCentroids = findQueryCentroids(binaryMask, 224, 224);
    const activeTemplateDots = getActiveTemplateDots();
    const M = activeTemplateDots.length;
    
    if (queryCentroids.length < M) {
        displayMessage(`Alignment warning: Align all ${M} dots to targets.`);
        return;
    }
    
    // Match query centroids to template centroids using nearest neighbors
    const matchedQueryDots = [];
    for (const td of activeTemplateDots) {
        let minDist = Infinity;
        let closestQ = null;
        for (const qc of queryCentroids) {
            const d = Math.sqrt((qc.cx - td.cx)**2 + (qc.cy - td.cy)**2);
            if (d < minDist) {
                minDist = d;
                closestQ = qc;
            }
        }
        matchedQueryDots.push(closestQ);
    }
    
    // Calculate average translation offsets dx, dy
    let sumDx = 0, sumDy = 0;
    for (let i = 0; i < M; i++) {
        sumDx += activeTemplateDots[i].cx - matchedQueryDots[i].cx;
        sumDy += activeTemplateDots[i].cy - matchedQueryDots[i].cy;
    }
    const dx = sumDx / M;
    const dy = sumDy / M;
    
    // 4. Build aligned query grid (120x80)
    // Draw solid black to debug canvas
    dqCtx.fillStyle = "#000000";
    dqCtx.fillRect(0, 0, GRID_W, GRID_H);
    
    // Draw the active dots into their grid slots
    // For slots that are not active under the current mask, they will stay solid black (matching the database)
    for (const td of activeTemplateDots) {
        const slot = GRID_SLOTS[td.slot];
        
        // Coordinates in query image shifted to align with template
        // We crop PATCH_SIZE x PATCH_SIZE centered at the template centroid (aligned)
        const cropX = td.cx - dx - PATCH_SIZE / 2;
        const cropY = td.cy - dy - PATCH_SIZE / 2;
        
        dqCtx.drawImage(
            offscreenCanvas,
            cropX, cropY, PATCH_SIZE, PATCH_SIZE,
            slot.x - PATCH_SIZE / 2, slot.y - PATCH_SIZE / 2, PATCH_SIZE, PATCH_SIZE
        );
    }
    
    // 5. Apply circular mask dynamically to query grid on canvas (blacking out corners)
    // In javascript, we can draw circular clipping paths for each slot to erase corner pixels
    dqCtx.save();
    dqCtx.globalCompositeOperation = "destination-in";
    dqCtx.fillStyle = "#000000";
    for (const slot of GRID_SLOTS) {
        dqCtx.beginPath();
        // Use eroded radius 15.5 for strict matching (erases background color bleed)
        dqCtx.arc(slot.x, slot.y, erodedRadius, 0, 2 * Math.PI);
        dqCtx.fill();
    }
    dqCtx.restore();
    
    // 6. Extract query pixels under the mask and normalize
    const queryGridData = dqCtx.getImageData(0, 0, GRID_W, GRID_H).data;
    const queryR = [];
    const queryG = [];
    const queryB = [];
    
    for (let idx = 0; idx < GRID_W * GRID_H; idx++) {
        if (erodedMask[idx] === 1) {
            queryR.push(queryGridData[idx * 4]);
            queryG.push(queryGridData[idx * 4 + 1]);
            queryB.push(queryGridData[idx * 4 + 2]);
        }
    }
    
    const getStats = (channel) => {
        const sum = channel.reduce((a, b) => a + b, 0);
        const mean = sum / channel.length;
        const variance = channel.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / channel.length;
        const std = Math.sqrt(variance) + 1e-6;
        return { mean, std };
    };
    
    const qRStats = getStats(queryR);
    const qGStats = getStats(queryG);
    const qBStats = getStats(queryB);
    
    const queryNorm = new Float32Array(maskPixelCount * 3);
    for (let idx = 0; idx < maskPixelCount; idx++) {
        queryNorm[idx * 3]     = (queryR[idx] - qRStats.mean) / qRStats.std;
        queryNorm[idx * 3 + 1] = (queryG[idx] - qGStats.mean) / qGStats.std;
        queryNorm[idx * 3 + 2] = (queryB[idx] - qBStats.mean) / qBStats.std;
    }
    
    // 7. Compute NCC against all database items
    const numJackets = jacketNames.length;
    const similarityResults = [];
    
    for (let j = 0; j < numJackets; j++) {
        const dbOffset = j * maskPixelCount * 3;
        let sumProd = 0;
        
        for (let idx = 0; idx < maskPixelCount * 3; idx++) {
            sumProd += queryNorm[idx] * databaseNormalizedPixels[dbOffset + idx];
        }
        
        const score = sumProd / (maskPixelCount * 3);
        similarityResults.push({ name: jacketNames[j], index: j, score });
    }
    
    // Sort and rank matches
    similarityResults.sort((a, b) => b.score - a.score);
    
    // 8. Render Results UI
    renderMatches(similarityResults.slice(0, 3));
}

function displayMessage(msg) {
    matchesList.innerHTML = `<div class="empty-state"><p>${msg}</p></div>`;
}

function renderMatches(topMatches) {
    matchesList.innerHTML = "";
    
    topMatches.forEach((match, index) => {
        const rankClass = index === 0 ? "rank-1" : "";
        const scorePercentage = Math.max(0, Math.min(100, match.score * 100));
        
        const matchItem = document.createElement("div");
        matchItem.className = `match-item ${rankClass}`;
        
        // Draw matched reference grid to offscreen to fetch as image
        // Or dynamically draw it to the debug reference canvas
        if (index === 0) {
            drawRefGridToCanvas(match.index);
        }
        
        // We link directly to RemyWiki's folder structure or local folder
        // For github pages, we can just load the local cropped jacket thumbnail from jackets directory
        const thumbUrl = `../jackets/${encodeURIComponent(match.name)}`;
        
        matchItem.innerHTML = `
            <div class="match-rank">${index + 1}</div>
            <img class="match-thumbnail" src="${thumbUrl}" alt="Jacket">
            <div class="match-info">
                <div class="match-name" title="${match.name}">${match.name}</div>
                <div class="score-container">
                    <div class="score-bar-wrapper">
                        <div class="score-bar" style="width: ${scorePercentage}%"></div>
                    </div>
                    <div class="score-text">${match.score.toFixed(4)}</div>
                </div>
            </div>
        `;
        
        matchesList.appendChild(matchItem);
    });
}

function drawRefGridToCanvas(jacketIdx) {
    // Draws the matched reference grid from jackets_sprite.webp to the debug reference canvas
    const col = jacketIdx % SPRITE_COLS;
    const row = Math.floor(jacketIdx / SPRITE_COLS);
    const sx = col * GRID_W;
    const sy = row * GRID_H;
    
    // Load the sprite image element and draw to canvas
    const spriteImg = new Image();
    spriteImg.src = "jackets_sprite.webp";
    
    spriteImg.onload = () => {
        drCtx.clearRect(0, 0, GRID_W, GRID_H);
        drCtx.drawImage(spriteImg, sx, sy, GRID_W, GRID_H, 0, 0, GRID_W, GRID_H);
    };
}

// 6. Test Image Upload Handler (Local Debugging Fallback)
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            // Draw static image to offscreen canvas and trigger matching
            offCtx.clearRect(0, 0, 224, 224);
            offCtx.drawImage(img, 0, 0, img.width, img.height, 0, 0, 224, 224);
            
            // Trigger processQueryFrame using offscreenCanvas instead of webcam
            // We temporarily override the video drawer
            const originalProcess = processQueryFrame;
            // Execute matching on the uploaded image directly
            processQueryFrame();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

// 7. Event Listeners & Initialize
maskSelect.addEventListener("change", (e) => {
    activeMaskName = e.target.value;
    drawOverlay();
    // Clear results UI when mask changes
    displayMessage("Waiting for query feed...");
});

fileInput.addEventListener("change", handleFileUpload);
window.addEventListener("resize", resizeCanvas);

// Initialize App
async function init() {
    await initDatabase();
    await setupCameras();
    
    // Start real-time processing loop at 8 FPS (every 125ms) to save CPU
    processingInterval = setInterval(processQueryFrame, 125);
}

init();
