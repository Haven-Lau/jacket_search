// Core constants
const SPRITE_COLS = 32;
const GRID_W = 80;
const GRID_H = 120;
const PATCH_SIZE = 40;

// The fixed slots in the 80x120 patch grid
const GRID_SLOTS = [
    { x: 20, y: 20 },  // 0: Top-Left
    { x: 60, y: 20 },  // 1: Top-Right
    { x: 20, y: 60 },  // 2: Center
    { x: 60, y: 60 },  // 3: Bottom-Left
    { x: 20, y: 100 }  // 4: Bottom-Right
];

const BASE_CENTROIDS = [
    { cx: 51, cy: 53 },   // TL
    { cx: 169, cy: 53 },  // TR
    { cx: 110, cy: 111 }, // C
    { cx: 51, cy: 171 },  // BL
    { cx: 168, cy: 171 }  // BR
];

// Global State
let jacketNames = [];
let jacketUrlMap = {};
let songMetadata = {};
let databaseLoaded = false;
let spriteCanvas = null; // Holds the 2560x8280 sprite
let maskConfigs = {}; // Stores configuration and pre-normalized DB per mask
let activeMaskName = "1-dot";
let isLiveFeed = true;
let sessionTopMatches = [];
let isFrozen = false;
let videoStream = null;
let processingInterval = null;
let bannedEntries = new Set();
let enableEdgeFade = false;
let enableColorFilter = true;

// DOM Elements
const video = document.getElementById("webcam-video");
const overlayCanvas = document.getElementById("overlay-canvas");
const overlayCtx = overlayCanvas.getContext("2d");
const maskSelect = document.getElementById("mask-select");
const matchesList = document.getElementById("matches-list");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const statusOverlay = document.getElementById("status-overlay");
const clearMatchesBtn = document.getElementById("clear-matches-btn");
const freezeBtn = document.getElementById("freeze-btn");
const exposureContainer = document.getElementById("exposure-container");
const exposureSlider = document.getElementById("exposure-slider");

const settingsBtn = document.getElementById("settings-btn");
const closeSettingsBtn = document.getElementById("close-settings-btn");
const settingsModal = document.getElementById("settings-modal");
const cameraList = document.getElementById("camera-list");
const refreshCameraBtn = document.getElementById("refresh-camera-btn");

let currentDeviceId = null;

settingsBtn.addEventListener("click", () => settingsModal.classList.remove("hidden"));
closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));
settingsModal.addEventListener("click", (e) => {
    if (e.target === settingsModal) {
        settingsModal.classList.add("hidden");
    }
});
refreshCameraBtn.addEventListener("click", () => {
    if (currentDeviceId) startCamera(currentDeviceId);
    else setupCameras();
});

freezeBtn.addEventListener("click", () => {
    isFrozen = !isFrozen;
    freezeBtn.innerHTML = isFrozen 
        ? `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>` 
        : `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>`;
    freezeBtn.className = isFrozen ? "btn primary-btn icon-btn" : "btn outline-btn icon-btn";
    freezeBtn.title = isFrozen ? "Unfreeze" : "Freeze";
});

clearMatchesBtn.addEventListener("click", () => {
    bannedEntries.clear();
    sessionTopMatches = [];
    matchesList.innerHTML = `<div class="empty-state">
            <div class="empty-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 22h14"/><path d="M5 2h14"/><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"/><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"/></svg>
            </div>
            <p>Waiting for clear visual feed...</p>
        </div>`;
});

const toggleFadeBtn = document.getElementById("toggle-fade-btn");
if (toggleFadeBtn) {
    toggleFadeBtn.addEventListener("click", () => {
        enableEdgeFade = !enableEdgeFade;
        toggleFadeBtn.textContent = `Fade: ${enableEdgeFade ? "ON" : "OFF"}`;
        toggleFadeBtn.className = enableEdgeFade ? "btn primary-btn" : "btn outline-btn";
    });
}

const toggleColorFilterBtn = document.getElementById("toggle-color-filter-btn");
if (toggleColorFilterBtn) {
    toggleColorFilterBtn.addEventListener("click", () => {
        enableColorFilter = !enableColorFilter;
        toggleColorFilterBtn.textContent = `Filter: ${enableColorFilter ? "ON" : "OFF"}`;
        toggleColorFilterBtn.className = enableColorFilter ? "btn primary-btn" : "btn outline-btn";
    });
}

// Offscreen canvas for fast video frame extraction
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = 224;
offscreenCanvas.height = 224;
const offCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

// Math Utilities
function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, v = max;
    const d = max - min;
    s = max === 0 ? 0 : d / max;
    if (max === min) {
        h = 0; 
    } else {
        switch (max) {
            case r: h = (g - b) / d + (g < b ? 6 : 0); break;
            case g: h = (b - r) / d + 2; break;
            case b: h = (r - g) / d + 4; break;
        }
        h /= 6;
    }
    return [h * 360, s, v]; 
}

function invert3x3(m) {
    const det = m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
              - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
              + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (Math.abs(det) < 1e-8) return null;
    return [
        [(m[1][1]*m[2][2] - m[1][2]*m[2][1])/det, (m[0][2]*m[2][1] - m[0][1]*m[2][2])/det, (m[0][1]*m[1][2] - m[0][2]*m[1][1])/det],
        [(m[1][2]*m[2][0] - m[1][0]*m[2][2])/det, (m[0][0]*m[2][2] - m[0][2]*m[2][0])/det, (m[0][2]*m[1][0] - m[0][0]*m[1][2])/det],
        [(m[1][0]*m[2][1] - m[1][1]*m[2][0])/det, (m[0][1]*m[2][0] - m[0][0]*m[2][1])/det, (m[0][0]*m[1][1] - m[0][1]*m[1][0])/det]
    ];
}

// Estimates a Similarity Transform (4-DoF: Scale, Rotation, Translation) mapping srcPts -> dstPts using linear least squares
function estimateAffine(srcPts, dstPts, srcAreas = null, dstAreas = null) {
    const N = srcPts.length;
    if (N === 0) return null;
    if (N < 2) {
        // Translation + Scale fallback for 1-dot
        let scale = 1;
        if (srcAreas && dstAreas && srcAreas[0] > 0) {
            const rawScale = Math.sqrt(dstAreas[0] / srcAreas[0]);
            scale = rawScale * 0.4 + 1.0 * 0.6; // Rely on user zoom 60%
            scale = Math.max(0.75, Math.min(scale, 1.5)); // Constrain scale to 75%-150%
        }
        const cx_t = srcPts[0].x;
        const cy_t = srcPts[0].y;
        const cx_q = dstPts[0].x;
        const cy_q = dstPts[0].y;
        
        const tx = cx_q - scale * cx_t;
        const ty = cy_q - scale * cy_t;
        return { a: scale, b: 0, tx: tx, c: 0, d: scale, ty: ty };
    }
    
    let sumX = 0, sumY = 0, sumU = 0, sumV = 0;
    for (let i = 0; i < N; i++) {
        sumX += srcPts[i].x; sumY += srcPts[i].y;
        sumU += dstPts[i].x; sumV += dstPts[i].y;
    }
    
    const meanX = sumX / N, meanY = sumY / N;
    const meanU = sumU / N, meanV = sumV / N;
    
    let den = 0, numA = 0, numB = 0;
    for (let i = 0; i < N; i++) {
        const dx = srcPts[i].x - meanX, dy = srcPts[i].y - meanY;
        const du = dstPts[i].x - meanU, dv = dstPts[i].y - meanV;
        
        den += dx*dx + dy*dy;
        numA += du*dx + dv*dy;
        numB += dv*dx - du*dy;
    }
    
    if (den < 1e-6) {
        // Fallback to translation if points are degenerate
        return { a: 1, b: 0, tx: meanU - meanX, c: 0, d: 1, ty: meanV - meanY };
    }
    
    const a = numA / den;
    const b = numB / den;
    
    const tx = meanU - a * meanX + b * meanY;
    const ty = meanV - b * meanX - a * meanY;
    
    return { a: a, b: -b, tx: tx, c: b, d: a, ty: ty };
}

// Warp using nearest neighbor
function warpAffineNearest(srcImgData, dstWidth, dstHeight, M) {
    const dst = new Uint8ClampedArray(dstWidth * dstHeight * 4);
    const src = srcImgData.data;
    const sw = srcImgData.width, sh = srcImgData.height;
    
    for (let dy = 0; dy < dstHeight; dy++) {
        for (let dx = 0; dx < dstWidth; dx++) {
            // Map dst -> src
            const sx = Math.round(M.a * dx + M.b * dy + M.tx);
            const sy = Math.round(M.c * dx + M.d * dy + M.ty);
            const dstIdx = (dy * dstWidth + dx) * 4;
            
            if (sx >= 0 && sx < sw && sy >= 0 && sy < sh) {
                const srcIdx = (sy * sw + sx) * 4;
                dst[dstIdx]   = src[srcIdx];
                dst[dstIdx+1] = src[srcIdx+1];
                dst[dstIdx+2] = src[srcIdx+2];
                dst[dstIdx+3] = src[srcIdx+3];
            } else {
                dst[dstIdx] = 0; dst[dstIdx+1] = 0; dst[dstIdx+2] = 0; dst[dstIdx+3] = 255;
            }
        }
    }
    return new ImageData(dst, dstWidth, dstHeight);
}

// Morphological Erode (3x3 kernel)
function erodeMask(mask, w, h) {
    const eroded = new Uint8Array(mask.length);
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const i = y * w + x;
            if (mask[i] && mask[i-1] && mask[i+1] && mask[i-w] && mask[i+w] &&
                mask[i-w-1] && mask[i-w+1] && mask[i+w-1] && mask[i+w+1]) {
                eroded[i] = 1;
            }
        }
    }
    return eroded;
}

// CCL for masks
function getDotProperties(binaryMask, width, height) {
    const visited = new Uint8Array(width * height);
    const dots = [];
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = y * width + x;
            if (binaryMask[idx] === 1 && visited[idx] === 0) {
                const queue = [idx];
                visited[idx] = 1;
                let sumX = 0, sumY = 0, count = 0;
                let head = 0;
                while (head < queue.length) {
                    const curr = queue[head++];
                    const px = curr % width, py = Math.floor(curr / width);
                    sumX += px; sumY += py; count++;
                    
                    if (px + 1 < width) { const n = curr + 1; if (binaryMask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
                    if (px - 1 >= 0) { const n = curr - 1; if (binaryMask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
                    if (py + 1 < height) { const n = curr + width; if (binaryMask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
                    if (py - 1 >= 0) { const n = curr - width; if (binaryMask[n] && !visited[n]) { visited[n] = 1; queue.push(n); } }
                }
                if (count > 100) dots.push({ cx: sumX / count, cy: sumY / count, area: count });
            }
        }
    }
    
    // Sort top-to-bottom, left-to-right
    dots.sort((a, b) => {
        if (Math.abs(a.cy - b.cy) > 20) return a.cy - b.cy;
        return a.cx - b.cx;
    });
    return dots;
}

// 1. Initial Database Loading
async function initDatabase() {
    try {
        loadingOverlay.classList.remove("hidden");
        
        loadingText.textContent = "Fetching index mapping...";
        const indexRes = await fetch("jacket_index.json");
        if (!indexRes.ok) throw new Error("Failed to load jacket_index.json");
        jacketNames = await indexRes.json();
        
        loadingText.textContent = "Fetching URLs mapping...";
        const urlRes = await fetch("jacket_urls.txt");
        if (urlRes.ok) {
            const urlsText = await urlRes.text();
            const urls = urlsText.split('\n').filter(l => l.trim().length > 0);
            for (const url of urls) {
                const rawFilename = url.split('/').pop();
                let decoded = decodeURIComponent(rawFilename);
                const invalidChars = '<>:"/\\|?*';
                for (const char of invalidChars) decoded = decoded.split(char).join('_');
                jacketUrlMap[decoded] = url.trim();
            }
        }
        
        loadingText.textContent = "Fetching metadata...";
        const metaRes = await fetch("song_metadata.json");
        if (metaRes.ok) {
            songMetadata = await metaRes.json();
        }
        
        loadingText.textContent = "Loading sprite sheet...";
        const progressBarContainer = document.getElementById("progress-container");
        const progressBar = document.getElementById("progress-bar");
        if(progressBarContainer) progressBarContainer.style.display = "block";

        const spriteImg = new Image();
        
        let loadedFromCache = false;
        let cache = null;
        try {
            if ('caches' in window) {
                cache = await caches.open('jacket-sprite-cache-v1');
                const cachedResponse = await cache.match('jackets_sprite.webp');
                if (cachedResponse) {
                    loadingText.textContent = "Loading sprite from cache...";
                    if (progressBar) progressBar.style.width = '100%';
                    const blob = await cachedResponse.blob();
                    await new Promise((resolve) => {
                        spriteImg.onload = () => { URL.revokeObjectURL(spriteImg.src); resolve(); };
                        spriteImg.src = URL.createObjectURL(blob);
                    });
                    loadedFromCache = true;
                }
            }
        } catch (e) {
            console.warn("Cache API not available or failed", e);
        }

        if (!loadedFromCache) {
            loadingText.textContent = "Downloading sprite sheet (~3.5 MB)...";
            await new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open('GET', 'jackets_sprite.webp', true);
                xhr.responseType = 'blob';
                xhr.onprogress = (e) => {
                    if (e.lengthComputable) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        if (progressBar) progressBar.style.width = percentComplete + '%';
                    }
                };
                xhr.onload = () => {
                    if (xhr.status === 200) {
                        const blob = xhr.response;
                        if (cache) {
                            try { cache.put('jackets_sprite.webp', new Response(blob)); } 
                            catch(e) { console.warn("Failed to cache sprite", e); }
                        }
                        spriteImg.onload = () => {
                            URL.revokeObjectURL(spriteImg.src);
                            resolve();
                        };
                        spriteImg.src = URL.createObjectURL(blob);
                    } else {
                        reject(new Error("Failed to load sprite"));
                    }
                };
                xhr.onerror = reject;
                xhr.send();
            });
        }

        if(progressBarContainer) progressBarContainer.style.display = "none";
        
        // Cache sprite raw pixels
        spriteCanvas = document.createElement("canvas");
        spriteCanvas.width = spriteImg.width;
        spriteCanvas.height = spriteImg.height;
        const sCtx = spriteCanvas.getContext("2d", { willReadFrequently: true });
        sCtx.drawImage(spriteImg, 0, 0);
        
        loadingText.textContent = "Processing masks and pre-computing NCC vectors...";
        await prepareMaskConfigs(sCtx);
        
        databaseLoaded = true;
        loadingOverlay.classList.add("hidden");
        
    } catch (err) {
        console.error(err);
        loadingText.textContent = "Initialization failed. Check console errors.";
    }
}

async function prepareMaskConfigs(sCtx) {
    const maskFiles = ["5-dot", "3-dot", "1-dot"];
    
    for (const name of maskFiles) {
        const maskImg = new Image();
        maskImg.src = `masks/${name}-mask.png`;
        await new Promise((resolve) => { maskImg.onload = resolve; });
        
        const mCanvas = document.createElement("canvas");
        mCanvas.width = 224; mCanvas.height = 224;
        const mCtx = mCanvas.getContext("2d");
        mCtx.drawImage(maskImg, 0, 0, 224, 224);
        
        const maskData = mCtx.getImageData(0, 0, 224, 224).data;
        const binary224 = new Uint8Array(224 * 224);
        for (let i = 0; i < 224 * 224; i++) {
            // Convert to grayscale and threshold > 127
            const gray = (maskData[i*4] + maskData[i*4+1] + maskData[i*4+2]) / 3;
            binary224[i] = gray > 127 ? 1 : 0;
        }
        
        const templateDots = getDotProperties(binary224, 224, 224);
        
        // Erode twice
        const eroded224 = erodeMask(erodeMask(binary224, 224, 224), 224, 224);
        
        // Build 80x120 eroded mask from the 40x40 patches around centroids
        const eroded80x120 = new Uint8Array(GRID_W * GRID_H);
        let pixelCount = 0;
        
        // Map each template dot to the correct base centroid index to pick the correct slot
        const tdSlotIndices = templateDots.map(td => {
            let minD = Infinity, minIdx = 0;
            for (let k = 0; k < BASE_CENTROIDS.length; k++) {
                const d = Math.hypot(td.cx - BASE_CENTROIDS[k].cx, td.cy - BASE_CENTROIDS[k].cy);
                if (d < minD) { minD = d; minIdx = k; }
            }
            return minIdx;
        });
        
        for (let i = 0; i < templateDots.length; i++) {
            const td = templateDots[i];
            const slot = GRID_SLOTS[tdSlotIndices[i]];
            for (let dy = -20; dy < 20; dy++) {
                for (let dx = -20; dx < 20; dx++) {
                    const mx = Math.round(td.cx) + dx;
                    const my = Math.round(td.cy) + dy;
                    const gx = slot.x + dx;
                    const gy = slot.y + dy;
                    if (mx >= 0 && mx < 224 && my >= 0 && my < 224 && gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
                        const val = eroded224[my * 224 + mx];
                        eroded80x120[gy * GRID_W + gx] = val;
                        if (val) pixelCount++;
                    }
                }
            }
        }
        
        // Pre-normalize the database for this mask
        const numJackets = jacketNames.length;
        const dbNorms = new Float32Array(numJackets * pixelCount * 3);
        const dbMeanHsv = new Float32Array(numJackets * 2);
        
        for (let j = 0; j < numJackets; j++) {
            const col = j % SPRITE_COLS;
            const row = Math.floor(j / SPRITE_COLS);
            const imgData = sCtx.getImageData(col * GRID_W, row * GRID_H, GRID_W, GRID_H).data;
            
            const rChannel = [], gChannel = [], bChannel = [];
            for (let idx = 0; idx < GRID_W * GRID_H; idx++) {
                if (eroded80x120[idx]) {
                    rChannel.push(imgData[idx*4]);
                    gChannel.push(imgData[idx*4+1]);
                    bChannel.push(imgData[idx*4+2]);
                }
            }
            
            const meanR = rChannel.reduce((a, b) => a + b, 0) / pixelCount;
            const meanG = gChannel.reduce((a, b) => a + b, 0) / pixelCount;
            const meanB = bChannel.reduce((a, b) => a + b, 0) / pixelCount;
            const hsv = rgbToHsv(meanR, meanG, meanB);
            dbMeanHsv[j * 2] = hsv[0];
            dbMeanHsv[j * 2 + 1] = hsv[1];
            
            // Center and divide by L2 norm to make true NCC a simple dot product
            const normalize = (channel) => {
                const mean = channel.reduce((a, b) => a + b, 0) / channel.length;
                const centered = channel.map(v => v - mean);
                const sumSq = centered.reduce((a, b) => a + b*b, 0);
                const norm = Math.sqrt(sumSq) || 1e-6;
                return centered.map(v => v / norm);
            };
            
            const rNorm = normalize(rChannel);
            const gNorm = normalize(gChannel);
            const bNorm = normalize(bChannel);
            
            const offset = j * pixelCount * 3;
            for (let idx = 0; idx < pixelCount; idx++) {
                dbNorms[offset + idx*3]     = rNorm[idx];
                dbNorms[offset + idx*3 + 1] = gNorm[idx];
                dbNorms[offset + idx*3 + 2] = bNorm[idx];
            }
        }
        
        // Generate a high-quality punch-hole mask for the overlay using the original mask image
        const oCanvas = document.createElement("canvas");
        oCanvas.width = maskImg.width;
        oCanvas.height = maskImg.height;
        const oCtx = oCanvas.getContext("2d");
        oCtx.drawImage(maskImg, 0, 0);
        
        const origData = oCtx.getImageData(0, 0, maskImg.width, maskImg.height);
        const punchImgData = new ImageData(maskImg.width, maskImg.height);
        for (let i = 0; i < maskImg.width * maskImg.height; i++) {
            punchImgData.data[i*4] = 255;
            punchImgData.data[i*4+1] = 255;
            punchImgData.data[i*4+2] = 255;
            punchImgData.data[i*4+3] = Math.min(255, origData.data[i*4] * 255 / 238); // Scale so 238 becomes fully transparent hole
        }
        const punchHoleBitmap = await createImageBitmap(punchImgData);
        
        maskConfigs[name] = { templateDots, slotIndices: tdSlotIndices, eroded80x120, pixelCount, dbNorms, dbMeanHsv, punchHoleBitmap };
    }
}

// 2. Camera Access
async function setupCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        
        cameraList.innerHTML = "";
        if (videoDevices.length === 0) {
            cameraList.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No cameras found</div>';
            return;
        }
        
        let selectedDeviceId = null;
        videoDevices.forEach((device, index) => {
            if (device.label.toLowerCase().includes("back") || device.label.toLowerCase().includes("rear")) {
                selectedDeviceId = device.deviceId;
            }
        });
        if (!selectedDeviceId && videoDevices.length > 0) selectedDeviceId = videoDevices[0].deviceId;

        videoDevices.forEach((device, index) => {
            const btn = document.createElement("button");
            btn.className = device.deviceId === selectedDeviceId ? "btn primary-btn" : "btn outline-btn";
            btn.style.width = "100%";
            btn.style.textAlign = "left";
            btn.textContent = device.label || `Camera ${index + 1}`;
            btn.onclick = async () => {
                Array.from(cameraList.children).forEach(c => c.className = "btn outline-btn");
                btn.className = "btn primary-btn";
                selectedDeviceId = device.deviceId;
                await startCamera(device.deviceId);
            };
            cameraList.appendChild(btn);
        });
        
        await startCamera(selectedDeviceId);
    } catch (err) {
        console.error("Camera detection error:", err);
    }
}

async function startCamera(deviceId) {
    currentDeviceId = deviceId;
    if (videoStream) videoStream.getTracks().forEach(t => t.stop());
    const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId }, width: 480, height: 480 } : { facingMode: "environment", width: 480, height: 480 }
    };
    try {
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        video.onloadedmetadata = () => { video.play(); resizeCanvas(); };
        
        // Check and setup exposure controls
        const track = videoStream.getVideoTracks()[0];
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        const videoWrapper = document.querySelector(".video-wrapper");
        
        if (capabilities.exposureMode && capabilities.exposureCompensation) {
            exposureContainer.style.display = "flex";
            if (videoWrapper) videoWrapper.classList.remove("rounded-bottom");
            exposureSlider.min = capabilities.exposureCompensation.min !== undefined ? capabilities.exposureCompensation.min : -3;
            exposureSlider.max = 0; // limit to negative/0 as requested
            exposureSlider.step = capabilities.exposureCompensation.step || 0.1;
            
            // Set default actual camera exposure to 60% of the range
            const minExp = parseFloat(exposureSlider.min);
            const maxExp = parseFloat(exposureSlider.max);
            // 60% of usual: 100% is maxExp (0), 0% is minExp
            const defaultExp = minExp + (maxExp - minExp) * 0.60;
            exposureSlider.value = defaultExp;
            
            // Handle slider changes
            exposureSlider.oninput = async () => {
                try {
                    await track.applyConstraints({
                        advanced: [{
                            exposureMode: "continuous",
                            exposureCompensation: parseFloat(exposureSlider.value)
                        }]
                    });
                } catch (e) {
                    console.error("Failed to apply exposure constraint:", e);
                }
            };
            
            // Apply initial if not zero
            if (parseFloat(exposureSlider.value) !== 0) {
                exposureSlider.oninput();
            }
        } else {
            exposureContainer.style.display = "none";
            if (videoWrapper) videoWrapper.classList.add("rounded-bottom");
        }
    } catch (err) {
        console.error("Error accessing camera:", err);
    }
}

function resizeCanvas() {
    overlayCanvas.width = video.clientWidth || 480;
    overlayCanvas.height = video.clientHeight || 480;
    drawOverlay();
}

// 3. Target Overlay
function drawOverlay() {
    const w = overlayCanvas.width, h = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, w, h);
    
    // Fill the entire canvas with semi-transparent black
    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.65)";
    overlayCtx.fillRect(0, 0, w, h);
    
    const boxSize = Math.min(w, h);
    const bx = (w - boxSize) / 2, by = (h - boxSize) / 2;
    
    if (!maskConfigs[activeMaskName]) return;
    const config = maskConfigs[activeMaskName];
    
    // Punch out holes for the dots
    if (config.punchHoleBitmap) {
        overlayCtx.globalCompositeOperation = "destination-out";
        overlayCtx.drawImage(config.punchHoleBitmap, bx, by, boxSize, boxSize);
        overlayCtx.globalCompositeOperation = "source-over"; // restore default
    }

    // removed purple stroke
    // overlayCtx.strokeStyle = "rgba(99, 102, 241, 0.5)";
    // overlayCtx.strokeRect(bx, by, boxSize, boxSize);
}

// 4. Query Processing & NCC Match
function processQueryFrame() {
    if (!databaseLoaded || !isLiveFeed || isFrozen) return;
    
    const vw = video.videoWidth, vh = video.videoHeight;
    if (vw === 0 || vh === 0) return;
    
    const boxSize = Math.min(vw, vh);
    const sx = (vw - boxSize) / 2, sy = (vh - boxSize) / 2;
    offCtx.drawImage(video, sx, sy, boxSize, boxSize, 0, 0, 224, 224);
    
    runMatchingPipeline();
}


function runMatchingPipeline() {
    const config = maskConfigs[activeMaskName];
    const templateDots = config.templateDots;
    
    // 1. Dynamic Background Sampling
    const imgData = offCtx.getImageData(0, 0, 224, 224);
    const pixels = imgData.data;
    
    let srcPts = [], dstPts = []; // We map Template (src) -> Query (dst)
    let srcAreas = [], dstAreas = [];
    
    if (activeMaskName === "1-dot") {
        // 1-dot: Center-Biased Region Growing
        let sumBgR = 0, sumBgG = 0, sumBgB = 0, countBg = 0;
        for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
                if (x < 5 || x >= 219 || y < 5 || y >= 219) { // 5px border
                    const i = (y*224 + x) * 4;
                    sumBgR += pixels[i]; sumBgG += pixels[i+1]; sumBgB += pixels[i+2]; countBg++;
                }
            }
        }
        const bgR = sumBgR/countBg, bgG = sumBgG/countBg, bgB = sumBgB/countBg;
        
        let sumFgR = 0, sumFgG = 0, sumFgB = 0, countFg = 0;
        const sampleFg = () => {
            for (let dy=107; dy<117; dy++) {
                for (let dx=107; dx<117; dx++) {
                    const i = (dy*224 + dx) * 4;
                    sumFgR += pixels[i]; sumFgG += pixels[i+1]; sumFgB += pixels[i+2]; countFg++;
                }
            }
        };
        sampleFg();
        const fgR = sumFgR/countFg, fgG = sumFgG/countFg, fgB = sumFgB/countFg;
        
        const visited = new Uint8Array(224 * 224);
        const queue = [112 * 224 + 112];
        visited[112 * 224 + 112] = 1;
        
        let sumX = 0, sumY = 0, dotCount = 0;
        let head = 0;
        
        const getBlueBalance = (r, g, b) => b - (r + g) / 2;
        const bgBlue = getBlueBalance(bgR, bgG, bgB);
        const fgBlue = getBlueBalance(fgR, fgG, fgB);
        
        while (head < queue.length) {
            const curr = queue[head++];
            const px = curr % 224, py = Math.floor(curr / 224);
            sumX += px; sumY += py; dotCount++;
            
            const neighbors = [curr + 1, curr - 1, curr + 224, curr - 224];
            for (const n of neighbors) {
                if (n >= 0 && n < 224 * 224 && !visited[n]) {
                    const r = pixels[n*4], g = pixels[n*4+1], b = pixels[n*4+2];
                    const pBlue = getBlueBalance(r, g, b);
                    const distFg = Math.sqrt((r-fgR)**2 + (g-fgG)**2 + (b-fgB)**2) + Math.abs(pBlue - fgBlue) * 1.5;
                    const distBg = Math.sqrt((r-bgR)**2 + (g-bgG)**2 + (b-bgB)**2) + Math.abs(pBlue - bgBlue) * 1.5;
                    
                    if (distFg * 1.15 < distBg) {
                        visited[n] = 1;
                        queue.push(n);
                    } else {
                        visited[n] = 2; // boundary
                    }
                }
            }
        }
        
        const templateArea = templateDots[0].area;
        const scale = Math.sqrt(dotCount / templateArea);
        
        if (dotCount > 10 && scale >= 0.4 && scale <= 2.5) {
            const centroidX = sumX / dotCount;
            const centroidY = sumY / dotCount;
            // Heavily weight the user's aim (112,112) as ground truth
            const cx = 112 * 0.8 + centroidX * 0.2;
            const cy = 112 * 0.8 + centroidY * 0.2;
            
            srcPts.push({x: templateDots[0].cx, y: templateDots[0].cy});
            dstPts.push({x: cx, y: cy});
            srcAreas.push(templateDots[0].area);
            dstAreas.push(dotCount);
            statusOverlay.classList.add("hidden");
            
        } else {
            // Fall back to user's aim
            srcPts.push({x: templateDots[0].cx, y: templateDots[0].cy});
            dstPts.push({x: 112, y: 112});
            srcAreas.push(templateDots[0].area);
            dstAreas.push(templateArea);
            statusOverlay.classList.add("hidden");
        }
    } else {
        // Multi-dot: Original Logic
        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        for (let y = 0; y < 224; y++) {
            for (let x = 0; x < 224; x++) {
                if (x < 5 || x >= 219 || y < 5 || y >= 219) { // 5px border
                    const i = (y*224 + x) * 4;
                    sumR += pixels[i]; sumG += pixels[i+1]; sumB += pixels[i+2]; count++;
                }
            }
        }
        const bgR = sumR/count, bgG = sumG/count, bgB = sumB/count;
        
        const getBlueBalance = (r, g, b) => b - (r + g) / 2;
        const bgBlue = getBlueBalance(bgR, bgG, bgB);
        
        for (const td of templateDots) {
            const tcx = Math.round(td.cx);
            const tcy = Math.round(td.cy);
            
            // Sample foreground at this dot's visual hole
            let sumFgR = 0, sumFgG = 0, sumFgB = 0, countFg = 0;
            for (let dy = -4; dy <= 4; dy++) {
                for (let dx = -4; dx <= 4; dx++) {
                    const nx = tcx + dx, ny = tcy + dy;
                    if (nx >= 0 && nx < 224 && ny >= 0 && ny < 224) {
                        const i = (ny*224 + nx) * 4;
                        sumFgR += pixels[i]; sumFgG += pixels[i+1]; sumFgB += pixels[i+2]; countFg++;
                    }
                }
            }
            if (countFg === 0) continue;
            
            const fgR = sumFgR/countFg, fgG = sumFgG/countFg, fgB = sumFgB/countFg;
            const fgBlue = getBlueBalance(fgR, fgG, fgB);
            
            const visited = new Uint8Array(224 * 224);
            const queue = [tcy * 224 + tcx];
            visited[tcy * 224 + tcx] = 1;
            
            let sumX = 0, sumY = 0, dotCount = 0;
            let head = 0;
            const searchRadius = 35; // Bound bleeding to a local box
            
            while (head < queue.length) {
                const curr = queue[head++];
                const px = curr % 224, py = Math.floor(curr / 224);
                sumX += px; sumY += py; dotCount++;
                
                const neighbors = [curr + 1, curr - 1, curr + 224, curr - 224];
                for (const n of neighbors) {
                    if (n >= 0 && n < 224 * 224 && !visited[n]) {
                        const nx = n % 224, ny = Math.floor(n / 224);
                        if (Math.abs(nx - tcx) > searchRadius || Math.abs(ny - tcy) > searchRadius) continue;
                        
                        const r = pixels[n*4], g = pixels[n*4+1], b = pixels[n*4+2];
                        const pBlue = getBlueBalance(r, g, b);
                        const distFg = Math.sqrt((r-fgR)**2 + (g-fgG)**2 + (b-fgB)**2) + Math.abs(pBlue - fgBlue) * 1.5;
                        const distBg = Math.sqrt((r-bgR)**2 + (g-bgG)**2 + (b-bgB)**2) + Math.abs(pBlue - bgBlue) * 1.5;
                        
                        if (distFg < distBg) {
                            visited[n] = 1;
                            queue.push(n);
                        }
                    }
                }
            }
            
            const scale = Math.sqrt(dotCount / td.area);
            if (dotCount > 5 && scale >= 0.4 && scale <= 2.5) {
                const cx = sumX / dotCount;
                const cy = sumY / dotCount;
                // Pure centroids, NO artificial weighting to visual holes
                
                srcPts.push({x: td.cx, y: td.cy});
                dstPts.push({x: cx, y: cy});
                srcAreas.push(td.area);
                dstAreas.push(dotCount);
            }
        }
        
        if (dstPts.length < templateDots.length) {
            drawOverlay();
            statusOverlay.classList.add("hidden");
            return;
        }
        statusOverlay.classList.add("hidden");
        
        // Lock circle sizes to average
        const avgArea = dstAreas.reduce((a, b) => a + b, 0) / dstAreas.length;
        for (let i = 0; i < dstAreas.length; i++) {
            dstAreas[i] = avgArea;
        }
    }
    
    // 4. Affine Alignment (Warp to perfectly align with templates)
    const M = estimateAffine(srcPts, dstPts, srcAreas, dstAreas);
    if (!M) return;
    
    // Draw static mask and dynamic green circles
    drawOverlay();
    
    const w = overlayCanvas.width, h = overlayCanvas.height;
    const boxSize = Math.min(w, h);
    const bx = (w - boxSize) / 2, by = (h - boxSize) / 2;
    
    // Draw green circles on centroids
    for (let i = 0; i < dstPts.length; i++) {
        const pt = dstPts[i];
        const drawX = (pt.x / 224) * boxSize + bx;
        const drawY = (pt.y / 224) * boxSize + by;
        const drawR = (Math.sqrt(dstAreas[i] / Math.PI) / 224) * boxSize;
        
        overlayCtx.beginPath();
        overlayCtx.arc(drawX, drawY, drawR, 0, 2 * Math.PI);
        
        overlayCtx.shadowColor = "rgba(99, 102, 241, 0.8)";
        overlayCtx.shadowBlur = 10;
        
        overlayCtx.strokeStyle = "rgba(99, 102, 241, 0.9)";
        overlayCtx.lineWidth = 2;
        overlayCtx.stroke();
        
        overlayCtx.shadowBlur = 0; // Reset
    }
    
    // 5. Build 80x120 Query Grid directly from imgData using M
    const queryGridData = new ImageData(GRID_W, GRID_H);
    for (let i = 0; i < GRID_W * GRID_H; i++) {
        queryGridData.data[i*4+3] = 255;
    }
    
    // Map each template dot to the correct base centroid index
    const tdSlotIndices = templateDots.map(td => {
        let minD = Infinity, minIdx = 0;
        for (let k = 0; k < BASE_CENTROIDS.length; k++) {
            const d = Math.hypot(td.cx - BASE_CENTROIDS[k].cx, td.cy - BASE_CENTROIDS[k].cy);
            if (d < minD) { minD = d; minIdx = k; }
        }
        return minIdx;
    });
    
    for (let i = 0; i < templateDots.length; i++) {
        const slot = GRID_SLOTS[tdSlotIndices[i]];
        const cx = Math.round(templateDots[i].cx);
        const cy = Math.round(templateDots[i].cy);
        
        for (let dy = -20; dy < 20; dy++) {
            for (let dx = -20; dx < 20; dx++) {
                const gx = slot.x + dx;
                const gy = slot.y + dy;
                const tx = cx + dx;
                const ty = cy + dy;
                
                const gridIdx = gy * GRID_W + gx;
                
                if (config.eroded80x120[gridIdx]) {
                    const sx = Math.round(M.a * tx + M.b * ty + M.tx);
                    const sy = Math.round(M.c * tx + M.d * ty + M.ty);
                    
                    if (sx >= 0 && sx < 224 && sy >= 0 && sy < 224) {
                        const srcIdx = (sy * 224 + sx) * 4;
                        queryGridData.data[gridIdx*4] = imgData.data[srcIdx];
                        queryGridData.data[gridIdx*4+1] = imgData.data[srcIdx+1];
                        queryGridData.data[gridIdx*4+2] = imgData.data[srcIdx+2];
                    }
                }
            }
        }
    }
    
    // 6. Extract pixels and apply edge fading for UI only
    const qR = [], qG = [], qB = [];
    
    // Collect pixels
    for (let y = 0; y < GRID_H; y++) {
        for (let x = 0; x < GRID_W; x++) {
            const i = y * GRID_W + x;
            if (config.eroded80x120[i]) {
                const origR = queryGridData.data[i*4];
                const origG = queryGridData.data[i*4+1];
                const origB = queryGridData.data[i*4+2];
                
                // Keep raw pixels for NCC so we don't break the dot product with the database
                qR.push(origR);
                qG.push(origG);
                qB.push(origB);
                
                // Determine edge proximity for visual fading on the UI canvas
                let maskSum = 0, maskCount = 0;
                for (let dy = -3; dy <= 3; dy++) {
                    for (let dx = -3; dx <= 3; dx++) {
                        const nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < GRID_W && ny >= 0 && ny < GRID_H) {
                            maskSum += config.eroded80x120[ny * GRID_W + nx] ? 1 : 0;
                            maskCount++;
                        }
                    }
                }
                const weight = maskSum / maskCount;
                
                // Visual fade to black at the edges
                if (enableEdgeFade) {
                    queryGridData.data[i*4] = origR * weight;
                    queryGridData.data[i*4+1] = origG * weight;
                    queryGridData.data[i*4+2] = origB * weight;
                } else {
                    queryGridData.data[i*4] = origR;
                    queryGridData.data[i*4+1] = origG;
                    queryGridData.data[i*4+2] = origB;
                }
            }
        }
    }
    
    const normalize = (channel) => {
        const mean = channel.reduce((a, b) => a + b, 0) / channel.length;
        const centered = channel.map(v => v - mean);
        const sumSq = centered.reduce((a, b) => a + b*b, 0);
        const norm = Math.sqrt(sumSq) || 1e-6;
        return centered.map(v => v / norm);
    };
    
    const qRNorm = normalize(qR), qGNorm = normalize(qG), qBNorm = normalize(qB);

    
    // 7. Compute True NCC (Dot Product of normalized vectors)
    const numJackets = jacketNames.length;
    const similarityResults = [];
    const pixelCount = config.pixelCount;
    const dbNorms = config.dbNorms;
    const dbMeanHsv = config.dbMeanHsv;
    
    // Calculate mean HSV for the query
    const meanQR = qR.reduce((a, b) => a + b, 0) / qR.length;
    const meanQG = qG.reduce((a, b) => a + b, 0) / qG.length;
    const meanQB = qB.reduce((a, b) => a + b, 0) / qB.length;
    const qHsv = rgbToHsv(meanQR, meanQG, meanQB);
    
    const stride = 2;
    for (let j = 0; j < numJackets; j++) {
        const dbOffset = j * pixelCount * 3;
        let sumProdR = 0, sumProdG = 0, sumProdB = 0;
        
        for (let idx = 0; idx < pixelCount; idx += stride) {
            sumProdR += qRNorm[idx] * dbNorms[dbOffset + idx*3];
            sumProdG += qGNorm[idx] * dbNorms[dbOffset + idx*3 + 1];
            sumProdB += qBNorm[idx] * dbNorms[dbOffset + idx*3 + 2];
        }
        
        let score = ((sumProdR + sumProdG + sumProdB) / 3.0) * stride;
        
        if (enableColorFilter) {
            const rHsv0 = dbMeanHsv[j * 2];
            const rHsv1 = dbMeanHsv[j * 2 + 1];
            
            const hueDiff = Math.abs(qHsv[0] - rHsv0);
            const hueDist = Math.min(hueDiff, 360 - hueDiff) / 180.0;
            const satDist = Math.abs(qHsv[1] - rHsv1);
            
            score -= (hueDist + satDist) * 0.2;
        }
        
        similarityResults.push({ name: jacketNames[j], index: j, score });
    }
    
    similarityResults.sort((a, b) => b.score - a.score);
    let topCurrent = similarityResults.slice(0, 5);
    
    let updated = false;
    for (const match of topCurrent) {
        if (match.score < 0.35) continue; // Filter bad matches
        if (bannedEntries.has(match.name)) continue;
        if (!songMetadata[match.name]) continue; // Only show active songs
        const existing = sessionTopMatches.find(m => m.name === match.name);
        if (existing) {
            const oldBoost = Math.min((existing.hitCount || 1) - 1, 15) * 0.015;
            existing.hitCount = (existing.hitCount || 1) + 1;
            const newBoost = Math.min(existing.hitCount - 1, 15) * 0.015;
            
            if (match.score > existing.score) {
                existing.score = match.score;
                existing.queryImageData = new ImageData(new Uint8ClampedArray(queryGridData.data), GRID_W, GRID_H);
                updated = true;
            }
            if (newBoost > oldBoost) {
                updated = true;
            }
        } else {
            match.queryImageData = new ImageData(new Uint8ClampedArray(queryGridData.data), GRID_W, GRID_H);
            match.hitCount = 1;
            sessionTopMatches.push(match);
            updated = true;
        }
    }
    
    if (updated || (!isLiveFeed && sessionTopMatches.length === 0)) {
        sessionTopMatches.forEach(m => {
            m.displayScore = m.score + Math.min((m.hitCount || 1) - 1, 15) * 0.015;
        });
        sessionTopMatches.sort((a, b) => b.displayScore - a.displayScore);
        sessionTopMatches = sessionTopMatches.slice(0, 6);
        renderMatches(sessionTopMatches.slice(0, 3));
    }
}

function renderMatches(topMatches) {
    matchesList.innerHTML = "";
    
    if (topMatches.length === 0) {
        matchesList.innerHTML = `<div class="empty-state">
            <div class="empty-icon">🧥</div>
            <p>No high-confidence matches yet.</p>
        </div>`;
        return;
    }
    
    topMatches.forEach((match, index) => {
        const scoreToUse = match.displayScore || match.score;
        const scorePercentage = Math.max(0, Math.min(100, scoreToUse * 100));
        const baseName = match.name.substring(0, match.name.lastIndexOf('.'));
        const thumbUrl = `thumbnails/${encodeURIComponent(baseName)}.webp`;
        
        const meta = songMetadata[match.name] || { title: match.name, artist: "Unknown Artist" };
        
        const matchItem = document.createElement("div");
        matchItem.className = `match-item ${index === 0 ? "rank-1" : ""}`;
        matchItem.style.flexDirection = "column";
        
        const glowOpacity = scorePercentage / 100;
        const r = Math.round(99 + (16 - 99) * glowOpacity);
        const g = Math.round(102 + (185 - 102) * glowOpacity);
        const b = Math.round(241 + (129 - 241) * glowOpacity);
        matchItem.style.boxShadow = `inset 0 0 20px rgba(${r}, ${g}, ${b}, ${glowOpacity * 0.5})`;
        
        matchItem.innerHTML = `
            <div style="display: flex; width: 100%; align-items: flex-start; margin-bottom: 6px; position: relative;">
                <div style="flex: 1; min-width: 0; text-align: center; padding: 0 20px;">
                    <div class="match-name" title="${meta.title}" style="width: 100%; font-size: 1.1rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${meta.title}</div>
                    <div class="match-artist" style="width: 100%; color: var(--text-muted); font-size: 0.75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${meta.artist}</div>
                </div>
                <button class="ban-btn" data-name="${match.name}" style="position: absolute; right: 0; top: 2px; background: none; border: none; color: #ef4444; cursor: pointer; padding: 0; display: flex; align-items: center; justify-content: center;" title="Temporarily ban this match">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>
            
            <div class="match-content" style="display: flex; align-items: center; width: 100%; gap: 0.75rem; justify-content: center;">
                <img src="${thumbUrl}" alt="Jacket" style="width: 60px; height: 60px; object-fit: cover; border-radius: var(--radius-sm); display: block; flex-shrink: 1; min-width: 40px;">
                <div class="dots-container" style="display: flex; gap: 4px; flex-shrink: 1; background: rgba(0,0,0,0.2); border-radius: var(--radius-sm); padding: 4px; min-width: 0;">
                </div>
            </div>

            <div style="display: flex; align-items: center; width: 100%; gap: 8px; margin-top: 6px;">
                <div class="score-text" style="font-size: 0.65rem; color: var(--text-muted); font-family: monospace; min-width: auto; text-align: left;">
                    ${scoreToUse.toFixed(4)} 
                    ${match.hitCount > 1 ? `<span style="color: var(--primary); margin-left: 2px;">(+${Math.min(match.hitCount - 1, 15) * 1.5}%)</span>` : ''}
                </div>
                <div class="score-bar-bg" style="flex: 1; height: 2px; background: rgba(255,255,255,0.1); border-radius: 1px;">
                    <div class="score-bar-fill" style="width: ${scorePercentage}%; height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent)); border-radius: 1px;"></div>
                </div>
            </div>
        `;
        
        const dotsContainer = matchItem.querySelector('.dots-container');
        const config = maskConfigs[activeMaskName];
        
        if (config && config.slotIndices) {
            const rCtxTemp = document.createElement("canvas").getContext("2d");
            rCtxTemp.canvas.width = 80; rCtxTemp.canvas.height = 120;
            const cols = Math.floor(spriteCanvas.width / 80);
            const sx = (match.index % cols) * 80;
            const sy = Math.floor(match.index / cols) * 120;
            rCtxTemp.drawImage(spriteCanvas, sx, sy, 80, 120, 0, 0, 80, 120);
            
            const refData = rCtxTemp.getImageData(0, 0, 80, 120);
            for (let i = 0; i < 80 * 120; i++) {
                if (!config.eroded80x120[i]) {
                    refData.data[i*4] = refData.data[i*4+1] = refData.data[i*4+2] = 0;
                }
            }
            rCtxTemp.putImageData(refData, 0, 0);

            config.slotIndices.forEach(slotIdx => {
                const slot = GRID_SLOTS[slotIdx];
                const dotPair = document.createElement("div");
                dotPair.style.display = "flex";
                dotPair.style.flexDirection = "column";
                dotPair.style.gap = "2px";
                dotPair.style.flex = "1 1 0";
                dotPair.style.minWidth = "20px";
                dotPair.style.maxWidth = "40px";
                
                const qCanvas = document.createElement("canvas");
                qCanvas.width = 40; qCanvas.height = 40;
                qCanvas.style.width = "100%";
                qCanvas.style.height = "auto";
                qCanvas.style.aspectRatio = "1/1";
                qCanvas.style.backgroundColor = "#000";
                qCanvas.style.imageRendering = "pixelated";
                qCanvas.title = "Query";
                
                if (match.queryImageData) {
                    const qCtx = qCanvas.getContext("2d");
                    qCtx.putImageData(match.queryImageData, -slot.x + 20, -slot.y + 20);
                }
                
                const rCanvas = document.createElement("canvas");
                rCanvas.width = 40; rCanvas.height = 40;
                rCanvas.style.width = "100%";
                rCanvas.style.height = "auto";
                rCanvas.style.aspectRatio = "1/1";
                rCanvas.style.backgroundColor = "#000";
                rCanvas.style.imageRendering = "pixelated";
                rCanvas.title = "Reference";
                
                const rCtx = rCanvas.getContext("2d");
                rCtx.drawImage(rCtxTemp.canvas, slot.x - 20, slot.y - 20, 40, 40, 0, 0, 40, 40);
                
                dotPair.appendChild(qCanvas);
                dotPair.appendChild(rCanvas);
                dotsContainer.appendChild(dotPair);
            });
        }
        
        matchesList.appendChild(matchItem);
    });
    
    // Add event listeners for ban buttons
    const banBtns = matchesList.querySelectorAll('.ban-btn');
    banBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const name = e.currentTarget.getAttribute('data-name');
            bannedEntries.add(name);
            sessionTopMatches = sessionTopMatches.filter(m => m.name !== name);
            renderMatches(sessionTopMatches.slice(0, 3));
        });
    });
}

// 5. Test Image Upload logic removed

const maskButtons = document.querySelectorAll("#mask-buttons button");

maskButtons.forEach(btn => {
    btn.addEventListener("click", (e) => {
        maskButtons.forEach(b => {
            b.classList.remove("primary-btn");
            b.classList.add("outline-btn");
        });
        const targetBtn = e.currentTarget;
        targetBtn.classList.remove("outline-btn");
        targetBtn.classList.add("primary-btn");
        activeMaskName = targetBtn.getAttribute("data-mask");
        drawOverlay();
    });
});

window.addEventListener("resize", resizeCanvas);

async function init() {
    await initDatabase();
    await setupCameras();
    
    // Run at 8 FPS
    processingInterval = setInterval(processQueryFrame, 125);
}

init();
