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
let activeMaskName = "5-dot";
let isLiveFeed = true;
let sessionTopMatches = [];
let isFrozen = false;
let videoStream = null;
let processingInterval = null;
let bannedEntries = new Set();
let enableEdgeFade = false;

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
refreshCameraBtn.addEventListener("click", () => {
    if (currentDeviceId) startCamera(currentDeviceId);
    else setupCameras();
});

freezeBtn.addEventListener("click", () => {
    isFrozen = !isFrozen;
    freezeBtn.textContent = isFrozen ? "Unfreeze" : "Freeze";
    freezeBtn.className = isFrozen ? "btn primary-btn btn-sm" : "btn outline-btn btn-sm";
});

clearMatchesBtn.addEventListener("click", () => {
    bannedEntries.clear();
    sessionTopMatches = [];
    matchesList.innerHTML = `<div class="empty-state">
        <div class="empty-icon">⏳</div>
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

// Offscreen canvas for fast video frame extraction
const offscreenCanvas = document.createElement("canvas");
offscreenCanvas.width = 224;
offscreenCanvas.height = 224;
const offCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

// Math Utilities
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
            scale = Math.sqrt(dstAreas[0] / srcAreas[0]);
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
        
        loadingText.textContent = "Downloading sprite sheet (~3.5 MB)...";
        const spriteImg = new Image();
        spriteImg.src = "jackets_sprite.webp";
        await new Promise((resolve, reject) => { spriteImg.onload = resolve; spriteImg.onerror = reject; });
        
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
        
        maskConfigs[name] = { templateDots, slotIndices: tdSlotIndices, eroded80x120, pixelCount, dbNorms, punchHoleBitmap };
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
        
        if (capabilities.exposureMode && capabilities.exposureCompensation) {
            exposureContainer.style.display = "flex";
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

    overlayCtx.strokeStyle = "rgba(99, 102, 241, 0.5)";
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(bx, by, boxSize, boxSize);
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

function showStatus(msg) {
    statusOverlay.textContent = msg;
    statusOverlay.classList.remove("hidden");
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
        const sampleBg = (cx, cy) => {
            for (let dy=0; dy<10; dy++) {
                for (let dx=0; dx<10; dx++) {
                    const i = ((cy+dy)*224 + (cx+dx)) * 4;
                    sumBgR += pixels[i]; sumBgG += pixels[i+1]; sumBgB += pixels[i+2]; countBg++;
                }
            }
        };
        sampleBg(0, 0); sampleBg(214, 0); sampleBg(0, 214); sampleBg(214, 214);
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
        
        while (head < queue.length) {
            const curr = queue[head++];
            const px = curr % 224, py = Math.floor(curr / 224);
            sumX += px; sumY += py; dotCount++;
            
            const neighbors = [curr + 1, curr - 1, curr + 224, curr - 224];
            for (const n of neighbors) {
                if (n >= 0 && n < 224 * 224 && !visited[n]) {
                    const r = pixels[n*4], g = pixels[n*4+1], b = pixels[n*4+2];
                    const distFg = Math.sqrt((r-fgR)**2 + (g-fgG)**2 + (b-fgB)**2);
                    const distBg = Math.sqrt((r-bgR)**2 + (g-bgG)**2 + (b-bgB)**2);
                    
                    if (distFg < distBg) {
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
            const cx = sumX / dotCount;
            const cy = sumY / dotCount;
            srcPts.push({x: templateDots[0].cx, y: templateDots[0].cy});
            dstPts.push({x: cx, y: cy});
            srcAreas.push(templateDots[0].area);
            dstAreas.push(dotCount);
            statusOverlay.classList.add("hidden");
            
            // Draw real-time outline
            drawOverlay();
            const w = overlayCanvas.width, h = overlayCanvas.height;
            const drawBoxSize = Math.min(w, h);
            const bx = (w - drawBoxSize) / 2, by = (h - drawBoxSize) / 2;
            const drawX = (cx / 224) * drawBoxSize + bx;
            const drawY = (cy / 224) * drawBoxSize + by;
            const drawR = (Math.sqrt(dotCount / Math.PI) / 224) * drawBoxSize;
            
            overlayCtx.beginPath();
            overlayCtx.arc(drawX, drawY, drawR, 0, 2 * Math.PI);
            overlayCtx.strokeStyle = "rgba(0, 255, 0, 0.8)";
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();
            
        } else {
            drawOverlay();
            showStatus(`Align the dot...`);
            return;
        }
    } else {
        // Multi-dot: Original Logic
        let sumR = 0, sumG = 0, sumB = 0, count = 0;
        const sample = (cx, cy) => {
            for (let dy=0; dy<10; dy++) {
                for (let dx=0; dx<10; dx++) {
                    const i = ((cy+dy)*224 + (cx+dx)) * 4;
                    sumR += pixels[i]; sumG += pixels[i+1]; sumB += pixels[i+2]; count++;
                }
            }
        };
        sample(0, 0); sample(214, 0); sample(0, 214); sample(214, 214);
        const bgR = sumR/count, bgG = sumG/count, bgB = sumB/count;
        
        const binaryMask = new Uint8Array(224 * 224);
        for (let i = 0; i < 224 * 224; i++) {
            const r = pixels[i*4], g = pixels[i*4+1], b = pixels[i*4+2];
            const dist = Math.sqrt((r-bgR)**2 + (g-bgG)**2 + (b-bgB)**2);
            binaryMask[i] = dist > 70 ? 1 : 0;
        }
        
        let queryDots = getDotProperties(binaryMask, 224, 224);
        
        // Filter by size: 40% to 250% scale relative to average template dot size
        const avgTemplateArea = templateDots.reduce((sum, td) => sum + td.area, 0) / templateDots.length;
        queryDots = queryDots.filter(qd => {
            const scale = Math.sqrt(qd.area / avgTemplateArea);
            return scale >= 0.4 && scale <= 2.5;
        });
        
        if (queryDots.length < templateDots.length) {
            drawOverlay();
            showStatus(`Align all ${templateDots.length} dots...`);
            return;
        }
        statusOverlay.classList.add("hidden");
        
        queryDots.sort((a, b) => b.area - a.area);
        const topQueryDots = queryDots.slice(0, Math.max(templateDots.length, 5));
        
        for (const td of templateDots) {
            let minDist = Infinity, closestQ = null;
            for (const qd of topQueryDots) {
                const d = Math.hypot(qd.cx - td.cx, qd.cy - td.cy);
                if (d < minDist) { minDist = d; closestQ = qd; }
            }
            if (closestQ) {
                srcPts.push({x: td.cx, y: td.cy});
                dstPts.push({x: closestQ.cx, y: closestQ.cy});
                srcAreas.push(td.area);
                dstAreas.push(closestQ.area);
            }
        }
        
        // Draw real-time outline for multi-dot
        drawOverlay();
        const w = overlayCanvas.width, h = overlayCanvas.height;
        const drawBoxSize = Math.min(w, h);
        const bx = (w - drawBoxSize) / 2, by = (h - drawBoxSize) / 2;
        
        for (let i = 0; i < dstPts.length; i++) {
            const pt = dstPts[i];
            const drawX = (pt.x / 224) * drawBoxSize + bx;
            const drawY = (pt.y / 224) * drawBoxSize + by;
            const drawR = (Math.sqrt(dstAreas[i] / Math.PI) / 224) * drawBoxSize;
            
            overlayCtx.beginPath();
            overlayCtx.arc(drawX, drawY, drawR, 0, 2 * Math.PI);
            overlayCtx.strokeStyle = "rgba(0, 255, 0, 0.8)";
            overlayCtx.lineWidth = 2;
            overlayCtx.stroke();
        }
    }
    
    // 4. Affine Alignment (Warp to perfectly align with templates)
    const M = estimateAffine(srcPts, dstPts, srcAreas, dstAreas);
    if (!M) return;
    
    // warpAffineNearest maps Template -> Query to sample pixels properly
    const warpedData = warpAffineNearest(imgData, 224, 224, M);
    
    // 5. Build 80x120 Query Grid
    const qCanvas = document.createElement("canvas");
    qCanvas.width = GRID_W; qCanvas.height = GRID_H;
    const dqCtx = qCanvas.getContext("2d", { willReadFrequently: true });
    
    dqCtx.fillStyle = "#000"; dqCtx.fillRect(0, 0, GRID_W, GRID_H);
    
    // Temporarily write warped data back to offscreen to drawImage crop
    offCtx.putImageData(warpedData, 0, 0);
    
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
        const td = templateDots[i], slot = GRID_SLOTS[tdSlotIndices[i]];
        const cx = Math.round(td.cx), cy = Math.round(td.cy);
        dqCtx.drawImage(
            offscreenCanvas,
            cx - 20, cy - 20, 40, 40,
            slot.x - 20, slot.y - 20, 40, 40
        );
    }
    
    // Apply visual black mask for debug canvas
    dqCtx.save();
    const queryGridData = dqCtx.getImageData(0, 0, GRID_W, GRID_H);
    for (let i = 0; i < GRID_W * GRID_H; i++) {
        if (!config.eroded80x120[i]) {
            queryGridData.data[i*4] = 0;
            queryGridData.data[i*4+1] = 0;
            queryGridData.data[i*4+2] = 0;
        }
    }
    dqCtx.putImageData(queryGridData, 0, 0);
    
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
    dqCtx.putImageData(queryGridData, 0, 0);
    
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
    
    for (let j = 0; j < numJackets; j++) {
        const dbOffset = j * pixelCount * 3;
        let sumProdR = 0, sumProdG = 0, sumProdB = 0;
        
        for (let idx = 0; idx < pixelCount; idx++) {
            sumProdR += qRNorm[idx] * dbNorms[dbOffset + idx*3];
            sumProdG += qGNorm[idx] * dbNorms[dbOffset + idx*3 + 1];
            sumProdB += qBNorm[idx] * dbNorms[dbOffset + idx*3 + 2];
        }
        
        const score = (sumProdR + sumProdG + sumProdB) / 3.0;
        similarityResults.push({ name: jacketNames[j], index: j, score });
    }
    
    similarityResults.sort((a, b) => b.score - a.score);
    const topCurrent = similarityResults.slice(0, 5);
    
    let updated = false;
    for (const match of topCurrent) {
        if (match.score < 0.45) continue; // Filter bad matches
        if (bannedEntries.has(match.name)) continue;
        if (!songMetadata[match.name]) continue; // Only show active songs
        const existing = sessionTopMatches.find(m => m.name === match.name);
        if (existing) {
            if (match.score > existing.score) {
                existing.score = match.score;
                existing.queryImageData = new ImageData(new Uint8ClampedArray(queryGridData.data), GRID_W, GRID_H);
                updated = true;
            }
        } else {
            match.queryImageData = new ImageData(new Uint8ClampedArray(queryGridData.data), GRID_W, GRID_H);
            sessionTopMatches.push(match);
            updated = true;
        }
    }
    
    if (updated || (!isLiveFeed && sessionTopMatches.length === 0)) {
        sessionTopMatches.sort((a, b) => b.score - a.score);
        sessionTopMatches = sessionTopMatches.slice(0, 3);
        renderMatches(sessionTopMatches);
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
        const scorePercentage = Math.max(0, Math.min(100, match.score * 100));
        const baseName = match.name.substring(0, match.name.lastIndexOf('.'));
        const thumbUrl = `thumbnails/${encodeURIComponent(baseName)}.webp`;
        
        const meta = songMetadata[match.name] || { title: match.name, artist: "Unknown Artist" };
        
        const matchItem = document.createElement("div");
        matchItem.className = `match-item ${index === 0 ? "rank-1" : ""}`;
        matchItem.style.flexDirection = "column";
        
        matchItem.innerHTML = `
            <div style="display: flex; width: 100%; gap: 0.75rem; align-items: flex-start;">
                <!-- Thumbnail -->
                <img src="${thumbUrl}" alt="Jacket" style="width: 60px; height: 60px; object-fit: cover; border-radius: var(--radius-sm); display: block; flex-shrink: 0;">
                
                <!-- Title and Artist -->
                <div style="flex: 1; min-width: 0;">
                    <div class="match-name" title="${meta.title}" style="width: 100%; font-size: 1.1rem; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${meta.title}</div>
                    <div class="match-artist" style="width: 100%; color: var(--text-muted); font-size: 0.9rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${meta.artist}</div>
                </div>
                
                <!-- X button and Score -->
                <div style="display: flex; flex-direction: column; align-items: flex-end; flex-shrink: 0; min-width: 80px;">
                    <button class="ban-btn" data-name="${match.name}" style="background: none; border: none; color: #ef4444; font-size: 1.2rem; cursor: pointer; padding: 0; line-height: 1;" title="Temporarily ban this match">✖</button>
                    <div class="score-container" style="width: 100%; text-align: right; margin-top: 4px;">
                        <div class="score-text" style="font-weight: bold; margin-bottom: 4px;">${match.score.toFixed(4)}</div>
                        <div class="score-bar-bg" style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px;">
                            <div class="score-bar-fill" style="width: ${scorePercentage}%; height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent)); border-radius: 3px;"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="match-content" style="display: flex; align-items: center; width: 100%; margin-top: 0.75rem; gap: 0.75rem; overflow-x: auto; padding-bottom: 4px;">
                <div class="dots-container" style="display: flex; gap: 6px; flex-shrink: 0; background: rgba(0,0,0,0.2); border-radius: var(--radius-sm); padding: 4px;">
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
                
                const qCanvas = document.createElement("canvas");
                qCanvas.width = 40; qCanvas.height = 40;
                qCanvas.style.width = "40px"; qCanvas.style.height = "40px";
                qCanvas.style.backgroundColor = "#000";
                qCanvas.style.imageRendering = "pixelated";
                qCanvas.title = "Query";
                
                if (match.queryImageData) {
                    const qCtx = qCanvas.getContext("2d");
                    qCtx.putImageData(match.queryImageData, -slot.x + 20, -slot.y + 20);
                }
                
                const rCanvas = document.createElement("canvas");
                rCanvas.width = 40; rCanvas.height = 40;
                rCanvas.style.width = "40px"; rCanvas.style.height = "40px";
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
            const name = e.target.getAttribute('data-name');
            bannedEntries.add(name);
            sessionTopMatches = sessionTopMatches.filter(m => m.name !== name);
            renderMatches(sessionTopMatches);
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
        e.target.classList.remove("outline-btn");
        e.target.classList.add("primary-btn");
        activeMaskName = e.target.getAttribute("data-mask");
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
