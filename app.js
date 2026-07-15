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

// Global State
let jacketNames = [];
let databaseLoaded = false;
let spriteCanvas = null; // Holds the 2560x8280 sprite
let maskConfigs = {}; // Stores configuration and pre-normalized DB per mask
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
const matchCountBadge = document.getElementById("match-count-badge");
const statusOverlay = document.getElementById("status-overlay");

const debugQueryCanvas = document.getElementById("debug-query-canvas");
const dqCtx = debugQueryCanvas.getContext("2d");
const debugRefCanvas = document.getElementById("debug-ref-canvas");
const drCtx = debugRefCanvas.getContext("2d");

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

// Estimates a full 6-DoF Affine transform mapping srcPts -> dstPts using linear least squares
function estimateAffine(srcPts, dstPts) {
    const N = srcPts.length;
    if (N === 0) return null;
    if (N < 3) {
        // Translation only fallback
        const tx = dstPts[0].x - srcPts[0].x;
        const ty = dstPts[0].y - srcPts[0].y;
        return { a: 1, b: 0, tx, c: 0, d: 1, ty };
    }
    
    let sumXX = 0, sumXY = 0, sumX = 0, sumYY = 0, sumY = 0;
    for (let i = 0; i < N; i++) {
        const x = srcPts[i].x, y = srcPts[i].y;
        sumXX += x*x; sumXY += x*y; sumX += x; sumYY += y*y; sumY += y;
    }
    const PTP = [
        [sumXX, sumXY, sumX],
        [sumXY, sumYY, sumY],
        [sumX,  sumY,  N   ]
    ];
    const PTP_inv = invert3x3(PTP);
    if (!PTP_inv) return null;
    
    let PTQx = [0, 0, 0], PTQy = [0, 0, 0];
    for (let i = 0; i < N; i++) {
        const x = srcPts[i].x, y = srcPts[i].y, qx = dstPts[i].x, qy = dstPts[i].y;
        PTQx[0] += x*qx; PTQx[1] += y*qx; PTQx[2] += qx;
        PTQy[0] += x*qy; PTQy[1] += y*qy; PTQy[2] += qy;
    }
    
    const a = PTP_inv[0][0]*PTQx[0] + PTP_inv[0][1]*PTQx[1] + PTP_inv[0][2]*PTQx[2];
    const b = PTP_inv[1][0]*PTQx[0] + PTP_inv[1][1]*PTQx[1] + PTP_inv[1][2]*PTQx[2];
    const tx = PTP_inv[2][0]*PTQx[0] + PTP_inv[2][1]*PTQx[1] + PTP_inv[2][2]*PTQx[2];
    
    const c = PTP_inv[0][0]*PTQy[0] + PTP_inv[0][1]*PTQy[1] + PTP_inv[0][2]*PTQy[2];
    const d = PTP_inv[1][0]*PTQy[0] + PTP_inv[1][1]*PTQy[1] + PTP_inv[1][2]*PTQy[2];
    const ty = PTP_inv[2][0]*PTQy[0] + PTP_inv[2][1]*PTQy[1] + PTP_inv[2][2]*PTQy[2];
    
    return { a, b, tx, c, d, ty };
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
        matchCountBadge.textContent = `${jacketNames.length} DB`;
        
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
        
        for (let i = 0; i < templateDots.length; i++) {
            const td = templateDots[i];
            const slot = GRID_SLOTS[i];
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
        
        maskConfigs[name] = { templateDots, eroded80x120, pixelCount, dbNorms };
    }
}

// 2. Camera Access
async function setupCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");
        
        cameraSelect.innerHTML = "";
        if (videoDevices.length === 0) {
            cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            return;
        }
        
        videoDevices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            if (device.label.toLowerCase().includes("back") || device.label.toLowerCase().includes("rear")) {
                option.selected = true;
            }
            cameraSelect.appendChild(option);
        });
        
        await startCamera(cameraSelect.value);
        cameraSelect.onchange = async () => await startCamera(cameraSelect.value);
    } catch (err) {
        console.error("Camera detection error:", err);
    }
}

async function startCamera(deviceId) {
    if (videoStream) videoStream.getTracks().forEach(t => t.stop());
    const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId }, width: 480, height: 480 } : { facingMode: "environment", width: 480, height: 480 }
    };
    try {
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = videoStream;
        video.onloadedmetadata = () => { video.play(); resizeCanvas(); };
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
    
    const boxSize = Math.min(w, h) * 0.85;
    const bx = (w - boxSize) / 2, by = (h - boxSize) / 2;
    
    overlayCtx.strokeStyle = "rgba(99, 102, 241, 0.5)";
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(bx, by, boxSize, boxSize);
    
    // Draw alignment circle overlays based on active mask
    if (!maskConfigs[activeMaskName]) return;
    const dots = maskConfigs[activeMaskName].templateDots;
    
    overlayCtx.fillStyle = "rgba(16, 185, 129, 0.3)";
    overlayCtx.strokeStyle = "rgba(16, 185, 129, 0.8)";
    overlayCtx.lineWidth = 1.5;
    
    for (const dot of dots) {
        const dx = bx + (dot.cx / 224) * boxSize;
        const dy = by + (dot.cy / 224) * boxSize;
        const r = (19 / 224) * boxSize;
        overlayCtx.beginPath();
        overlayCtx.arc(dx, dy, r, 0, 2 * Math.PI);
        overlayCtx.fill(); overlayCtx.stroke();
    }
}

// 4. Query Processing & NCC Match
function processQueryFrame() {
    if (!databaseLoaded) return;
    
    const vw = video.videoWidth, vh = video.videoHeight;
    if (vw === 0 || vh === 0) return;
    
    const boxSize = Math.min(vw, vh) * 0.85;
    const sx = (vw - boxSize) / 2, sy = (vh - boxSize) / 2;
    offCtx.drawImage(video, sx, sy, boxSize, boxSize, 0, 0, 224, 224);
    
    runMatchingPipeline();
}

function showStatus(msg) {
    statusOverlay.textContent = msg;
    statusOverlay.classList.remove("hidden");
    // clear matches
    matchesList.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>${msg}</p></div>`;
}

function runMatchingPipeline() {
    const config = maskConfigs[activeMaskName];
    const templateDots = config.templateDots;
    
    // 1. Dynamic Background Sampling
    const imgData = offCtx.getImageData(0, 0, 224, 224);
    const pixels = imgData.data;
    
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
    
    // 2. Segment dots (Dist > 70)
    const binaryMask = new Uint8Array(224 * 224);
    for (let i = 0; i < 224 * 224; i++) {
        const r = pixels[i*4], g = pixels[i*4+1], b = pixels[i*4+2];
        const dist = Math.sqrt((r-bgR)**2 + (g-bgG)**2 + (b-bgB)**2);
        binaryMask[i] = dist > 70 ? 1 : 0;
    }
    
    // 3. Find Query Centroids
    const queryDots = getDotProperties(binaryMask, 224, 224);
    if (queryDots.length < templateDots.length) {
        showStatus(`Align all ${templateDots.length} dots...`);
        return;
    }
    statusOverlay.classList.add("hidden");
    
    // Take largest N dots and match to template using NN
    queryDots.sort((a, b) => b.area - a.area);
    const topQueryDots = queryDots.slice(0, templateDots.length);
    
    const srcPts = [], dstPts = []; // We map Template (src) -> Query (dst)
    for (const td of templateDots) {
        let minDist = Infinity, closestQ = null;
        for (const qd of topQueryDots) {
            const d = Math.hypot(qd.cx - td.cx, qd.cy - td.cy);
            if (d < minDist) { minDist = d; closestQ = qd; }
        }
        srcPts.push({x: td.cx, y: td.cy});
        dstPts.push({x: closestQ.cx, y: closestQ.cy});
    }
    
    // 4. Affine Alignment (Warp to perfectly align with templates)
    const M = estimateAffine(srcPts, dstPts);
    if (!M) return;
    
    // warpAffineNearest maps Template -> Query to sample pixels properly
    const warpedData = warpAffineNearest(imgData, 224, 224, M);
    
    // 5. Build 80x120 Query Grid
    dqCtx.fillStyle = "#000"; dqCtx.fillRect(0, 0, GRID_W, GRID_H);
    
    // Temporarily write warped data back to offscreen to drawImage crop
    offCtx.putImageData(warpedData, 0, 0);
    
    for (let i = 0; i < templateDots.length; i++) {
        const td = templateDots[i], slot = GRID_SLOTS[i];
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
    
    // 6. Extract pixels and normalize query
    const qR = [], qG = [], qB = [];
    for (let i = 0; i < GRID_W * GRID_H; i++) {
        if (config.eroded80x120[i]) {
            qR.push(queryGridData.data[i*4]);
            qG.push(queryGridData.data[i*4+1]);
            qB.push(queryGridData.data[i*4+2]);
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
    renderMatches(similarityResults.slice(0, 5));
}

function renderMatches(topMatches) {
    matchesList.innerHTML = "";
    
    topMatches.forEach((match, index) => {
        const rankClass = index === 0 ? "rank-1" : "";
        const scorePercentage = Math.max(0, Math.min(100, match.score * 100));
        
        const matchItem = document.createElement("div");
        matchItem.className = `match-item ${rankClass}`;
        
        if (index === 0) drawRefGridToCanvas(match.index);
        
        const thumbUrl = `jackets/${encodeURIComponent(match.name)}`;
        
        matchItem.innerHTML = `
            <div class="match-rank">${index + 1}</div>
            <img class="match-thumbnail" src="${thumbUrl}" alt="Jacket">
            <div class="match-info">
                <div class="match-name" title="${match.name}">${match.name}</div>
                <div class="score-container">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: ${scorePercentage}%"></div>
                    </div>
                    <div class="score-text">${match.score.toFixed(4)}</div>
                </div>
            </div>
        `;
        matchesList.appendChild(matchItem);
    });
}

function drawRefGridToCanvas(jacketIdx) {
    if (!spriteCanvas) return;
    const col = jacketIdx % SPRITE_COLS;
    const row = Math.floor(jacketIdx / SPRITE_COLS);
    
    drCtx.fillStyle = "#000"; drCtx.fillRect(0, 0, GRID_W, GRID_H);
    drCtx.drawImage(spriteCanvas, col * GRID_W, row * GRID_H, GRID_W, GRID_H, 0, 0, GRID_W, GRID_H);
    
    // Apply visual black mask
    const config = maskConfigs[activeMaskName];
    if (config) {
        const refGridData = drCtx.getImageData(0, 0, GRID_W, GRID_H);
        for (let i = 0; i < GRID_W * GRID_H; i++) {
            if (!config.eroded80x120[i]) {
                refGridData.data[i*4] = 0; refGridData.data[i*4+1] = 0; refGridData.data[i*4+2] = 0;
            }
        }
        drCtx.putImageData(refGridData, 0, 0);
    }
}

// 5. Test Image Upload
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            offCtx.clearRect(0, 0, 224, 224);
            offCtx.drawImage(img, 0, 0, img.width, img.height, 0, 0, 224, 224);
            runMatchingPipeline(); // Trigger once on upload
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

// Event Listeners & Initialize
maskSelect.addEventListener("change", (e) => {
    activeMaskName = e.target.value;
    drawOverlay();
    showStatus("Waiting for query feed...");
});

fileInput.addEventListener("change", handleFileUpload);
window.addEventListener("resize", resizeCanvas);

async function init() {
    await initDatabase();
    await setupCameras();
    // Run at 8 FPS
    processingInterval = setInterval(processQueryFrame, 125);
}

init();
