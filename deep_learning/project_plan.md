# Deep Learning / Embedding Image Retrieval (ONNX & Vector Search)

## Objective
Build a serverless web app using ONNX Runtime Web / Transformers.js to perform client-side vector similarity search on embeddings or keypoint matching via SuperPoint+LightGlue.

## Roadmap / Tasks

### Phase 1: Embedding / Model Prep (Python)
- [ ] Write Python script to run image embedding generation (e.g. using lightweight MobileNetV3 or custom CNN).
- [ ] Alternatively, prepare standard ONNX models for extraction (e.g., SuperPoint).
- [ ] Pre-calculate embeddings for the database of reference images.
- [ ] Save database to `embeddings.json` (or binary float array).

### Phase 2: Web App Frontend
- [ ] Create UI with camera capture.
- [ ] Implement high-performance canvas capture.

### Phase 3: ONNX / Transformers.js Integration
- [ ] Integrate ONNX Runtime Web (`onnxruntime-web`) or Transformers.js.
- [ ] Load embedding model / keypoint extractor into mobile browser.
- [ ] On startup, load precomputed reference database.
- [ ] Real-time matching logic:
  - Extract embedding/keypoints from live camera cropped frame.
  - Compute similarity (Cosine similarity for embeddings / LightGlue matching for keypoints).
  - Rank top-k matches.

### Phase 4: Mobile Optimization
- [ ] Enable WebGL/WebGPU acceleration in ONNX Runtime.
- [ ] Optimize model file size (quantization to INT8) to minimize loading time.
- [ ] Add loading state indicator while the model is downloading.
