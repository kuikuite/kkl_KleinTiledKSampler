# Face Region Klein Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace uniform Klein tiling with face-mask-aware face and non-face region tiling for sampling plus VAE encode/decode.

**Architecture:** Keep the existing sampler compatible when no mask is connected. Add shared region planning helpers that clamp tiles to Klein's 2048 image-pixel limit and align image-space boundaries to multiples of 16, then reuse that planning in the sampler and the new VAE encode/decode nodes.

**Tech Stack:** Python, PyTorch tensors, ComfyUI node contracts, static unit tests with lightweight ComfyUI stubs.

---

### Task 1: Region Planning Tests

**Files:**
- Create: `test_face_region_planning.py`
- Modify: `sz_KleinTiledKSampler.py`

- [ ] Write tests for 16-aligned non-face and face tile planning.
- [ ] Run `python3 -m unittest test_face_region_planning.py -v` and confirm failures for missing helpers.
- [ ] Implement `_align_to_16`, `_image_to_latent_tile`, `_prepare_face_mask`, `_soften_face_mask`, and `_get_face_tile_positions`.
- [ ] Run `python3 -m unittest test_face_region_planning.py -v` and confirm pass.

### Task 2: Sampler Contract

**Files:**
- Create: `test_node_contract_static.py`
- Modify: `sz_KleinTiledKSampler.py`

- [ ] Write static tests that `SZ_KleinTiledKSampler` exposes `face_tile_width`, `face_tile_height`, `face_overlap`, `face_padding`, `face_mask_threshold`, `face_mask_grow`, `face_mask_blur`, and optional `face_mask`.
- [ ] Run `python3 -m unittest test_node_contract_static.py -v` and confirm failures.
- [ ] Add the inputs to the sampler while preserving the existing no-mask code path.
- [ ] Run `python3 -m unittest test_node_contract_static.py -v` and confirm pass.

### Task 3: Face/Non-Face Sampler Execution

**Files:**
- Modify: `sz_KleinTiledKSampler.py`
- Test: `test_face_region_planning.py`

- [ ] Add accumulation helpers that can weight tile writes by face or non-face masks.
- [ ] Update `sample()` so no mask uses the old uniform path, while a mask uses non-face tiles plus face tiles.
- [ ] Keep tile sizes in image pixels, cap them to 2048, align them to 16, then divide by 8 for latent tiles.
- [ ] Run `python3 -m unittest discover -v`.

### Task 4: Face Region VAE Nodes

**Files:**
- Modify: `sz_KleinTiledKSampler.py`
- Modify: `__init__.py`
- Test: `test_vae_node_contract_static.py`

- [ ] Add `SZ_KleinFaceRegionVAEEncode` and `SZ_KleinFaceRegionVAEDecode`.
- [ ] Register both nodes in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- [ ] Implement mask-aware encode/decode by tiling face and non-face regions and soft-mask blending outputs.
- [ ] Run `python3 -m unittest discover -v`.

### Task 5: Workflow and Docs

**Files:**
- Modify: `Klein-高清放大-SZ分块采样器.json`
- Modify: `README.md`

- [ ] Update the workflow node types from core `VAEEncodeTiled` / `VAEDecodeTiled` to the new SZ face-region VAE nodes where mask is available.
- [ ] Document mask input, 2048 cap, and 16-multiple alignment.
- [ ] Run `python3 -m unittest discover -v`.
- [ ] Run `python3 -m py_compile sz_KleinTiledKSampler.py __init__.py`.
