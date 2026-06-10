# Face Mask Dynamic Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one shared face-aware dynamic tiling planner and use it in VAE encode, Klein tiled sampling, and VAE decode.

**Architecture:** Keep the existing single-file node layout in `sz_KleinTiledKSampler.py`. Add planner helpers to `SZ_KleinRegionPlanner`, then wire encode, sampler, and decode to consume planner output instead of separately generating face/background grids.

**Tech Stack:** Python, PyTorch tensors, ComfyUI custom node contracts, local `unittest` static tests.

---

### Task 1: Planner Tests

**Files:**
- Modify: `test_face_region_planning.py`

- [ ] **Step 1: Add tests for aligned face region and background decomposition**

Add these tests to `FaceRegionPlanningTest`:

```python
def test_face_plan_decomposes_background_into_few_large_regions(self):
    plan = self.sampler._plan_face_aware_tiles_from_bbox(
        bbox=(320, 704, 448, 832),
        image_h=1024,
        image_w=1280,
        tile_h=512,
        tile_w=512,
        overlap=128,
        face_tile_h=384,
        face_tile_w=384,
        face_overlap=192,
        face_padding=1.0,
    )

    face_tiles = [tile for tile in plan if tile["kind"] == "face"]
    background_tiles = [tile for tile in plan if tile["kind"] == "background"]

    self.assertTrue(face_tiles)
    self.assertEqual(len(background_tiles), 7)
    self.assertTrue(all(tile["image_rect"][0] % 16 == 0 for tile in plan))
    self.assertTrue(all(tile["image_rect"][1] % 16 == 0 for tile in plan))
    self.assertTrue(any(tile["source_region"] == "top" for tile in background_tiles))
    self.assertTrue(any(tile["source_region"] == "bottom" for tile in background_tiles))
    self.assertTrue(any(tile["source_region"] == "left" for tile in background_tiles))
    self.assertTrue(any(tile["source_region"] == "right" for tile in background_tiles))

def test_background_band_that_fits_tile_stays_single_tile(self):
    plan = self.sampler._split_background_around_face_region(
        face_region=(256, 768, 256, 768),
        image_h=1024,
        image_w=1024,
        tile_h=512,
        tile_w=512,
        overlap=128,
    )

    self.assertEqual(
        [tile["image_rect"] for tile in plan],
        [
            (0, 0, 256, 1024),
            (768, 0, 256, 1024),
            (256, 0, 512, 256),
            (256, 768, 512, 256),
        ],
    )

def test_regular_plan_fallback_without_face_bbox(self):
    plan = self.sampler._plan_face_aware_tiles_from_bbox(
        bbox=None,
        image_h=512,
        image_w=768,
        tile_h=512,
        tile_w=512,
        overlap=128,
        face_tile_h=384,
        face_tile_w=384,
        face_overlap=192,
        face_padding=1.35,
    )

    self.assertEqual(len(plan), 2)
    self.assertTrue(all(tile["kind"] == "background" for tile in plan))
```

- [ ] **Step 2: Run the focused planning tests and confirm they fail**

Run: `python3 -m unittest test_face_region_planning.py -v`

Expected: FAIL with missing `_plan_face_aware_tiles_from_bbox` or `_split_background_around_face_region`.

### Task 2: Shared Planner Implementation

**Files:**
- Modify: `sz_KleinTiledKSampler.py`

- [ ] **Step 1: Implement planner helpers on `SZ_KleinRegionPlanner`**

Add helper methods:

```python
def _make_tile_record(self, kind, y0, x0, h, w, source_region=None):
    return {
        "kind": kind,
        "image_rect": (int(y0), int(x0), int(h), int(w)),
        "latent_rect": (
            int(y0) // self.LATENT_DOWNSCALE,
            int(x0) // self.LATENT_DOWNSCALE,
            max(1, int(h) // self.LATENT_DOWNSCALE),
            max(1, int(w) // self.LATENT_DOWNSCALE),
        ),
        "source_region": source_region,
    }

def _tile_rect_region_minimal(self, y0, y1, x0, x1, tile_h, tile_w, overlap, source_region):
    region_h = max(0, int(y1) - int(y0))
    region_w = max(0, int(x1) - int(x0))
    if region_h <= 0 or region_w <= 0:
        return []
    if region_h <= tile_h and region_w <= tile_w:
        return [self._make_tile_record("background", y0, x0, region_h, region_w, source_region)]
    local_tiles = self._get_tile_positions(region_h, region_w, tile_h, tile_w, overlap)
    return [
        self._make_tile_record("background", y0 + ly, x0 + lx, th, tw, source_region)
        for ly, lx, th, tw in local_tiles
    ]

def _split_background_around_face_region(self, face_region, image_h, image_w, tile_h, tile_w, overlap):
    y0, y1, x0, x1 = face_region
    pieces = []
    pieces.extend(self._tile_rect_region_minimal(0, y0, 0, image_w, tile_h, tile_w, overlap, "top"))
    pieces.extend(self._tile_rect_region_minimal(y1, image_h, 0, image_w, tile_h, tile_w, overlap, "bottom"))
    pieces.extend(self._tile_rect_region_minimal(y0, y1, 0, x0, tile_h, tile_w, overlap, "left"))
    pieces.extend(self._tile_rect_region_minimal(y0, y1, x1, image_w, tile_h, tile_w, overlap, "right"))
    return pieces
```

Also add:

```python
def _face_region_from_bbox(self, bbox, image_h, image_w, face_tile_h, face_tile_w, padding, align_unit):
    # Expand bbox by padding, align edges, clamp, and ensure minimum tile size.
```

and:

```python
def _plan_face_aware_tiles_from_bbox(...):
    # Return regular background grid when bbox is None.
    # Return background records from _split_background_around_face_region plus face records.
```

- [ ] **Step 2: Run focused tests and confirm planner passes**

Run: `python3 -m unittest test_face_region_planning.py -v`

Expected: PASS.

### Task 3: Wire Sampler To Planner

**Files:**
- Modify: `sz_KleinTiledKSampler.py`

- [ ] **Step 1: Replace separate sampler grids with shared planner output**

In `SZ_KleinTiledKSampler.sample`, after `face_mask_up` and `face_mask_soft`, compute:

```python
tile_plan = self._plan_face_aware_tiles_from_mask(
    face_mask_up,
    H * self.LATENT_DOWNSCALE,
    W * self.LATENT_DOWNSCALE,
    tile_height,
    tile_width,
    overlap,
    face_tile_height,
    face_tile_width,
    face_overlap,
    face_padding,
    face_mask_threshold,
)
background_positions = [tile["latent_rect"] for tile in tile_plan if tile["kind"] == "background"]
face_positions = [tile["latent_rect"] for tile in tile_plan if tile["kind"] == "face"]
```

Sort each list by content with `_sort_tiles_by_content`, then pass background positions with inverse face mask and face positions with face mask.

- [ ] **Step 2: Run node contract and planning tests**

Run: `python3 -m unittest test_face_region_planning.py test_node_contract_static.py -v`

Expected: PASS.

### Task 4: Wire VAE Encode And Decode To Planner

**Files:**
- Modify: `sz_KleinTiledKSampler.py`

- [ ] **Step 1: Update VAE encode planning**

In `SZ_KleinFaceRegionVAEEncode.encode`, compute `tile_plan` once using image dimensions. Use:

```python
background_tiles = [tile["image_rect"] for tile in tile_plan if tile["kind"] == "background"]
face_tiles = [tile["image_rect"] for tile in tile_plan if tile["kind"] == "face"]
```

Accumulate background tiles with inverse face mask, then face tiles with face mask.

- [ ] **Step 2: Update VAE decode planning**

In `SZ_KleinFaceRegionVAEDecode.decode`, compute the same image-space plan with `image_h = H * 8` and `image_w = W * 8`. Use each tile's `latent_rect` for latent crop/decode.

- [ ] **Step 3: Run all local tests**

Run: `python3 -m unittest discover -v`

Expected: PASS.

### Task 5: Docs And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README behavior notes**

Add a short note that the face-region VAE encode, sampler, and VAE decode share one dynamic tile plan: face mask bbox is isolated, and non-face areas are split into top/bottom/left/right rectangles with minimal tiling.

- [ ] **Step 2: Run static search for old misleading text**

Run: `rg -n "regular grid|average grid|same face and non-face planning idea|普通 tile|普通区域" README.md sz_KleinTiledKSampler.py`

Expected: No text should claim the face-mask path uses a full regular non-face grid.

- [ ] **Step 3: Run final tests**

Run: `python3 -m unittest discover -v`

Expected: PASS.

