# Comfy-SZ-KleinKSampler
Dedicated face-region block sampling nodes for Klein series models.

## Nodes

- `SZ KleinTiled KSampler`
- `SZ Klein Face Region VAE Encode`
- `SZ Klein Face Region VAE Decode`

The sampler keeps the old behavior when `face_mask` is not connected. When a
mask is connected, it splits work into non-face and face regions:

- `tile_width` / `tile_height` / `overlap` control the non-face region.
- `face_tile_width` / `face_tile_height` / `face_overlap` control face tiles.
- `face_padding` expands the face bbox before face tiles are planned.
- `face_mask_threshold` decides which mask pixels count as face.
- `face_mask_grow` and `face_mask_blur` soften the merge boundary.

The VAE encode node, sampler, and VAE decode node now share one dynamic tile
plan. A valid mask creates one padded, 16-aligned face bbox region first. The
remaining image is split around that region into top, bottom, left, and right
background rectangles, and each rectangle is tiled with the fewest blocks that
fit the requested tile size and overlap. This keeps VAE and sampler boundaries
aligned while avoiding a full-image background grid when only a few large
non-face blocks are needed.

## Klein Size Rules

All tile controls are image-pixel values. The nodes clamp tile sizes to Klein's
2048 maximum input size, align tile sizes and overlaps to multiples of 16, then
convert to latent space internally.

Input images should already be resized or padded to width and height multiples
of 16. The included workflow sets the scaling nodes to `round_to_multiple = 16`.

Recommended defaults:

- Non-face tile: `2048 x 2048`, overlap `128`
- Face tile: `768 x 768`, overlap `192`
- Face padding: `1.35`
- Mask threshold: `0.2`

## Face Mask

Use any existing ComfyUI node that outputs a `MASK`, then connect it to the
sampler and the face-region VAE nodes. White is face, black is non-face.
