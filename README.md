# Comfy-SZ-KleinKSampler
Dedicated face-region block sampling nodes for Klein series models.

## Nodes

- `SZ KleinTiled KSampler`
- `SZ Klein Face Region VAE Encode`
- `SZ Klein Face Region VAE Decode`

The sampler keeps the old behavior when `face_mask` is not connected. When a
mask is connected, it splits work into non-face and face regions:

- `tile_width` / `tile_height` / `overlap` control the non-face region.
- `face_tile_width` / `face_tile_height` / `face_overlap` control the upscaled
  face processing canvas and overlap when a face region needs multiple tiles.
- `face_padding` expands the face bbox before face tiles are planned.
- `face_mask_threshold` decides which mask pixels count as face.
- `face_mask_grow` and `face_mask_blur` soften the merge boundary.

The VAE encode node, sampler, and VAE decode node now share one dynamic tile
plan. A valid mask creates one padded, 16-aligned face bbox region first. The
background pass then lays down a full-image underlay using the fewest regular
tiles for the requested background tile size and overlap. The face bbox stays
compact in the final image, but each face crop is resized into the
`face_tile_size` processing canvas before VAE encode, sampling, and VAE decode.
After processing, only the canvas content area is resized back to the original
face bbox and blended through the softened mask. This gives the model a larger
face to work on for ID preservation without pasting a huge face tile back into
the image.

## Klein Size Rules

All tile controls are image-pixel values. The nodes clamp tile sizes to Klein's
2048 maximum input size, align tile sizes and overlaps to multiples of 16, then
convert to latent space internally.

Input images should already be resized or padded to width and height multiples
of 16. The included workflow sets the scaling nodes to `round_to_multiple = 16`.

Recommended defaults:

- Non-face tile: `2048 x 2048`, overlap `128`
- Face processing tile: `768 x 768`, overlap `192`
- Face padding: `1.35`
- Mask threshold: `0.2`
- Mask grow / blur: `64 / 64`

## Face Mask

Use any existing ComfyUI node that outputs a `MASK`, then connect it to the
sampler and the face-region VAE nodes. White is face, black is non-face.
