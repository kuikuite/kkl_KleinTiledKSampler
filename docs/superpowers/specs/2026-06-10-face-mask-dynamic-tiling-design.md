# Face Mask Dynamic Tiling Design

## Goal

Update `Comfy-SZ-KleinKSampler` so VAE encode, Klein tiled sampling, and VAE
decode all use the same face-aware dynamic tile plan. A connected external
`face_mask` should reserve the detected face/person mask bbox as its own tile
region. The remaining image area should be covered by as few non-face tiles as
possible while preserving Klein and VAE alignment constraints.

## Constraints

- All user-facing tile and overlap controls are image-pixel values.
- Image-space tile boundaries must align to multiples of 16.
- Latent-space tile boundaries must align to the matching latent scale.
- Tiles must not exceed Klein's 2048 image-pixel maximum.
- The existing sampler behavior should remain available when `face_mask` is not
  connected or has no active pixels.
- Each tile path must support explicit overlap values from node inputs.

## Shared Planner

Add shared planning helpers to `SZ_KleinRegionPlanner`.

The planner produces region records with:

- `kind`: `face` or `background`
- `image_rect`: image-space `(y0, x0, h, w)`
- `latent_rect`: latent-space `(y0, x0, h, w)`
- `mask_mode`: face tiles are blended with the softened face mask, background
  tiles are blended with the inverse softened mask when a mask exists.

When there is a valid face mask:

1. Find the active mask bbox using `face_mask_threshold`.
2. Expand it by `face_padding`.
3. Align bbox edges to 16-pixel image units.
4. Clamp to image bounds.
5. Ensure the face region is at least the requested face tile size where bounds
   allow.
6. Generate face tiles inside that face region using `face_tile_width`,
   `face_tile_height`, and `face_overlap`.
7. Split the remaining image into four non-overlapping background rectangles:
   top band, bottom band, left band beside the face region, and right band beside
   the face region.
8. For each non-empty background rectangle, generate the fewest possible aligned
   tiles using the background tile size and overlap.

If there is no usable mask, the planner returns the existing regular grid.

## VAE Encode

`SZ Klein Face Region VAE Encode` will call the planner in image space. For each
planned tile it will:

1. Crop pixels using the image rect.
2. Encode the crop with `vae.encode`.
3. Convert the write rect to latent space.
4. Accumulate the encoded tile with a feather weight.
5. Multiply the weight by the proper region mask when a face mask exists.

The output latent keeps the same shape as a normal VAE encode for the aligned
input size.

## Sampler

`SZ KleinTiled KSampler` will call the same planner converted to latent space.
It will process:

1. Background tiles using inverse face mask weighting.
2. Face tiles using softened face mask weighting.

The existing global noise, `latent_blend`, conditioning reference cropping,
content sorting, pair batching, feather merge, and color preserve behavior stay
intact.

## VAE Decode

`SZ Klein Face Region VAE Decode` will call the planner in latent space and
decode each latent tile. It will write the decoded image tile into the output
image rect using the same feather and mask rules as encode.

## Error Handling

- If input image width or height is not divisible by 16, raise the existing clear
  alignment error.
- If the mask is empty or absent, fall back to the regular tiled path.
- If a region receives zero accumulated weight due to an extremely tight mask,
  clamp the denominator and preserve existing behavior.

## Tests

Add static/unit tests for:

- face bbox expansion and 16-pixel image alignment
- non-face rectangle decomposition around the face region
- fewest-tile behavior when background bands fit within one tile
- planner fallback when mask is absent
- VAE and sampler nodes exposing the shared dynamic tiling controls

