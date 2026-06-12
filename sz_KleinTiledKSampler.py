"""
SZ KleinTiled KSampler
----------------------
适用于 FLUX.2 Klein 模型的分块采样器。
主要用途：图像放大修复和细节增强。

核心功能：
  1. 外部接入 latent_blend 作为全局引导
  2. 生成空间连续的全局噪声图
  3. 支持 face_mask：人脸和非人脸区域分别规划 tile
  4. overlap 羽化混合写回
  5. 自动对齐原始图像的色彩统计量，防止饱和度漂移
  6. VAE encode/decode 节点复用同一套人脸/非人脸分区规则
"""

import math

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import latent_preview


class SZ_KleinRegionPlanner:
    KLEIN_MAX_IMAGE_SIZE = 2048
    IMAGE_ALIGNMENT = 16
    LATENT_DOWNSCALE = 8
    DEFAULT_VAE_DOWNSCALE = 16
    LATENT_ALIGNMENT = IMAGE_ALIGNMENT // LATENT_DOWNSCALE

    def _align_image_size(self, value, minimum=64, maximum=None):
        maximum = self.KLEIN_MAX_IMAGE_SIZE if maximum is None else maximum
        value = int(round(value))
        value = max(minimum, min(maximum, value))
        value = (value // self.IMAGE_ALIGNMENT) * self.IMAGE_ALIGNMENT
        return max(minimum, value)

    def _align_image_overlap(self, value, maximum=None):
        maximum = self.KLEIN_MAX_IMAGE_SIZE if maximum is None else maximum
        value = int(round(value))
        value = max(0, min(maximum, value))
        return (value // self.IMAGE_ALIGNMENT) * self.IMAGE_ALIGNMENT

    def _image_to_latent_tile(self, image_px):
        return max(1, self._align_image_size(image_px) // self.LATENT_DOWNSCALE)

    def _image_to_latent_overlap(self, image_px):
        return max(0, self._align_image_overlap(image_px) // self.LATENT_DOWNSCALE)

    def _align_down(self, value, unit):
        if unit <= 1:
            return int(value)
        return int(value) // unit * unit

    def _align_up(self, value, unit):
        if unit <= 1:
            return int(value)
        return int(math.ceil(float(value) / float(unit)) * unit)

    def _expand_region_to_min_size(self, start, end, limit, target, align_unit):
        target = min(limit, target)
        if end - start >= target:
            return start, end

        deficit = target - (end - start)
        start = max(0, start - deficit // 2)
        end = min(limit, end + deficit - deficit // 2)
        if end - start < target:
            if start == 0:
                end = min(limit, target)
            elif end == limit:
                start = max(0, limit - target)

        start = self._align_down(start, align_unit)
        end = self._align_up(end, align_unit)
        if end > limit:
            end = limit
            start = max(0, self._align_down(end - target, align_unit))
        if end - start < target:
            end = min(limit, start + target)
        return start, end

    def _anchored_tile_start(self, start, end, limit, tile, align_unit):
        if tile >= limit:
            return 0
        center = (start + end - tile) / 2.0
        candidates = [
            self._align_down(center, align_unit),
            self._align_up(center, align_unit),
            self._align_down(start, align_unit),
            self._align_up(end - tile, align_unit),
            0,
            limit - tile,
        ]
        valid = []
        for candidate in candidates:
            candidate = max(0, min(int(candidate), limit - tile))
            if candidate <= start and candidate + tile >= end:
                valid.append(candidate)
        if valid:
            return min(valid, key=lambda value: abs(value - center))
        return max(0, min(self._align_down(center, align_unit), limit - tile))

    def _get_tile_positions(self, H, W, tile_h, tile_w, overlap):
        tile_h = min(max(1, int(tile_h)), H)
        tile_w = min(max(1, int(tile_w)), W)
        overlap = max(0, int(overlap))
        if tile_h >= H and tile_w >= W:
            return [(0, 0, H, W)]
        stride_h = max(1, tile_h - overlap)
        stride_w = max(1, tile_w - overlap)
        positions = []
        seen = set()
        y = 0
        while True:
            y0 = min(y, H - tile_h)
            x = 0
            while True:
                x0 = min(x, W - tile_w)
                key = (y0, x0)
                if key not in seen:
                    seen.add(key)
                    positions.append((y0, x0, tile_h, tile_w))
                if x0 + tile_w >= W:
                    break
                x += stride_w
            if y0 + tile_h >= H:
                break
            y += stride_h
        return positions

    def _normalise_latent_downscale(self, latent_downscale=None):
        if latent_downscale is None:
            return self.DEFAULT_VAE_DOWNSCALE
        if isinstance(latent_downscale, (tuple, list)):
            latent_downscale = latent_downscale[0] if latent_downscale else None
        try:
            latent_downscale = int(round(float(latent_downscale)))
        except Exception:
            latent_downscale = self.DEFAULT_VAE_DOWNSCALE
        return max(1, latent_downscale)

    def _infer_latent_downscale_from_latent(self, latent, fallback=None):
        if isinstance(latent, dict):
            for key in ("downscale_ratio_spacial", "downscale_ratio", "latent_downscale"):
                value = latent.get(key, None)
                if value is not None:
                    return self._normalise_latent_downscale(value)
        return self._normalise_latent_downscale(fallback)

    def _infer_latent_downscale_from_mask(self, face_mask, latent_h, latent_w,
                                          fallback=None):
        fallback = self._normalise_latent_downscale(fallback)
        if face_mask is None or not hasattr(face_mask, "shape"):
            return fallback
        try:
            mask_h = int(face_mask.shape[-2])
            mask_w = int(face_mask.shape[-1])
        except Exception:
            return fallback
        if latent_h <= 0 or latent_w <= 0 or mask_h <= 0 or mask_w <= 0:
            return fallback
        if mask_h == latent_h and mask_w == latent_w:
            return fallback
        if mask_h % latent_h == 0 and mask_w % latent_w == 0:
            scale_h = mask_h // latent_h
            scale_w = mask_w // latent_w
            if scale_h == scale_w:
                return self._normalise_latent_downscale(scale_h)
        return fallback

    def _make_tile_record(self, kind, y0, x0, h, w, source_region=None,
                          latent_downscale=None):
        y0 = int(y0)
        x0 = int(x0)
        h = int(h)
        w = int(w)
        latent_downscale = self._normalise_latent_downscale(latent_downscale)
        return {
            "kind": kind,
            "image_rect": (y0, x0, h, w),
            "latent_rect": (
                y0 // latent_downscale,
                x0 // latent_downscale,
                max(1, h // latent_downscale),
                max(1, w // latent_downscale),
            ),
            "source_region": source_region,
        }

    def _tile_rect_region_minimal(self, y0, y1, x0, x1, tile_h, tile_w,
                                  overlap, source_region, latent_downscale=None):
        region_h = max(0, int(y1) - int(y0))
        region_w = max(0, int(x1) - int(x0))
        if region_h <= 0 or region_w <= 0:
            return []
        if region_h <= tile_h and region_w <= tile_w:
            return [
                self._make_tile_record(
                    "background", y0, x0, region_h, region_w, source_region,
                    latent_downscale
                )
            ]

        return [
            self._make_tile_record(
                "background", y0 + local_y, x0 + local_x, th, tw, source_region,
                latent_downscale
            )
            for local_y, local_x, th, tw in self._get_tile_positions(
                region_h, region_w, tile_h, tile_w, overlap
            )
        ]

    def _split_background_around_face_region(self, face_region, image_h, image_w,
                                             tile_h, tile_w, overlap,
                                             latent_downscale=None):
        y0, y1, x0, x1 = face_region
        pieces = []
        pieces.extend(
            self._tile_rect_region_minimal(
                0, y0, 0, image_w, tile_h, tile_w, overlap, "top",
                latent_downscale
            )
        )
        pieces.extend(
            self._tile_rect_region_minimal(
                y1, image_h, 0, image_w, tile_h, tile_w, overlap, "bottom",
                latent_downscale
            )
        )
        pieces.extend(
            self._tile_rect_region_minimal(
                y0, y1, 0, x0, tile_h, tile_w, overlap, "left",
                latent_downscale
            )
        )
        pieces.extend(
            self._tile_rect_region_minimal(
                y0, y1, x1, image_w, tile_h, tile_w, overlap, "right",
                latent_downscale
            )
        )
        return pieces

    def _face_region_from_bbox(self, bbox, image_h, image_w, face_tile_h,
                               face_tile_w, padding, align_unit):
        if bbox is None:
            return None
        orig_y_min, orig_y_max, orig_x_min, orig_x_max = bbox
        if orig_y_max <= orig_y_min or orig_x_max <= orig_x_min:
            return None

        box_h = max(1, orig_y_max - orig_y_min)
        box_w = max(1, orig_x_max - orig_x_min)
        pad_h = int(round((max(1.0, padding) - 1.0) * box_h / 2.0))
        pad_w = int(round((max(1.0, padding) - 1.0) * box_w / 2.0))

        y_min = max(0, self._align_down(orig_y_min - pad_h, align_unit))
        y_max = min(image_h, self._align_up(orig_y_max + pad_h, align_unit))
        x_min = max(0, self._align_down(orig_x_min - pad_w, align_unit))
        x_max = min(image_w, self._align_up(orig_x_max + pad_w, align_unit))

        y_min, y_max = self._expand_region_to_min_size(
            y_min, y_max, image_h, min(face_tile_h, image_h), align_unit
        )
        x_min, x_max = self._expand_region_to_min_size(
            x_min, x_max, image_w, min(face_tile_w, image_w), align_unit
        )
        return (y_min, y_max, x_min, x_max)

    def _regular_tile_plan(self, image_h, image_w, tile_h, tile_w, overlap,
                           latent_downscale=None):
        return [
            self._make_tile_record(
                "background", y0, x0, th, tw, "full", latent_downscale
            )
            for y0, x0, th, tw in self._get_tile_positions(
                image_h, image_w, tile_h, tile_w, overlap
            )
        ]

    def _plan_face_aware_tiles_from_bbox(self, bbox, image_h, image_w, tile_h,
                                         tile_w, overlap, face_tile_h,
                                         face_tile_w, face_overlap,
                                         face_padding, align_unit=None,
                                         latent_downscale=None):
        latent_downscale = self._normalise_latent_downscale(latent_downscale)
        image_h = int(image_h)
        image_w = int(image_w)
        tile_h = min(self._align_image_size(tile_h), image_h)
        tile_w = min(self._align_image_size(tile_w), image_w)
        overlap = self._align_image_overlap(overlap)
        face_tile_h = min(self._align_image_size(face_tile_h), image_h)
        face_tile_w = min(self._align_image_size(face_tile_w), image_w)
        face_overlap = self._align_image_overlap(face_overlap)
        align_unit = self.IMAGE_ALIGNMENT if align_unit is None else max(1, align_unit)

        face_region = self._face_region_from_bbox(
            bbox, image_h, image_w, face_tile_h, face_tile_w,
            face_padding, align_unit
        )
        if face_region is None:
            return self._regular_tile_plan(
                image_h, image_w, tile_h, tile_w, overlap, latent_downscale
            )

        background_tiles = self._split_background_around_face_region(
            face_region, image_h, image_w, tile_h, tile_w, overlap,
            latent_downscale
        )
        y0, y1, x0, x1 = face_region
        face_tiles = [
            self._make_tile_record(
                "face", y0 + local_y, x0 + local_x, th, tw, "face",
                latent_downscale
            )
            for local_y, local_x, th, tw in self._get_tile_positions(
                y1 - y0, x1 - x0, face_tile_h, face_tile_w, face_overlap
            )
        ]
        return background_tiles + face_tiles

    def _plan_face_aware_tiles_from_mask(self, face_mask, image_h, image_w,
                                         tile_h, tile_w, overlap, face_tile_h,
                                         face_tile_w, face_overlap,
                                         face_padding, threshold=0.2,
                                         mask_space="image",
                                         latent_downscale=None):
        latent_downscale = self._normalise_latent_downscale(latent_downscale)
        bbox = self._bbox_from_mask(face_mask, threshold)
        if bbox is not None and mask_space == "latent":
            bbox = tuple(int(value) * latent_downscale for value in bbox)
        return self._plan_face_aware_tiles_from_bbox(
            bbox, image_h, image_w, tile_h, tile_w, overlap,
            face_tile_h, face_tile_w, face_overlap, face_padding,
            latent_downscale=latent_downscale,
        )

    def _bbox_from_mask(self, mask, threshold):
        if mask is None:
            return None
        if mask.dim() == 4:
            mask = mask[:, 0]
        if mask.dim() == 3:
            mask = mask.max(dim=0).values
        active = mask > threshold
        if not bool(active.any()):
            return None

        ys, xs = active.nonzero(as_tuple=True)
        return (
            int(ys.min().item()),
            int(ys.max().item()) + 1,
            int(xs.min().item()),
            int(xs.max().item()) + 1,
        )

    def _get_face_tile_positions_from_bbox(self, bbox, H, W, tile_h, tile_w,
                                           overlap, padding, align_unit=None):
        if bbox is None:
            return []
        align_unit = self.LATENT_ALIGNMENT if align_unit is None else max(1, align_unit)
        orig_y_min, orig_y_max, orig_x_min, orig_x_max = bbox
        if orig_y_max <= orig_y_min or orig_x_max <= orig_x_min:
            return []

        box_h = max(1, orig_y_max - orig_y_min)
        box_w = max(1, orig_x_max - orig_x_min)
        pad_h = int(round((max(1.0, padding) - 1.0) * box_h / 2.0))
        pad_w = int(round((max(1.0, padding) - 1.0) * box_w / 2.0))

        y_min = max(0, self._align_down(orig_y_min - pad_h, align_unit))
        y_max = min(H, self._align_up(orig_y_max + pad_h, align_unit))
        x_min = max(0, self._align_down(orig_x_min - pad_w, align_unit))
        x_max = min(W, self._align_up(orig_x_max + pad_w, align_unit))

        y_min, y_max = self._expand_region_to_min_size(
            y_min, y_max, H, min(tile_h, H), align_unit
        )
        x_min, x_max = self._expand_region_to_min_size(
            x_min, x_max, W, min(tile_w, W), align_unit
        )

        region_h = max(1, y_max - y_min)
        region_w = max(1, x_max - x_min)
        local_tiles = self._get_tile_positions(region_h, region_w, tile_h, tile_w, overlap)
        positions = []

        if box_h <= tile_h and box_w <= tile_w:
            anchor_y = self._anchored_tile_start(
                orig_y_min, orig_y_max, H, min(tile_h, H), align_unit
            )
            anchor_x = self._anchored_tile_start(
                orig_x_min, orig_x_max, W, min(tile_w, W), align_unit
            )
            positions.append((anchor_y, anchor_x, min(tile_h, H), min(tile_w, W)))

        positions.extend((y_min + y0, x_min + x0, th, tw)
                         for (y0, x0, th, tw) in local_tiles)
        deduped = []
        seen = set()
        for tile in positions:
            key = (tile[0], tile[1], tile[2], tile[3])
            if key not in seen:
                seen.add(key)
                deduped.append(tile)
        return deduped

    def _get_face_tile_positions(self, face_mask, H, W, tile_h, tile_w, overlap,
                                 padding, threshold=0.2, align_unit=None):
        if face_mask is None:
            return []
        bbox = self._bbox_from_mask(face_mask, threshold)
        return self._get_face_tile_positions_from_bbox(
            bbox, H, W, tile_h, tile_w, overlap, padding, align_unit
        )

    def _prepare_face_mask(self, face_mask, H, W, B, device):
        """把 ComfyUI MASK 统一成当前空间的 (B,1,H,W) 软 mask。"""
        if face_mask is None:
            return None
        mask = face_mask.to(device=device, dtype=torch.float32)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4:
            if mask.shape[1] != 1:
                mask = mask[:, :1]
        else:
            return None
        if mask.shape[0] != B:
            mask = mask[:1].expand(B, -1, -1, -1).clone()
        if mask.shape[2] != H or mask.shape[3] != W:
            mask = F.interpolate(mask, size=(H, W), mode="bilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    def _soften_mask_units(self, mask, grow, blur, threshold):
        if mask is None:
            return None
        mask = (mask > threshold).to(dtype=mask.dtype)
        grow = max(0, int(grow))
        blur = max(0, int(blur))
        if grow > 0:
            kernel = grow * 2 + 1
            mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=grow)
        if blur > 0:
            kernel = blur * 2 + 1
            mask = F.pad(mask, (blur, blur, blur, blur), mode="replicate")
            mask = F.avg_pool2d(mask, kernel_size=kernel, stride=1)
        return mask.clamp(0.0, 1.0)

    def _soften_face_mask(self, face_mask, grow_px, blur_px, threshold=0.2,
                          latent_downscale=None):
        latent_downscale = self._normalise_latent_downscale(latent_downscale)
        grow = int(round(grow_px / latent_downscale))
        blur = int(round(blur_px / latent_downscale))
        return self._soften_mask_units(face_mask, grow, blur, threshold)

    def _soften_image_mask(self, face_mask, grow_px, blur_px, threshold=0.2):
        return self._soften_mask_units(face_mask, grow_px, blur_px, threshold)

    def _resize_to_latent_size(self, tensor, H, W):
        if tensor.shape[2] == H and tensor.shape[3] == W:
            return tensor
        return F.interpolate(tensor, size=(H, W), mode="bilinear", align_corners=False)

    def _make_weight_mask(self, h, w, device):
        wy = torch.arange(h, dtype=torch.float32, device=device)
        wy = (torch.min(wy, h - 1 - wy) + 1.0)
        wx = torch.arange(w, dtype=torch.float32, device=device)
        wx = (torch.min(wx, w - 1 - wx) + 1.0)
        weight = (wy.unsqueeze(1) * wx.unsqueeze(0))
        weight = weight / weight.max()
        return weight.unsqueeze(0).unsqueeze(0)

    def _make_image_weight_mask(self, h, w, device):
        return self._make_weight_mask(h, w, device).movedim(1, -1)

    def _validate_image_multiple_of_16(self, H, W):
        if H % self.IMAGE_ALIGNMENT != 0 or W % self.IMAGE_ALIGNMENT != 0:
            raise ValueError(
                "Klein face-region nodes require image dimensions to be multiples "
                "of 16. Resize or pad the image before this node."
            )


class SZ_KleinTiledKSampler(SZ_KleinRegionPlanner):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "latent_image":  ("LATENT",),
                "latent_blend":  ("LATENT",),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 100
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "tile_width": ("INT", {
                    "default": 2048, "min": 64, "max": 2048, "step": 16
                }),
                "tile_height": ("INT", {
                    "default": 2048, "min": 64, "max": 2048, "step": 16
                }),
                "overlap": ("INT", {
                    "default": 128, "min": 0, "max": 1024, "step": 16
                }),
                "face_tile_width": ("INT", {
                    "default": 768, "min": 64, "max": 2048, "step": 16
                }),
                "face_tile_height": ("INT", {
                    "default": 768, "min": 64, "max": 2048, "step": 16
                }),
                "face_overlap": ("INT", {
                    "default": 192, "min": 0, "max": 1024, "step": 16
                }),
                "face_padding": ("FLOAT", {
                    "default": 1.35, "min": 1.0, "max": 3.0, "step": 0.05
                }),
                "face_mask_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "face_mask_grow": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 16
                }),
                "face_mask_blur": ("INT", {
                    "default": 24, "min": 0, "max": 256, "step": 16
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "color_preserve": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            },
            "optional": {
                "face_mask": ("MASK",),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "sample"
    CATEGORY      = "SZ"

    def _sort_tiles_by_content(self, tile_positions, blend_up):
        """按 latent_blend 内容丰富度排序，内容多的 tile 优先处理。"""
        scores = []
        for (y0, x0, th, tw) in tile_positions:
            region = blend_up[:, :, y0:y0+th, x0:x0+tw]
            scores.append(float(region.var()))
        order = sorted(range(len(tile_positions)),
                       key=lambda i: scores[i], reverse=True)
        return [tile_positions[i] for i in order]

    def _scale_conditioning_refs(self, conditioning, aH, aW):
        scaled_cond = []
        for cond_pair in conditioning:
            cond_dict = cond_pair[1].copy()
            if "reference_latents" in cond_dict:
                cond_dict["reference_latents"] = [
                    F.interpolate(ref, size=(aH, aW),
                                  mode="bilinear", align_corners=False)
                    for ref in cond_dict["reference_latents"]
                ]
            scaled_cond.append([cond_pair[0], cond_dict])
        return scaled_cond

    def _crop_conditioning_refs(self, conditioning, y0, x0, th, tw):
        """按 tile 位置裁剪 reference_latents。"""
        cropped_cond = []
        for cond_pair in conditioning:
            cond_dict = cond_pair[1].copy()
            if "reference_latents" in cond_dict:
                cond_dict["reference_latents"] = [
                    ref[:, :, y0:y0+th, x0:x0+tw].clone()
                    for ref in cond_dict["reference_latents"]
                ]
            cropped_cond.append([cond_pair[0], cond_dict])
        return cropped_cond

    def _merge_conditioning_refs(self, cond1, cond2):
        """两个 tile 的 reference_latents 在 batch 维度合并。"""
        merged = []
        for cp1, cp2 in zip(cond1, cond2):
            cond_dict = cp1[1].copy()
            if "reference_latents" in cond_dict:
                refs1 = cp1[1]["reference_latents"]
                refs2 = cp2[1]["reference_latents"]
                cond_dict["reference_latents"] = [
                    torch.cat([r1, r2], dim=0)
                    for r1, r2 in zip(refs1, refs2)
                ]
            merged.append([cp1[0], cond_dict])
        return merged

    def _make_callback(self, pbar, previewer, total_steps):
        def callback(step, x0, x, total):
            preview_bytes = None
            if previewer:
                try:
                    preview_bytes = previewer.decode_latent_to_preview_image(
                        "JPEG", x0[:1]
                    )
                except Exception:
                    pass
            pbar.update_absolute(step + 1, total_steps, preview_bytes)
        return callback

    def _process_tile(self, m, positive, negative, samples,
                      global_noise, blend_up,
                      y0, x0, th, tw, B,
                      blend_strength,
                      steps, cfg, sampler_name, scheduler, denoise, seed,
                      previewer, device):
        """处理单个 tile。"""
        tile_noise  = global_noise[:, :, y0:y0+th, x0:x0+tw].clone()
        blend_tile  = blend_up    [:, :, y0:y0+th, x0:x0+tw]

        base_weight = max(0.0, 1.0 - blend_strength)
        tile_noise  = tile_noise * base_weight + blend_tile * blend_strength

        tile_ref      = samples[:, :, y0:y0+th, x0:x0+tw].clone().cpu()
        tile_positive = self._crop_conditioning_refs(positive, y0, x0, th, tw)
        tile_negative = self._crop_conditioning_refs(negative, y0, x0, th, tw)

        inner_pbar    = comfy.utils.ProgressBar(steps)
        tile_callback = self._make_callback(inner_pbar, previewer, steps)

        tile_result = comfy.sample.sample(
            m, tile_noise, steps, cfg, sampler_name, scheduler,
            tile_positive, tile_negative, tile_ref,
            denoise=denoise, seed=seed, callback=tile_callback,
        ).to(device)
        return self._resize_to_latent_size(tile_result, th, tw)

    def _process_tile_pair(self, m, positive, negative, samples,
                           global_noise, blend_up,
                           y0a, x0a, tha, twa,
                           y0b, x0b, thb, twb,
                           B,
                           blend_strength,
                           steps, cfg, sampler_name, scheduler, denoise, seed,
                           previewer, device):
        """两个相同尺寸的 tile 合并成 batch=2 一起处理。"""
        def prep_noise(y0, x0, th, tw):
            noise  = global_noise[:, :, y0:y0+th, x0:x0+tw].clone()
            blend  = blend_up    [:, :, y0:y0+th, x0:x0+tw]
            base_weight = max(0.0, 1.0 - blend_strength)
            return noise * base_weight + blend * blend_strength

        noise_a    = prep_noise(y0a, x0a, tha, twa)
        noise_b    = prep_noise(y0b, x0b, thb, twb)
        tile_noise = torch.cat([noise_a, noise_b], dim=0)

        ref_a    = samples[:, :, y0a:y0a+tha, x0a:x0a+twa].clone().cpu()
        ref_b    = samples[:, :, y0b:y0b+thb, x0b:x0b+twb].clone().cpu()
        tile_ref = torch.cat([ref_a, ref_b], dim=0)

        pos_a = self._crop_conditioning_refs(positive, y0a, x0a, tha, twa)
        pos_b = self._crop_conditioning_refs(positive, y0b, x0b, thb, twb)
        neg_a = self._crop_conditioning_refs(negative, y0a, x0a, tha, twa)
        neg_b = self._crop_conditioning_refs(negative, y0b, x0b, thb, twb)
        tile_positive = self._merge_conditioning_refs(pos_a, pos_b)
        tile_negative = self._merge_conditioning_refs(neg_a, neg_b)

        inner_pbar    = comfy.utils.ProgressBar(steps)
        tile_callback = self._make_callback(inner_pbar, previewer, steps)

        pair_result = comfy.sample.sample(
            m, tile_noise, steps, cfg, sampler_name, scheduler,
            tile_positive, tile_negative, tile_ref,
            denoise=denoise, seed=seed, callback=tile_callback,
        ).to(device)
        pair_result = self._resize_to_latent_size(pair_result, tha, twa)

        return pair_result[:B], pair_result[B:]

    def _accumulate_tile(self, result, weight_map, tile_result, y0, x0, th, tw,
                         weight, region_mask=None):
        if region_mask is not None:
            mask = region_mask[:, :, y0:y0+th, x0:x0+tw]
            if mask.shape[2] != weight.shape[2] or mask.shape[3] != weight.shape[3]:
                mask = F.interpolate(mask, size=weight.shape[2:],
                                     mode="bilinear", align_corners=False)
            weight = weight * mask.to(device=weight.device, dtype=weight.dtype)
        result[:, :, y0:y0+th, x0:x0+tw] += tile_result * weight
        weight_map[:, :, y0:y0+th, x0:x0+tw] += weight

    def _process_and_accumulate_tiles(self, model, positive, negative, samples,
                                      global_noise, blend_up, tile_positions,
                                      result, weight_map, region_mask,
                                      B, blend_strength,
                                      steps, cfg, sampler_name, scheduler,
                                      denoise, seed, previewer, device,
                                      outer_pbar, total_progress,
                                      progress_offset=0):
        idx = 0
        total_tiles = len(tile_positions)
        while idx < total_tiles:
            y0a, x0a, tha, twa = tile_positions[idx]

            if idx + 1 < total_tiles:
                y0b, x0b, thb, twb = tile_positions[idx + 1]
                same_size = (tha == thb and twa == twb)
            else:
                same_size = False

            if same_size and B == 1:
                result_a, result_b = self._process_tile_pair(
                    model, positive, negative, samples,
                    global_noise, blend_up,
                    y0a, x0a, tha, twa,
                    y0b, x0b, thb, twb,
                    B, blend_strength,
                    steps, cfg, sampler_name, scheduler, denoise, seed,
                    previewer, device
                )
                weight_a = self._make_weight_mask(tha, twa, device)
                weight_b = self._make_weight_mask(thb, twb, device)
                self._accumulate_tile(result, weight_map, result_a, y0a, x0a, tha, twa,
                                      weight_a, region_mask)
                self._accumulate_tile(result, weight_map, result_b, y0b, x0b, thb, twb,
                                      weight_b, region_mask)
                outer_pbar.update_absolute(progress_offset + idx + 2,
                                           total_progress, None)
                idx += 2
            else:
                tile_result = self._process_tile(
                    model, positive, negative, samples,
                    global_noise, blend_up,
                    y0a, x0a, tha, twa, B,
                    blend_strength,
                    steps, cfg, sampler_name, scheduler, denoise, seed,
                    previewer, device
                )
                weight = self._make_weight_mask(tha, twa, device)
                self._accumulate_tile(result, weight_map, tile_result, y0a, x0a, tha, twa,
                                      weight, region_mask)
                outer_pbar.update_absolute(progress_offset + idx + 1,
                                           total_progress, None)
                idx += 1

    def _match_color_stats(self, result, original, strength):
        """
        对齐生成结果和原始图像的色彩统计量（均值+标准差）
        防止分块采样后饱和度/明度/对比度漂移
        strength=1.0 完全对齐，strength=0.0 不对齐
        """
        if strength <= 0.0:
            return result
        matched = result.clone()
        for c in range(result.shape[1]):
            orig_mean = original[:, c].mean()
            orig_std  = original[:, c].std()
            res_mean  = result[:, c].mean()
            res_std   = result[:, c].std()
            if res_std > 1e-8:
                normalized = (result[:, c] - res_mean) / res_std
                adjusted   = normalized * orig_std + orig_mean
                matched[:, c] = result[:, c] * (1.0 - strength) + adjusted * strength
        return matched

    def sample(self, model, positive, negative, latent_image, latent_blend,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tile_width, tile_height, overlap,
               face_tile_width, face_tile_height, face_overlap,
               face_padding, face_mask_threshold,
               face_mask_grow, face_mask_blur,
               blend_strength, color_preserve,
               face_mask=None):

        device  = comfy.model_management.get_torch_device()
        samples = latent_image["samples"].clone().to(device)
        B, C, H, W = samples.shape
        latent_downscale = self._infer_latent_downscale_from_latent(latent_image)

        previewer = latent_preview.get_previewer(device, model.model.latent_format)

        # ── 处理 latent_blend ────────────────────────────────────────────
        b = latent_blend["samples"].to(device)
        if b.shape[0] != B:
            b = b.expand(B, -1, -1, -1).clone()
        if b.shape[2] != H or b.shape[3] != W:
            b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
        b_min = b.min(); b_max = b.max()
        if b_max - b_min > 1e-8:
            b = (b - b_min) / (b_max - b_min) * 2.0 - 1.0
        blend_up = b

        # ── 全局噪声图 ────────────────────────────────────────────────────
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        global_noise = torch.randn((B, C, H, W), generator=gen).to(device)

        face_mask_up = self._prepare_face_mask(face_mask, H, W, B, device)
        face_mask_soft = self._soften_face_mask(
            face_mask_up, face_mask_grow, face_mask_blur, face_mask_threshold,
            latent_downscale
        )
        tile_plan = self._plan_face_aware_tiles_from_mask(
            face_mask_up,
            H * latent_downscale,
            W * latent_downscale,
            tile_height,
            tile_width,
            overlap,
            face_tile_height,
            face_tile_width,
            face_overlap,
            face_padding,
            face_mask_threshold,
            mask_space="latent",
            latent_downscale=latent_downscale,
        )
        background_positions = [
            tile["latent_rect"] for tile in tile_plan if tile["kind"] == "background"
        ]
        face_positions = [
            tile["latent_rect"] for tile in tile_plan if tile["kind"] == "face"
        ]
        background_positions = self._sort_tiles_by_content(background_positions, blend_up)
        if face_positions:
            face_positions = self._sort_tiles_by_content(face_positions, blend_up)
            print(
                f"[SZ_KleinTiledKSampler] 非人脸区域 {len(background_positions)} 个 tile，"
                f"人脸区域 {len(face_positions)} 个 tile"
            )
        else:
            print(f"[SZ_KleinTiledKSampler] 共 {len(background_positions)} 个 tile")

        result     = torch.zeros((B, C, H, W), device=device)
        weight_map = torch.zeros((B, 1, H, W), device=device)
        total_tiles = len(background_positions) + len(face_positions)
        outer_pbar = comfy.utils.ProgressBar(total_tiles)

        background_region_mask = None

        self._process_and_accumulate_tiles(
            model, positive, negative, samples,
            global_noise, blend_up, background_positions,
            result, weight_map, background_region_mask,
            B, blend_strength,
            steps, cfg, sampler_name, scheduler, denoise, seed,
            previewer, device, outer_pbar, total_tiles, 0
        )

        if face_positions and face_mask_soft is not None:
            self._process_and_accumulate_tiles(
                model, positive, negative, samples,
                global_noise, blend_up, face_positions,
                result, weight_map, face_mask_soft,
                B, blend_strength,
                steps, cfg, sampler_name, scheduler, denoise, seed,
                previewer, device, outer_pbar, total_tiles, len(background_positions)
            )

        result = result / weight_map.clamp(min=1e-8)

        # ── 色彩统计对齐（防止饱和度漂移）────────────────────────────────
        if color_preserve > 0.0:
            result = self._match_color_stats(result, samples, color_preserve)

        return ({"samples": result.cpu()},)


class SZ_KleinFaceRegionVAEEncode(SZ_KleinRegionPlanner):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "tile_size": ("INT", {
                    "default": 2048, "min": 64, "max": 2048, "step": 16
                }),
                "overlap": ("INT", {
                    "default": 128, "min": 0, "max": 1024, "step": 16
                }),
                "face_tile_size": ("INT", {
                    "default": 768, "min": 64, "max": 2048, "step": 16
                }),
                "face_overlap": ("INT", {
                    "default": 192, "min": 0, "max": 1024, "step": 16
                }),
                "face_padding": ("FLOAT", {
                    "default": 1.35, "min": 1.0, "max": 3.0, "step": 0.05
                }),
                "face_mask_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "face_mask_grow": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 16
                }),
                "face_mask_blur": ("INT", {
                    "default": 24, "min": 0, "max": 256, "step": 16
                }),
            },
            "optional": {
                "face_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "SZ"

    def _accumulate_encoded_tile(self, vae, pixels, y0, x0, th, tw,
                                 result_state, region_mask, device,
                                 latent_downscale):
        latent = vae.encode(pixels[:, y0:y0+th, x0:x0+tw, :]).to(device)
        B, C, lh, lw = latent.shape
        if lh > 0 and lw > 0:
            tile_scale_h = max(1, int(round(float(th) / float(lh))))
            tile_scale_w = max(1, int(round(float(tw) / float(lw))))
            if tile_scale_h == tile_scale_w:
                latent_downscale = tile_scale_h
        ly0 = y0 // latent_downscale
        lx0 = x0 // latent_downscale

        result, weight_map = result_state
        if result is None:
            full_h = pixels.shape[1] // latent_downscale
            full_w = pixels.shape[2] // latent_downscale
            result = torch.zeros((B, C, full_h, full_w), device=device)
            weight_map = torch.zeros((B, 1, full_h, full_w), device=device)

        weight = self._make_weight_mask(lh, lw, device)
        if region_mask is not None:
            mask = region_mask[:, :, y0:y0+th, x0:x0+tw]
            mask = F.interpolate(mask, size=(lh, lw), mode="bilinear", align_corners=False)
            weight = weight * mask.to(device=device, dtype=weight.dtype)

        result[:, :, ly0:ly0+lh, lx0:lx0+lw] += latent * weight
        weight_map[:, :, ly0:ly0+lh, lx0:lx0+lw] += weight
        return result, weight_map

    def encode(self, pixels, vae, tile_size, overlap, face_tile_size,
               face_overlap, face_padding, face_mask_threshold,
               face_mask_grow, face_mask_blur, face_mask=None):
        device = comfy.model_management.get_torch_device()
        pixels = pixels.to(device)
        B, H, W, C = pixels.shape
        self._validate_image_multiple_of_16(H, W)

        face_mask_up = self._prepare_face_mask(face_mask, H, W, B, device)
        face_mask_soft = self._soften_image_mask(
            face_mask_up, face_mask_grow, face_mask_blur, face_mask_threshold
        )
        latent_downscale = self.DEFAULT_VAE_DOWNSCALE
        tile_plan = self._plan_face_aware_tiles_from_mask(
            face_mask_up,
            H,
            W,
            tile_size,
            tile_size,
            overlap,
            face_tile_size,
            face_tile_size,
            face_overlap,
            face_padding,
            face_mask_threshold,
            latent_downscale=latent_downscale,
        )
        background_tiles = [
            tile["image_rect"] for tile in tile_plan if tile["kind"] == "background"
        ]
        face_tiles = [
            tile["image_rect"] for tile in tile_plan if tile["kind"] == "face"
        ]

        result_state = (None, None)
        background_region_mask = None

        total_tiles = len(background_tiles) + len(face_tiles)
        pbar = comfy.utils.ProgressBar(total_tiles)
        done = 0
        for y0, x0, th, tw in background_tiles:
            result_state = self._accumulate_encoded_tile(
                vae, pixels, y0, x0, th, tw, result_state, background_region_mask, device,
                latent_downscale
            )
            done += 1
            pbar.update_absolute(done, total_tiles, None)

        for y0, x0, th, tw in face_tiles:
            result_state = self._accumulate_encoded_tile(
                vae, pixels, y0, x0, th, tw, result_state, face_mask_soft, device,
                latent_downscale
            )
            done += 1
            pbar.update_absolute(done, total_tiles, None)

        result, weight_map = result_state
        if result is None:
            result = vae.encode(pixels).to(device)
        else:
            result = result / weight_map.clamp(min=1e-8)
        return ({"samples": result.cpu()},)


class SZ_KleinFaceRegionVAEDecode(SZ_KleinRegionPlanner):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {
                    "default": 2048, "min": 64, "max": 2048, "step": 16
                }),
                "overlap": ("INT", {
                    "default": 128, "min": 0, "max": 1024, "step": 16
                }),
                "face_tile_size": ("INT", {
                    "default": 768, "min": 64, "max": 2048, "step": 16
                }),
                "face_overlap": ("INT", {
                    "default": 192, "min": 0, "max": 1024, "step": 16
                }),
                "face_padding": ("FLOAT", {
                    "default": 1.35, "min": 1.0, "max": 3.0, "step": 0.05
                }),
                "face_mask_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "face_mask_grow": ("INT", {
                    "default": 0, "min": 0, "max": 256, "step": 16
                }),
                "face_mask_blur": ("INT", {
                    "default": 24, "min": 0, "max": 256, "step": 16
                }),
            },
            "optional": {
                "face_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "SZ"

    def _accumulate_decoded_tile(self, vae, samples, y0, x0, th, tw,
                                 result_state, region_mask, device,
                                 latent_downscale):
        image = vae.decode(samples[:, :, y0:y0+th, x0:x0+tw]).to(device)
        B, ih, iw, C = image.shape
        if th > 0 and tw > 0:
            tile_scale_h = max(1, int(round(float(ih) / float(th))))
            tile_scale_w = max(1, int(round(float(iw) / float(tw))))
            if tile_scale_h == tile_scale_w:
                latent_downscale = tile_scale_h
        iy0 = y0 * latent_downscale
        ix0 = x0 * latent_downscale

        result, weight_map = result_state
        if result is None:
            full_h = samples.shape[2] * latent_downscale
            full_w = samples.shape[3] * latent_downscale
            result = torch.zeros((B, full_h, full_w, C), device=device)
            weight_map = torch.zeros((B, full_h, full_w, 1), device=device)

        weight = self._make_image_weight_mask(ih, iw, device)
        if region_mask is not None:
            mask = region_mask[:, :, y0:y0+th, x0:x0+tw]
            mask = F.interpolate(mask, size=(ih, iw), mode="bilinear", align_corners=False)
            mask = mask.movedim(1, -1)
            weight = weight * mask.to(device=device, dtype=weight.dtype)

        result[:, iy0:iy0+ih, ix0:ix0+iw, :] += image * weight
        weight_map[:, iy0:iy0+ih, ix0:ix0+iw, :] += weight
        return result, weight_map

    def decode(self, samples, vae, tile_size, overlap, face_tile_size,
               face_overlap, face_padding, face_mask_threshold,
               face_mask_grow, face_mask_blur, face_mask=None):
        device = comfy.model_management.get_torch_device()
        latent = samples["samples"].to(device)
        B, C, H, W = latent.shape
        latent_downscale = self._infer_latent_downscale_from_mask(
            face_mask, H, W, self._infer_latent_downscale_from_latent(samples)
        )

        face_mask_up = self._prepare_face_mask(face_mask, H, W, B, device)
        face_mask_soft = self._soften_face_mask(
            face_mask_up, face_mask_grow, face_mask_blur, face_mask_threshold,
            latent_downscale
        )
        tile_plan = self._plan_face_aware_tiles_from_mask(
            face_mask_up,
            H * latent_downscale,
            W * latent_downscale,
            tile_size,
            tile_size,
            overlap,
            face_tile_size,
            face_tile_size,
            face_overlap,
            face_padding,
            face_mask_threshold,
            mask_space="latent",
            latent_downscale=latent_downscale,
        )
        background_tiles = [
            tile["latent_rect"] for tile in tile_plan if tile["kind"] == "background"
        ]
        face_tiles = [
            tile["latent_rect"] for tile in tile_plan if tile["kind"] == "face"
        ]

        result_state = (None, None)
        background_region_mask = None

        total_tiles = len(background_tiles) + len(face_tiles)
        pbar = comfy.utils.ProgressBar(total_tiles)
        done = 0
        for y0, x0, th, tw in background_tiles:
            result_state = self._accumulate_decoded_tile(
                vae, latent, y0, x0, th, tw, result_state, background_region_mask, device,
                latent_downscale
            )
            done += 1
            pbar.update_absolute(done, total_tiles, None)

        for y0, x0, th, tw in face_tiles:
            result_state = self._accumulate_decoded_tile(
                vae, latent, y0, x0, th, tw, result_state, face_mask_soft, device,
                latent_downscale
            )
            done += 1
            pbar.update_absolute(done, total_tiles, None)

        result, weight_map = result_state
        if result is None:
            result = vae.decode(latent).to(device)
        else:
            result = result / weight_map.clamp(min=1e-8)
        return (result.cpu(),)


NODE_CLASS_MAPPINGS = {
    "SZ_KleinTiledKSampler": SZ_KleinTiledKSampler,
    "SZ_KleinFaceRegionVAEEncode": SZ_KleinFaceRegionVAEEncode,
    "SZ_KleinFaceRegionVAEDecode": SZ_KleinFaceRegionVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SZ_KleinTiledKSampler": "SZ KleinTiled KSampler",
    "SZ_KleinFaceRegionVAEEncode": "SZ Klein Face Region VAE Encode",
    "SZ_KleinFaceRegionVAEDecode": "SZ Klein Face Region VAE Decode",
}
