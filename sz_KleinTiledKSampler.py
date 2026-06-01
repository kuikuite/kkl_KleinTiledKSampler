"""
SZ KleinTiled KSampler
----------------------
适用于 FLUX.2 Klein 模型的分块采样器。
主要用途：图像放大修复和细节增强。

核心功能：
  1. 外部接入 latent_blend 作为全局引导
  2. 生成空间连续的全局噪声图
  3. 分块对应采样（相同尺寸的 tile 两两并行，加速约40-50%）
  4. overlap 羽化混合写回
  5. 自动对齐原始图像的色彩统计量，防止饱和度漂移
  6. 可选 face_mask：人脸区域单独规划更细 tile，再与非人脸区域按同样羽化逻辑合并
"""

import torch
import torch.nn.functional as F
import comfy.samplers
import comfy.sample
import comfy.model_management
import comfy.utils
import latent_preview


class SZ_KleinFaceMaskTiledKSampler:

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
                    "default": 512, "min": 64, "max": 2048, "step": 8
                }),
                "tile_height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 8
                }),
                "overlap": ("INT", {
                    "default": 128, "min": 0, "max": 512, "step": 8
                }),
                "face_tile_width": ("INT", {
                    "default": 384, "min": 64, "max": 2048, "step": 8
                }),
                "face_tile_height": ("INT", {
                    "default": 384, "min": 64, "max": 2048, "step": 8
                }),
                "face_overlap": ("INT", {
                    "default": 192, "min": 0, "max": 1024, "step": 8
                }),
                "face_padding": ("FLOAT", {
                    "default": 1.35, "min": 1.0, "max": 3.0, "step": 0.05
                }),
                "face_mask_threshold": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "color_preserve": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05
                }),
            },
            "optional": {
                "face_mask":     ("MASK",),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "sample"
    CATEGORY      = "SZ"

    # ──────────────────────────────────────────────────────────────────────

    def _get_tile_positions(self, H, W, tile_h, tile_w, overlap):
        tile_h = min(tile_h, H)
        tile_w = min(tile_w, W)
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

    def _make_weight_mask(self, h, w, device):
        wy = torch.arange(h, dtype=torch.float32, device=device)
        wy = (torch.min(wy, h - 1 - wy) + 1.0)
        wx = torch.arange(w, dtype=torch.float32, device=device)
        wx = (torch.min(wx, w - 1 - wx) + 1.0)
        weight = (wy.unsqueeze(1) * wx.unsqueeze(0))
        weight = weight / weight.max()
        return weight.unsqueeze(0).unsqueeze(0)

    def _prepare_face_mask(self, face_mask, H, W, B, device):
        """把 ComfyUI MASK 统一成 latent 空间的 (B,1,H,W) 软 mask。"""
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

    def _get_face_tile_positions(self, face_mask, H, W, tile_h, tile_w, overlap,
                                 padding, threshold=0.2):
        """根据 face_mask 的有效区域生成一组人脸 tile，内部仍使用贴边去重的分块逻辑。"""
        if face_mask is None:
            return []
        mask = face_mask
        if mask.dim() == 4:
            mask = mask[:, 0]
        if mask.dim() == 3:
            mask = mask.max(dim=0).values
        active = mask > threshold
        if not bool(active.any()):
            return []

        ys, xs = active.nonzero(as_tuple=True)
        y_min = int(ys.min().item())
        y_max = int(ys.max().item()) + 1
        x_min = int(xs.min().item())
        x_max = int(xs.max().item()) + 1
        box_h = max(1, y_max - y_min)
        box_w = max(1, x_max - x_min)
        pad_h = int(round((padding - 1.0) * box_h / 2.0))
        pad_w = int(round((padding - 1.0) * box_w / 2.0))
        y_min = max(0, y_min - pad_h)
        y_max = min(H, y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(W, x_max + pad_w)

        region_h = y_max - y_min
        region_w = x_max - x_min
        local_tiles = self._get_tile_positions(region_h, region_w, tile_h, tile_w, overlap)
        return [(y_min + y0, x_min + x0, th, tw) for (y0, x0, th, tw) in local_tiles]

    def _accumulate_tile(self, result, weight_map, tile_result, y0, x0, th, tw,
                         weight, region_mask=None):
        if region_mask is not None:
            weight = weight * region_mask[:, :, y0:y0+th, x0:x0+tw]
        result[:, :, y0:y0+th, x0:x0+tw] += tile_result * weight
        weight_map[:, :, y0:y0+th, x0:x0+tw] += weight

    def _process_and_accumulate_tiles(self, model, positive, negative, samples,
                                      global_noise, blend_up, tile_positions,
                                      result, weight_map, region_mask,
                                      B, blend_strength,
                                      steps, cfg, sampler_name, scheduler,
                                      denoise, seed, previewer, device,
                                      outer_pbar, progress_offset=0):
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
                                           progress_offset + total_tiles, None)
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
                                           progress_offset + total_tiles, None)
                idx += 1

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
        )
        return tile_result.to(device)

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

        return pair_result[:B], pair_result[B:]

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

    # ──────────────────────────────────────────────────────────────────────

    def sample(self, model, positive, negative, latent_image, latent_blend,
               seed, steps, cfg, sampler_name, scheduler, denoise,
               tile_width, tile_height, overlap,
               face_tile_width, face_tile_height, face_overlap,
               face_padding, face_mask_threshold,
               blend_strength, color_preserve,
               face_mask=None):

        device  = comfy.model_management.get_torch_device()
        samples = latent_image["samples"].clone().to(device)
        B, C, H, W = samples.shape

        tile_h = max(1, tile_height // 8)
        tile_w = max(1, tile_width  // 8)
        ovlp   = max(1, overlap     // 8)
        face_tile_h = max(1, face_tile_height // 8)
        face_tile_w = max(1, face_tile_width  // 8)
        face_ovlp   = max(1, face_overlap     // 8)

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

        # ── 规划 tile，按内容排序 ─────────────────────────────────────────
        face_mask_up = self._prepare_face_mask(face_mask, H, W, B, device)
        tile_positions = self._get_tile_positions(H, W, tile_h, tile_w, ovlp)
        tile_positions = self._sort_tiles_by_content(tile_positions, blend_up)
        face_positions = self._get_face_tile_positions(
            face_mask_up, H, W, face_tile_h, face_tile_w, face_ovlp,
            face_padding, face_mask_threshold
        )
        face_positions = self._sort_tiles_by_content(face_positions, blend_up) if face_positions else []
        total_tiles    = len(tile_positions) + len(face_positions)
        if face_positions:
            print(f"[SZ_KleinFaceMaskTiledKSampler] 普通区域 {len(tile_positions)} 个 tile，人脸区域 {len(face_positions)} 个 tile")
        else:
            print(f"[SZ_KleinFaceMaskTiledKSampler] 共 {len(tile_positions)} 个 tile")

        # ── 分块对应采样（两两并行） ──────────────────────────────────────
        result     = torch.zeros((B, C, H, W), device=device)
        weight_map = torch.zeros((B, 1, H, W), device=device)
        outer_pbar = comfy.utils.ProgressBar(total_tiles)

        if face_positions and face_mask_up is not None:
            non_face_mask = (1.0 - face_mask_up).clamp(0.0, 1.0)
        else:
            non_face_mask = None

        self._process_and_accumulate_tiles(
            model, positive, negative, samples,
            global_noise, blend_up, tile_positions,
            result, weight_map, non_face_mask,
            B, blend_strength,
            steps, cfg, sampler_name, scheduler, denoise, seed,
            previewer, device, outer_pbar, 0
        )

        if face_positions and face_mask_up is not None:
            self._process_and_accumulate_tiles(
                model, positive, negative, samples,
                global_noise, blend_up, face_positions,
                result, weight_map, face_mask_up,
                B, blend_strength,
                steps, cfg, sampler_name, scheduler, denoise, seed,
                previewer, device, outer_pbar, len(tile_positions)
            )

        result = result / weight_map.clamp(min=1e-8)

        # ── 色彩统计对齐（防止饱和度漂移）────────────────────────────────
        if color_preserve > 0.0:
            result = self._match_color_stats(result, samples, color_preserve)

        return ({"samples": result.cpu()},)


# ──────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "SZ_KleinFaceMaskTiledKSampler": SZ_KleinFaceMaskTiledKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SZ_KleinFaceMaskTiledKSampler": "SZ KleinTiled KSampler (Face Mask)",
}
