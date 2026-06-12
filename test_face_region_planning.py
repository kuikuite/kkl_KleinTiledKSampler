import importlib.util
import sys
import types
import unittest
from pathlib import Path

def load_module():
    torch = types.ModuleType("torch")
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.functional = types.ModuleType("torch.nn.functional")
    torch.float32 = object()
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    comfy = types.ModuleType("comfy")
    comfy.samplers = types.SimpleNamespace(
        KSampler=types.SimpleNamespace(SAMPLERS=("sampler",), SCHEDULERS=("scheduler",))
    )
    comfy.sample = types.SimpleNamespace()
    comfy.model_management = types.SimpleNamespace()
    comfy.utils = types.SimpleNamespace(ProgressBar=object)
    sys.modules["comfy"] = comfy
    sys.modules["comfy.samplers"] = comfy.samplers
    sys.modules["comfy.sample"] = comfy.sample
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.utils"] = comfy.utils
    sys.modules["latent_preview"] = types.SimpleNamespace()

    path = Path(__file__).with_name("sz_KleinTiledKSampler.py")
    spec = importlib.util.spec_from_file_location("sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FaceRegionPlanningTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.sampler = self.module.SZ_KleinTiledKSampler()

    def test_image_tile_values_are_capped_and_aligned_before_latent_conversion(self):
        self.assertEqual(self.sampler._image_to_latent_tile(4096), 256)
        self.assertEqual(self.sampler._image_to_latent_tile(770), 96)
        self.assertEqual(self.sampler._image_to_latent_tile(63), 8)

    def test_no_mask_returns_no_face_tiles(self):
        self.assertEqual(
            self.sampler._get_face_tile_positions(None, 64, 64, 16, 16, 4, 1.35, 0.2),
            [],
        )

    def test_bbox_expands_clamps_and_generates_16_aligned_face_tiles(self):
        tiles = self.sampler._get_face_tile_positions_from_bbox(
            bbox=(33, 55, 41, 70),
            H=128,
            W=128,
            tile_h=32,
            tile_w=32,
            overlap=8,
            padding=1.35,
        )

        self.assertTrue(tiles)
        self.assertTrue(all(y % 2 == 0 and x % 2 == 0 for y, x, h, w in tiles))
        self.assertTrue(all(h == 32 and w == 32 for y, x, h, w in tiles))
        self.assertGreaterEqual(min(y for y, x, h, w in tiles), 0)
        self.assertGreaterEqual(min(x for y, x, h, w in tiles), 0)
        self.assertLessEqual(max(y + h for y, x, h, w in tiles), 128)
        self.assertLessEqual(max(x + w for y, x, h, w in tiles), 128)
        self.assertTrue(any(y <= 33 and x <= 41 and y + h >= 55 and x + w >= 70 for y, x, h, w in tiles))

    def test_region_planning_clamps_large_face_tiles_to_klein_limit(self):
        tiles = self.sampler._get_face_tile_positions_from_bbox(
            bbox=(100, 700, 120, 900),
            H=3000,
            W=3000,
            tile_h=self.sampler._image_to_latent_tile(4096),
            tile_w=self.sampler._image_to_latent_tile(4096),
            overlap=self.sampler._image_to_latent_overlap(256),
            padding=1.2,
        )

        self.assertTrue(tiles)
        self.assertTrue(all(h <= 256 and w <= 256 for y, x, h, w in tiles))

    def test_face_plan_uses_full_background_underlay_plus_face_overlay(self):
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
        self.assertEqual(len(background_tiles), 9)
        self.assertTrue(all(tile["source_region"] == "full" for tile in background_tiles))
        self.assertTrue(all(tile["image_rect"][0] % 16 == 0 for tile in plan))
        self.assertTrue(all(tile["image_rect"][1] % 16 == 0 for tile in plan))
        covered_columns = sorted({tile["image_rect"][1] for tile in background_tiles})
        covered_rows = sorted({tile["image_rect"][0] for tile in background_tiles})
        self.assertEqual(covered_columns, [0, 384, 768])
        self.assertEqual(covered_rows, [0, 384, 512])

    def test_face_plan_keeps_full_background_underlay_to_avoid_mask_holes(self):
        plan = self.sampler._plan_face_aware_tiles_from_bbox(
            bbox=(320, 704, 448, 832),
            image_h=1024,
            image_w=1280,
            tile_h=2048,
            tile_w=2048,
            overlap=128,
            face_tile_h=384,
            face_tile_w=384,
            face_overlap=192,
            face_padding=1.0,
            latent_downscale=16,
        )

        background_tiles = [tile for tile in plan if tile["kind"] == "background"]
        self.assertEqual(len(background_tiles), 1)
        self.assertEqual(background_tiles[0]["image_rect"], (0, 0, 1024, 1280))
        self.assertEqual(background_tiles[0]["latent_rect"], (0, 0, 64, 80))

    def test_small_face_mask_does_not_expand_to_full_image_when_face_tile_is_large(self):
        plan = self.sampler._plan_face_aware_tiles_from_bbox(
            bbox=(184, 240, 224, 280),
            image_h=512,
            image_w=512,
            tile_h=512,
            tile_w=512,
            overlap=128,
            face_tile_h=768,
            face_tile_w=768,
            face_overlap=192,
            face_padding=1.35,
            latent_downscale=16,
        )

        face_tiles = [tile for tile in plan if tile["kind"] == "face"]
        self.assertEqual(len(face_tiles), 1)
        self.assertLessEqual(face_tiles[0]["image_rect"][2], 112)
        self.assertLessEqual(face_tiles[0]["image_rect"][3], 112)
        y0, x0, h, w = face_tiles[0]["image_rect"]
        self.assertLessEqual(y0, 184)
        self.assertLessEqual(x0, 224)
        self.assertGreaterEqual(y0 + h, 240)
        self.assertGreaterEqual(x0 + w, 280)

    def test_small_face_tile_records_upscaled_processing_canvas(self):
        plan = self.sampler._plan_face_aware_tiles_from_bbox(
            bbox=(184, 240, 224, 300),
            image_h=512,
            image_w=512,
            tile_h=512,
            tile_w=512,
            overlap=128,
            face_tile_h=768,
            face_tile_w=768,
            face_overlap=192,
            face_padding=1.35,
            latent_downscale=16,
        )

        face_tile = next(tile for tile in plan if tile["kind"] == "face")

        self.assertEqual(face_tile["process_image_size"], (768, 768))
        self.assertEqual(face_tile["process_latent_size"], (48, 48))
        py0, px0, ph, pw = face_tile["process_content_rect"]
        self.assertEqual(px0, 0)
        self.assertEqual(pw, 768)
        self.assertGreater(py0, 0)
        self.assertLess(py0 + ph, 768)
        self.assertEqual(ph % 16, 0)
        self.assertEqual(pw % 16, 0)

    def test_background_band_that_fits_tile_stays_single_tile(self):
        plan = self.sampler._split_background_around_face_region(
            face_region=(256, 768, 256, 768),
            image_h=1024,
            image_w=1024,
            tile_h=512,
            tile_w=2048,
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

    def test_plan_uses_supplied_latent_downscale_for_latent_rects(self):
        plan = self.sampler._plan_face_aware_tiles_from_bbox(
            bbox=None,
            image_h=1024,
            image_w=2048,
            tile_h=2048,
            tile_w=2048,
            overlap=128,
            face_tile_h=768,
            face_tile_w=768,
            face_overlap=192,
            face_padding=1.35,
            latent_downscale=16,
        )

        self.assertEqual(plan[0]["image_rect"], (0, 0, 1024, 2048))
        self.assertEqual(plan[0]["latent_rect"], (0, 0, 64, 128))

    def test_decoded_tile_accumulation_uses_actual_vae_decode_scale(self):
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def __getitem__(self, key):
                return self

            def __setitem__(self, key, value):
                return None

            def __iadd__(self, other):
                return self

            def __mul__(self, other):
                return self

            def movedim(self, source, destination):
                return self

        class FakeImage(FakeTensor):
            def to(self, device=None):
                return self

        class FakeVAE:
            def decode(self, samples):
                return FakeImage((1, 1024, 2048, 3))

        torch = sys.modules["torch"]
        original_zeros = getattr(torch, "zeros", None)
        torch.zeros = lambda shape, device=None: FakeTensor(shape)
        original_make_image_weight_mask = self.module.SZ_KleinFaceRegionVAEDecode._make_image_weight_mask
        self.module.SZ_KleinFaceRegionVAEDecode._make_image_weight_mask = (
            lambda self, h, w, device: FakeTensor((1, h, w, 1))
        )
        try:
            result, weight_map = self.module.SZ_KleinFaceRegionVAEDecode()._accumulate_decoded_tile(
                FakeVAE(),
                FakeTensor((1, 16, 64, 128)),
                0,
                0,
                64,
                128,
                (None, None),
                None,
                "cpu",
                8,
            )
        finally:
            if original_zeros is None:
                delattr(torch, "zeros")
            else:
                torch.zeros = original_zeros
            self.module.SZ_KleinFaceRegionVAEDecode._make_image_weight_mask = (
                original_make_image_weight_mask
            )

        self.assertEqual(result.shape, (1, 1024, 2048, 3))
        self.assertEqual(weight_map.shape, (1, 1024, 2048, 1))


if __name__ == "__main__":
    unittest.main()
