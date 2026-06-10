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
        self.assertEqual(len(background_tiles), 8)
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


if __name__ == "__main__":
    unittest.main()
