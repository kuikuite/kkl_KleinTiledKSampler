import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


def load_sampler_class():
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
    return module.SZ_KleinFaceMaskTiledKSampler


class FaceMaskPlanningTest(unittest.TestCase):
    def setUp(self):
        self.sampler = load_sampler_class()()

    def test_no_mask_returns_no_face_tiles(self):
        self.assertEqual(self.sampler._get_face_tile_positions(None, 64, 64, 16, 16, 4, 2), [])

    def test_mask_bbox_expands_and_clamps_to_latent_size(self):
        mask = torch.zeros((1, 80, 80))
        mask[:, 20:36, 30:46] = 1

        tiles = self.sampler._get_face_tile_positions(mask, 80, 80, 16, 16, 4, 1.5)

        self.assertTrue(tiles)
        ys = [tile[0] for tile in tiles]
        xs = [tile[1] for tile in tiles]
        self.assertGreaterEqual(min(ys), 0)
        self.assertGreaterEqual(min(xs), 0)
        self.assertLessEqual(max(y + h for y, x, h, w in tiles), 80)
        self.assertLessEqual(max(x + w for y, x, h, w in tiles), 80)
        self.assertTrue(any(y <= 20 and x <= 30 and y + h >= 36 and x + w >= 46 for y, x, h, w in tiles))

    def test_face_tiles_use_same_pinned_grid_logic_inside_face_region(self):
        mask = torch.zeros((1, 40, 40))
        mask[:, 10:30, 10:30] = 1

        tiles = self.sampler._get_face_tile_positions(mask, 40, 40, 12, 12, 4, 1.0)

        self.assertEqual(len(set((y, x) for y, x, h, w in tiles)), len(tiles))
        self.assertTrue(all(h == 12 and w == 12 for y, x, h, w in tiles))


if __name__ == "__main__":
    unittest.main()
