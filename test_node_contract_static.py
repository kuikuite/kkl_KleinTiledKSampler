import importlib.util
import inspect
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


class NodeContractStaticTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_sampler_exposes_face_region_controls_and_optional_mask(self):
        input_types = self.module.SZ_KleinTiledKSampler.INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]

        for name in (
            "face_tile_width",
            "face_tile_height",
            "face_overlap",
            "face_padding",
            "face_mask_threshold",
            "face_mask_grow",
            "face_mask_blur",
        ):
            self.assertIn(name, required)

        self.assertIn("face_mask", optional)
        self.assertEqual(required["face_tile_width"][1]["max"], 2048)
        self.assertEqual(required["tile_width"][1]["max"], 2048)

    def test_sampler_sample_signature_accepts_face_region_controls(self):
        params = inspect.signature(self.module.SZ_KleinTiledKSampler.sample).parameters

        for name in (
            "face_tile_width",
            "face_tile_height",
            "face_overlap",
            "face_padding",
            "face_mask_threshold",
            "face_mask_grow",
            "face_mask_blur",
            "face_mask",
        ):
            self.assertIn(name, params)

    def test_face_region_vae_nodes_are_registered(self):
        self.assertIn("SZ_KleinFaceRegionVAEEncode", self.module.NODE_CLASS_MAPPINGS)
        self.assertIn("SZ_KleinFaceRegionVAEDecode", self.module.NODE_CLASS_MAPPINGS)
        self.assertEqual(
            self.module.NODE_DISPLAY_NAME_MAPPINGS["SZ_KleinFaceRegionVAEEncode"],
            "SZ Klein Face Region VAE Encode",
        )
        self.assertEqual(
            self.module.NODE_DISPLAY_NAME_MAPPINGS["SZ_KleinFaceRegionVAEDecode"],
            "SZ Klein Face Region VAE Decode",
        )

    def test_package_init_exports_all_node_mappings(self):
        package_init = Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")

        self.assertIn("NODE_CLASS_MAPPINGS", package_init)
        self.assertIn("NODE_DISPLAY_NAME_MAPPINGS", package_init)
        self.assertIn("sz_KleinTiledKSampler", package_init)

    def test_face_region_vae_nodes_accept_mask_and_alignment_controls(self):
        encode_required = self.module.SZ_KleinFaceRegionVAEEncode.INPUT_TYPES()["required"]
        encode_optional = self.module.SZ_KleinFaceRegionVAEEncode.INPUT_TYPES()["optional"]
        decode_required = self.module.SZ_KleinFaceRegionVAEDecode.INPUT_TYPES()["required"]
        decode_optional = self.module.SZ_KleinFaceRegionVAEDecode.INPUT_TYPES()["optional"]

        self.assertIn("pixels", encode_required)
        self.assertIn("samples", decode_required)
        for required in (encode_required, decode_required):
            self.assertIn("tile_size", required)
            self.assertIn("face_tile_size", required)
            self.assertEqual(required["tile_size"][1]["max"], 2048)
            self.assertEqual(required["face_tile_size"][1]["max"], 2048)
        self.assertIn("face_mask", encode_optional)
        self.assertIn("face_mask", decode_optional)

    def test_face_mask_defaults_are_soft_enough_for_holey_person_masks(self):
        sampler_required = self.module.SZ_KleinTiledKSampler.INPUT_TYPES()["required"]
        encode_required = self.module.SZ_KleinFaceRegionVAEEncode.INPUT_TYPES()["required"]
        decode_required = self.module.SZ_KleinFaceRegionVAEDecode.INPUT_TYPES()["required"]

        self.assertEqual(sampler_required["face_tile_width"][1]["default"], 768)
        self.assertEqual(sampler_required["face_tile_height"][1]["default"], 768)
        for required in (sampler_required, encode_required, decode_required):
            if "face_tile_size" in required:
                self.assertEqual(required["face_tile_size"][1]["default"], 768)
            self.assertEqual(required["face_mask_grow"][1]["default"], 64)
            self.assertEqual(required["face_mask_blur"][1]["default"], 64)

    def test_sampler_uses_shared_face_aware_tile_plan(self):
        source = inspect.getsource(self.module.SZ_KleinTiledKSampler.sample)

        self.assertIn("_plan_face_aware_tiles_from_mask", source)
        self.assertIn("background_positions", source)
        self.assertIn("face_tiles", source)

    def test_vae_nodes_use_shared_face_aware_tile_plan(self):
        encode_source = inspect.getsource(self.module.SZ_KleinFaceRegionVAEEncode.encode)
        decode_source = inspect.getsource(self.module.SZ_KleinFaceRegionVAEDecode.decode)

        self.assertIn("_plan_face_aware_tiles_from_mask", encode_source)
        self.assertIn("background_tiles", encode_source)
        self.assertIn("face_tiles", encode_source)
        self.assertIn("_plan_face_aware_tiles_from_mask", decode_source)
        self.assertIn("background_tiles", decode_source)
        self.assertIn("face_tiles", decode_source)

    def test_face_paths_use_upscaled_processing_canvas(self):
        sampler_source = "\n".join([
            inspect.getsource(self.module.SZ_KleinTiledKSampler.sample),
            inspect.getsource(self.module.SZ_KleinTiledKSampler._process_face_tile),
            inspect.getsource(self.module.SZ_KleinTiledKSampler._process_and_accumulate_tiles),
            inspect.getsource(self.module.SZ_KleinTiledKSampler._process_and_accumulate_face_tiles),
        ])
        encode_source = "\n".join([
            inspect.getsource(self.module.SZ_KleinFaceRegionVAEEncode.encode),
            inspect.getsource(self.module.SZ_KleinFaceRegionVAEEncode._accumulate_encoded_face_tile),
        ])
        decode_source = "\n".join([
            inspect.getsource(self.module.SZ_KleinFaceRegionVAEDecode.decode),
            inspect.getsource(self.module.SZ_KleinFaceRegionVAEDecode._accumulate_decoded_face_tile),
        ])

        self.assertIn("_process_face_tile", sampler_source)
        self.assertIn("_process_and_accumulate_face_tiles", sampler_source)
        self.assertIn("process_content_latent_rect", sampler_source)
        self.assertIn("_face_conditioning_refs", sampler_source)
        self.assertIn("_accumulate_encoded_face_tile", encode_source)
        self.assertIn("process_content_rect", encode_source)
        self.assertIn("_accumulate_decoded_face_tile", decode_source)
        self.assertIn("process_latent_size", decode_source)
        self.assertIn("process_content_rect", decode_source)

    def test_background_tiles_are_not_masked_out_inside_face_bbox(self):
        sampler_source = inspect.getsource(self.module.SZ_KleinTiledKSampler.sample)
        encode_source = inspect.getsource(self.module.SZ_KleinFaceRegionVAEEncode.encode)
        decode_source = inspect.getsource(self.module.SZ_KleinFaceRegionVAEDecode.decode)

        self.assertIn("background_region_mask = None", sampler_source)
        self.assertIn("background_region_mask = None", encode_source)
        self.assertIn("background_region_mask = None", decode_source)
        self.assertNotIn("non_face_mask = (1.0 - face_mask_soft)", sampler_source)
        self.assertNotIn("non_face_mask = (1.0 - face_mask_soft)", encode_source)
        self.assertNotIn("non_face_mask = (1.0 - face_mask_soft)", decode_source)


if __name__ == "__main__":
    unittest.main()
