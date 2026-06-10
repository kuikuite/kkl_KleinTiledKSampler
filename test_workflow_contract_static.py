import json
import unittest
from pathlib import Path


class WorkflowContractStaticTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("Klein-高清放大-SZ分块采样器.json")
        self.workflow = json.loads(path.read_text(encoding="utf-8"))

    def test_workflow_uses_face_region_vae_nodes(self):
        node_types = {node["type"] for node in self.workflow["nodes"]}

        self.assertNotIn("VAEEncodeTiled", node_types)
        self.assertNotIn("VAEDecodeTiled", node_types)
        self.assertIn("SZ_KleinFaceRegionVAEEncode", node_types)
        self.assertIn("SZ_KleinFaceRegionVAEDecode", node_types)

    def test_scaling_nodes_round_to_16_for_klein_alignment(self):
        scale_nodes = [
            node for node in self.workflow["nodes"]
            if node["type"] == "LayerUtility: ImageScaleByAspectRatio V2"
        ]

        self.assertTrue(scale_nodes)
        self.assertTrue(all(node["widgets_values"][5] == "16" for node in scale_nodes))

    def test_sampler_has_face_region_widget_values(self):
        samplers = [
            node for node in self.workflow["nodes"]
            if node["type"] == "SZ_KleinTiledKSampler"
        ]

        self.assertTrue(samplers)
        for sampler in samplers:
            self.assertGreaterEqual(len(sampler["widgets_values"]), 19)

    def test_sampler_widget_input_order_matches_node_signature(self):
        sampler = next(
            node for node in self.workflow["nodes"]
            if node["type"] == "SZ_KleinTiledKSampler"
        )
        widget_inputs = [
            item["name"] for item in sampler["inputs"]
            if "widget" in item
        ]

        self.assertEqual(widget_inputs, [
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "tile_width",
            "tile_height",
            "overlap",
            "face_tile_width",
            "face_tile_height",
            "face_overlap",
            "face_padding",
            "face_mask_threshold",
            "face_mask_grow",
            "face_mask_blur",
            "blend_strength",
            "color_preserve",
        ])

    def test_face_mask_sockets_are_visible_in_workflow(self):
        for node in self.workflow["nodes"]:
            if node["type"] in {
                "SZ_KleinTiledKSampler",
                "SZ_KleinFaceRegionVAEEncode",
                "SZ_KleinFaceRegionVAEDecode",
            }:
                input_names = [item["name"] for item in node["inputs"]]
                self.assertIn("face_mask", input_names)


if __name__ == "__main__":
    unittest.main()
