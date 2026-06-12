import json
import unittest
from pathlib import Path


class WorkflowContractStaticTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("Klein-高清放大-SZ分块采样器.json")
        self.workflow = json.loads(path.read_text(encoding="utf-8"))

    def load_dynamic_workflow(self):
        path = Path(__file__).with_name("Klein-高清放大-SZ动态人脸mask分块工作流.json")
        return json.loads(path.read_text(encoding="utf-8"))

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

    def test_dynamic_workflow_connects_external_face_mask_to_sz_nodes(self):
        workflow = self.load_dynamic_workflow()
        link_ids = {link[0] for link in workflow["links"]}

        for link_id in (1811, 1812, 1813, 1814, 1815, 1816, 1817):
            self.assertIn(link_id, link_ids)

        target_links = {
            (700, "face_mask"): 1813,
            (703, "face_mask"): 1814,
            (699, "face_mask"): 1815,
            (702, "face_mask"): 1816,
            (707, "face_mask"): 1817,
        }
        for (node_id, input_name), expected_link in target_links.items():
            node = next(item for item in workflow["nodes"] if item["id"] == node_id)
            actual_link = next(item for item in node["inputs"] if item["name"] == input_name)["link"]
            self.assertEqual(actual_link, expected_link)

    def test_dynamic_workflow_defaults_soften_holey_face_masks(self):
        workflow = self.load_dynamic_workflow()
        for node in workflow["nodes"]:
            values = node.get("widgets_values", [])
            if node["type"] in {"SZ_KleinFaceRegionVAEEncode", "SZ_KleinFaceRegionVAEDecode"}:
                self.assertEqual(values[2], 256)
                self.assertEqual(values[6], 64)
                self.assertEqual(values[7], 64)
            elif node["type"] == "SZ_KleinTiledKSampler":
                self.assertEqual(values[10], 256)
                self.assertEqual(values[11], 256)
                self.assertEqual(values[15], 64)
                self.assertEqual(values[16], 64)


if __name__ == "__main__":
    unittest.main()
