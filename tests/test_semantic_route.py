import json
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import open3d as o3d

from hovsg.utils.semantic_route import (
    load_navigation_graph,
    object_position_from_metadata,
    route_to_ranked_targets,
)


class SemanticRouteTest(unittest.TestCase):
    def make_graph(self):
        graph = nx.Graph()
        graph.add_node((0.0, 0.0, 0), pos=(0.0, 0.0, 0.0))
        graph.add_node((0.0, 1.0, 0), pos=(1.0, 0.0, 0.0))
        graph.add_node((0.0, 2.0, 0), pos=(2.0, 0.0, 0.0))
        graph.add_edge((0.0, 0.0, 0), (0.0, 1.0, 0), dist=1.0)
        graph.add_edge((0.0, 1.0, 0), (0.0, 2.0, 0), dist=1.0)
        return graph

    def test_load_and_route_to_ranked_target(self):
        graph = self.make_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            with path.open("w", encoding="utf-8") as file:
                json.dump(nx.node_link_data(graph), file)
            loaded = load_navigation_graph(path)

        result = route_to_ranked_targets(
            loaded,
            [0.1, 0.0, 0.0],
            [{"object_id": "chair", "position": [1.9, 0.0, 0.0], "score": 0.8}],
        )

        self.assertEqual(result["target"]["object_id"], "chair")
        self.assertEqual(len(result["route_positions"]), 3)
        self.assertAlmostEqual(result["route_length_m"], 2.0)

    def test_empty_graph_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text('{"nodes": [], "links": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_navigation_graph(path)

    def test_object_position_loads_historical_2d_vertices_from_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "object.json"
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(
                np.asarray([[1.0, 0.2, 3.0], [3.0, 0.8, 5.0]])
            )
            o3d.io.write_point_cloud(str(metadata_path.with_suffix(".ply")), cloud)

            position = object_position_from_metadata(
                metadata_path,
                [[1.0, 3.0], [3.0, 5.0]],
            )

        np.testing.assert_allclose(position, [2.0, 0.5, 4.0])


if __name__ == "__main__":
    unittest.main()
