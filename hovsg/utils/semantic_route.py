"""Utilities for routing from a world position to a semantic object."""

import json
from pathlib import Path

import networkx as nx
import numpy as np


def load_navigation_graph(path):
    """Load a node-link JSON graph while preserving tuple-valued node IDs."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    graph = nx.Graph()
    for node in payload.get("nodes", []):
        attributes = dict(node)
        node_id = _node_id(attributes.pop("id"))
        if "pos" not in attributes:
            raise ValueError(f"Navigation node {node_id!r} has no world position")
        graph.add_node(node_id, **attributes)

    links = payload.get("links", payload.get("edges", []))
    for link in links:
        attributes = dict(link)
        source = _node_id(attributes.pop("source"))
        target = _node_id(attributes.pop("target"))
        if source not in graph or target not in graph:
            raise ValueError("Navigation edge references an unknown node")
        graph.add_edge(source, target, **attributes)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError(f"Navigation graph is empty: {path}")
    return graph


def nearest_node(graph, position, candidates=None):
    """Return the graph node nearest to a 3-D world position."""
    position = np.asarray(position, dtype=np.float64)
    nodes = list(graph.nodes if candidates is None else candidates)
    if not nodes:
        raise ValueError("No navigation nodes are available")
    positions = np.asarray([graph.nodes[node]["pos"] for node in nodes], dtype=np.float64)
    distances = np.linalg.norm(positions - position[None, :], axis=1)
    index = int(np.argmin(distances))
    return nodes[index], float(distances[index])


def route_to_ranked_targets(graph, start_position, ranked_targets):
    """Route to the first semantically ranked target reachable from the start."""
    start_node, start_snap_distance = nearest_node(graph, start_position)
    component = nx.node_connected_component(graph, start_node)

    for target in ranked_targets:
        target_node, target_snap_distance = nearest_node(
            graph, target["position"], candidates=component
        )
        try:
            path = nx.shortest_path(graph, start_node, target_node, weight="dist")
        except nx.NetworkXNoPath:
            continue
        route_positions = [list(map(float, graph.nodes[node]["pos"])) for node in path]
        route_length_m = float(
            sum(
                np.linalg.norm(np.asarray(end) - np.asarray(begin))
                for begin, end in zip(route_positions[:-1], route_positions[1:])
            )
        )
        return {
            "target": target,
            "start_node": list(start_node) if isinstance(start_node, tuple) else start_node,
            "target_node": list(target_node) if isinstance(target_node, tuple) else target_node,
            "start_snap_distance_m": start_snap_distance,
            "target_snap_distance_m": target_snap_distance,
            "route_length_m": route_length_m,
            "route_positions": route_positions,
            "route_node_ids": [
                list(node) if isinstance(node, tuple) else node for node in path
            ],
        }
    raise RuntimeError("None of the semantic target candidates is reachable")


def _node_id(value):
    return tuple(value) if isinstance(value, list) else value
