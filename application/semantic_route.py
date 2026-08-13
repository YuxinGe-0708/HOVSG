"""Query saved HOV-SG objects and plan a route on a verified navigation graph."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch

from hovsg.dataloader.real_robot import RealRobotDataset
from hovsg.utils.clip_utils import get_text_feats_multiple_templates
from hovsg.utils.semantic_route import load_navigation_graph, route_to_ranked_targets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", choices=("first", "last"), default="last")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def load_ranked_objects(objects_dir, text_feature, top_k):
    candidates = []
    for metadata_path in sorted(objects_dir.glob("*.json")):
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        embedding = np.asarray(metadata.get("embedding", []), dtype=np.float32)
        vertices = np.asarray(metadata.get("vertices", []), dtype=np.float64)
        if embedding.ndim != 1 or embedding.size == 0 or vertices.shape != (8, 3):
            continue
        norm = float(np.linalg.norm(embedding))
        if norm <= 0 or not np.isfinite(norm) or not np.all(np.isfinite(vertices)):
            continue
        candidates.append(
            {
                "object_id": str(metadata["object_id"]),
                "room_id": str(metadata["room_id"]),
                "name": str(metadata.get("name", "")),
                "score": float(np.dot(text_feature, embedding / norm)),
                "position": vertices.mean(axis=0).tolist(),
            }
        )
    if not candidates:
        raise RuntimeError(f"No valid semantic objects found in {objects_dir}")
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]


def load_start_position(dataset_dir, start_frame):
    dataset = RealRobotDataset(
        {
            "root_dir": str(dataset_dir),
            "transforms": None,
            "min_depth_m": None,
            "max_depth_m": None,
            "pose_mode": "raw",
            "planar_pose_height_m": None,
        }
    )
    index = 0 if start_frame == "first" else -1
    pose = dataset.load_pose(dataset.data_list[index][2])
    return np.asarray(pose[:3, 3], dtype=np.float64)


def save_route_image(graph_dir, result, output_path):
    image_path = graph_dir / "nav_graph" / "top_down_rgb.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing navigation background: {image_path}")
    pixels = [np.rint(node[:2]).astype(np.int32) for node in result["route_node_ids"]]
    pixels = [(int(pixel[1]), int(pixel[0])) for pixel in pixels]
    for begin, end in zip(pixels[:-1], pixels[1:]):
        cv2.line(image, begin, end, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(image, pixels[0], 7, (255, 0, 0), -1)
    cv2.circle(image, pixels[-1], 7, (0, 255, 0), -1)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save route image: {output_path}")


def main():
    args = parse_args()
    nav_path = args.graph_dir / "nav_graph" / "global_nav_graph_graph.json"
    object_dir = args.graph_dir / "objects"
    if not nav_path.is_file():
        raise FileNotFoundError(
            f"Verified navigation graph not found: {nav_path}. "
            "Run graph construction with create_navigation=true and resolve any "
            "navigation safety rejection before planning a route."
        )
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Semantic object graph not found: {object_dir}")
    if not args.clip_checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {args.clip_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=str(args.clip_checkpoint), device=device
    )
    model.eval()
    text_feature = get_text_feats_multiple_templates([args.query], model, 1024)[0]
    ranked_targets = load_ranked_objects(object_dir, text_feature, args.top_k)
    graph = load_navigation_graph(nav_path)
    start_position = load_start_position(args.dataset_dir, args.start_frame)
    result = route_to_ranked_targets(graph, start_position, ranked_targets)
    result.update(
        {
            "query": args.query,
            "start_frame": args.start_frame,
            "start_position": start_position.tolist(),
            "navigation_nodes": graph.number_of_nodes(),
            "navigation_edges": graph.number_of_edges(),
            "ranked_targets": ranked_targets,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "route.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=True)
    save_route_image(args.graph_dir, result, args.output_dir / "route.png")
    print(
        "[HOVSG][SEMANTIC_ROUTE][OK] "
        f"query={args.query!r} object_id={result['target']['object_id']} "
        f"score={result['target']['score']:.4f} "
        f"route_length_m={result['route_length_m']:.3f} "
        f"waypoints={len(result['route_positions'])} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
