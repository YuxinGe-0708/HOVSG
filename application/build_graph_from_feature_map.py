import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict

from hovsg.graph.graph import Graph
from hovsg.utils.runtime_status import print_status


@hydra.main(version_base=None, config_path="../config", config_name="create_graph")
def main(params: DictConfig):
    """Build the hierarchy/navigation graph from saved feature-map outputs."""
    save_dir = Path(
        os.path.join(
            str(params.main.save_path),
            str(params.main.dataset),
            str(params.main.scene_id),
        )
    )
    feature_map_value = params.main.get("feature_map_path", None)
    feature_map_dir = (
        Path(str(feature_map_value)) if feature_map_value else save_dir
    )
    dataset_dir = Path(
        os.path.join(
            str(params.main.dataset_path),
            str(params.main.split),
            str(params.main.scene_id),
        )
    )

    params.main.save_path = str(save_dir)
    params.main.dataset_path = str(dataset_dir)
    with open_dict(params.pipeline):
        params.pipeline.skip_sam_initialization = True

    required_paths = [
        feature_map_dir / "full_pcd.ply",
        feature_map_dir / "mask_feats.pt",
        feature_map_dir / "objects",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Saved feature map is incomplete; missing: " + ", ".join(missing_paths)
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    print_status(
        "GRAPH_RESUME",
        "START",
        dataset_path=dataset_dir,
        feature_map_path=feature_map_dir,
        output_path=save_dir,
        create_navigation=bool(params.pipeline.get("create_navigation", True)),
    )

    hovsg = Graph(params)
    hovsg.load_full_pcd(str(feature_map_dir))
    hovsg.load_full_pcd_feats(str(feature_map_dir), full_feats=False)
    hovsg.load_masked_pcds(str(feature_map_dir))

    if hovsg.full_pcd.is_empty():
        raise RuntimeError("Loaded full_pcd.ply is empty")
    if not hovsg.mask_pcds:
        raise RuntimeError("No object point clouds were loaded")
    if len(hovsg.mask_pcds) != len(hovsg.mask_feats):
        raise RuntimeError(
            "Loaded object point clouds and mask features have different lengths: "
            f"pcds={len(hovsg.mask_pcds)}, feats={len(hovsg.mask_feats)}"
        )

    print_status(
        "FEATURE_MAP_LOAD",
        "OK",
        full_points=len(hovsg.full_pcd.points),
        object_masks=len(hovsg.mask_pcds),
        mask_features=len(hovsg.mask_feats),
        full_point_features_loaded=False,
    )

    hovsg.build_graph(save_path=str(save_dir))
    print_status(
        "GRAPH_RESUME",
        "DONE",
        floors=len(hovsg.floors),
        rooms=len(hovsg.rooms),
        objects=len(hovsg.objects),
        output_path=save_dir,
    )


if __name__ == "__main__":
    main()
