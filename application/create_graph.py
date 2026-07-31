import os
import hydra
from omegaconf import DictConfig
from hovsg.graph.graph import Graph
from hovsg.utils.runtime_status import print_status

# pylint: disable=all


@hydra.main(version_base=None, config_path="../config", config_name="create_graph")
def main(params: DictConfig):
    # create logging directory
    save_dir = os.path.join(params.main.save_path, params.main.dataset, params.main.scene_id)
    params.main.save_path = save_dir
    params.main.dataset_path = os.path.join(params.main.dataset_path, params.main.split, params.main.scene_id)
    print_status(
        "PIPELINE",
        "START",
        dataset=params.main.dataset,
        scene=params.main.scene_id,
        input_path=params.main.dataset_path,
        output_path=save_dir,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    print_status("OUTPUT_DIRECTORY", "OK", path=save_dir, exists=os.path.isdir(save_dir))

    # create graph
    hovsg = Graph(params)
    print_status("GRAPH_INITIALIZATION", "OK", dataset_type=type(hovsg.dataset).__name__)
    hovsg.create_feature_map() # create feature map
    print_status(
        "FEATURE_MAP",
        "OK",
        full_points=len(hovsg.full_pcd.points),
        object_masks=len(hovsg.mask_pcds),
        mask_features=len(hovsg.mask_feats),
    )

    # save full point cloud, features, and masked point clouds (pcd for all objects)
    hovsg.save_masked_pcds(path=save_dir, state="both")
    hovsg.save_full_pcd(path=save_dir)
    hovsg.save_full_pcd_feats(path=save_dir)
    print_status("FEATURE_OUTPUTS", "OK", path=save_dir)
    
    # for debugging: load preconstructed map as follows
    # hovsg.load_full_pcd(path=save_dir)
    # hovsg.load_full_pcd_feats(path=save_dir)
    # hovsg.load_masked_pcds(path=save_dir)
    
    # create graph, only if dataset is not Replia or ScanNet
    print(params.main.dataset)
    if params.main.dataset != "replica" and params.main.dataset != "scannet" and params.pipeline.create_graph:
        hovsg.build_graph(save_path=save_dir)
        print_status(
            "HIERARCHICAL_GRAPH",
            "OK",
            floors=len(hovsg.floors),
            rooms=len(hovsg.rooms),
            objects=len(hovsg.objects),
            path=os.path.join(save_dir, "graph"),
        )
    else:
        print("Skipping hierarchical scene graph creation for Replica and ScanNet datasets.")
        print_status(
            "HIERARCHICAL_GRAPH",
            "SKIP",
            create_graph=params.pipeline.create_graph,
            dataset=params.main.dataset,
        )

    print_status("PIPELINE", "DONE", output_path=save_dir)

if __name__ == "__main__":
    main()
