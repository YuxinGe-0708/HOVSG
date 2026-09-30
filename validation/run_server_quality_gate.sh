#!/usr/bin/env bash
set -o pipefail
cd /mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-multi-part
export PYTHONPATH=/mnt/data/huangyixuan/isaac/HOV-SG-graph-dev-multi-part
/mnt/data/huangyixuan/conda_envs/pfp_saliency_env/bin/python \
  application/build_graph_from_feature_map.py \
  --config-name real_robot \
  main.dataset_path=/mnt/data/huangyixuan/isaac/robot_recordings/multipart \
  main.split=room_orbbec_20260807_104332 \
  main.scene_id=part_01 \
  main.feature_map_path=/mnt/data/huangyixuan/isaac/hovsg_outputs/semantic/room_orbbec_20260807_104332_part01_decoupled/real_robot/part_01 \
  main.save_path=/mnt/data/huangyixuan/isaac/hovsg_outputs/semantic/room_orbbec_20260807_104332_ground_family_quality_gate \
  models.clip.checkpoint=/mnt/data/huangyixuan/checkpoints/hovsg/laion2b_s32b_b79k.bin \
  models.sam.checkpoint=/mnt/data/huangyixuan/checkpoints/hovsg/sam_vit_h_4b8939.pth \
  pipeline.create_graph=true \
  pipeline.create_navigation=true \
  pipeline.pose_mode=raw \
  pipeline.skip_sam_initialization=true
