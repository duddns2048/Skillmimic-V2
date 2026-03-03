#!/bin/bash
# Inference script for testing trained ParaHome policies
# Usage: bash scripts/parahome_inference.sh [checkpoint_path]
# Example: bash scripts/parahome_inference.sh runs/SkillMimicParahome_00-00-00/nn/SkillMimicParahome.pth

# Default checkpoint path (change this to your trained checkpoint)
CHECKPOINT=${1:-"output/output_1120_0623/SkillMimic_17-04-50-41_kettle/nn/SkillMimic_e14696_r13.0225.pth"}

python skillmimic/run.py \
--task SkillMimicParahome \
--test \
--checkpoint "$CHECKPOINT" \
--num_envs 1 \
--episode_length 5000 \
--cfg_env skillmimic/data/cfg/parahome_sm/parahome.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/parahome.yaml \
--motion_file skillmimic/data/motions/ParaHome/wristmimic/s110_move_kettle \
--asset_file_name mjcf/parahome/s110.xml \
--in_scene_obj_static desk,diningtable \
--in_scene_obj_dynamic kettle \
--state_init_random_prob 0.1 \
--state_switch_prob 0 \
--hist_length 60 \
--history_embedding_size 3 \
--hist_ckpt hist_encoder/ParaHome/checkpoints/hist_model.ckpt \
--episode_length 600  \
--state_init 2