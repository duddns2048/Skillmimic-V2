python skillmimic/run.py \
  --play_dataset \
  --task SkillMimic2BallPlay \
  --test \
  --num_envs 1 \
  --episode_length 1000 \
  --state_init 2 \
  --cfg_env skillmimic/data/cfg/skillmimic.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
  --motion_file skillmimic/data/motions/BallPlay-Pick