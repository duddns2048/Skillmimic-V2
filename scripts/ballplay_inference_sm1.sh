python skillmimic/run.py \
--test \
--task SkillMimic1BallPlay \
--num_envs 2 \
--cfg_env skillmimic/data/cfg/skillmimic_test.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file skillmimic/data/motions/BallPlay \
--state_init 2 \
--episode_length 600 \
--checkpoint models/BallPlay/SkillMimic/model.pth