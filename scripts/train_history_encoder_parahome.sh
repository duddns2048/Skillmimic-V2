export PYTHONPATH=/home/wongyun/sim/SkillMimic-V2:$PYTHONPATH
python skillmimic/utils/state_prediction_parahome.py \
--motion_dir skillmimic/data/motions/ParaHome/ \
--batch_size 256 \
--history_length 60 \
--embedding_dim 3 \
--lr 0.001 \
--max_epochs 3000 \
--output_dir hist_encoder/ParaHome \
