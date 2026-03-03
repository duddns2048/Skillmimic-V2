import pickle
import torch
import numpy as np

# Load object transformations from InterMimic
with open('/home/wongyun/sim/InterMimic/intermimic/data/assets/parahome/seq/s110/object_transformations.pkl', 'rb') as f:
    obj_trans = pickle.load(f)

# Load motion PT file
motion_file = '/home/wongyun/sim/SkillMimic-V2/skillmimic/data/motions/ParaHome/place_kettle/003_s10_place_kettle_2580_2680.pt'
motion_data = torch.load(motion_file)

print("="*80)
print("OBJECT TRANSFORMATIONS STRUCTURE")
print("="*80)
print(f"Type: {type(obj_trans)}")
print(f"Length (num frames): {len(obj_trans)}")
print(f"\nKeys in frame 0: {list(obj_trans[0].keys())}")

# Check kettle position
if 'kettle_base' in obj_trans[0]:
    kettle_mat = obj_trans[0]['kettle_base']
    print(f"\nKettle transformation matrix at frame 0:")
    print(kettle_mat)
    print(f"\nKettle position from matrix [0,3], [1,3], [2,3]:")
    print(f"  X: {kettle_mat[0,3]}")
    print(f"  Y: {kettle_mat[1,3]}")
    print(f"  Z: {kettle_mat[2,3]}")
    print(f"\nKettle position from matrix [:3, 3]:")
    print(f"  {kettle_mat[:3, 3]}")

# Check diningtable position
if 'diningtable_base' in obj_trans[0]:
    table_mat = obj_trans[0]['diningtable_base']
    print(f"\n" + "="*80)
    print(f"Diningtable transformation matrix at frame 0:")
    print(table_mat)
    print(f"\nDiningtable position from matrix [0,3], [1,3], [2,3]:")
    print(f"  X: {table_mat[0,3]}")
    print(f"  Y: {table_mat[1,3]}")
    print(f"  Z: {table_mat[2,3]}")

print(f"\n" + "="*80)
print("MOTION FILE STRUCTURE (s10 scene)")
print("="*80)
print(f"Type: {type(motion_data)}")

if isinstance(motion_data, torch.Tensor):
    print(f"\nTensor shape: {motion_data.shape}")
    print(f"First frame root position (indices [0:3]):")
    print(f"  {motion_data[0, :3]}")

    # Object position should be at specific index
    # root_pos(3) + root_rot(3) + dof_pos(60*3) + joint_pos(71*3) = 3 + 3 + 180 + 213 = 399
    obj_start_idx = 3 + 3 + 180 + 213
    if motion_data.shape[1] > obj_start_idx + 3:
        print(f"\nObject position at indices [{obj_start_idx}:{obj_start_idx+3}] (first frame):")
        print(f"  {motion_data[0, obj_start_idx:obj_start_idx+3]}")
