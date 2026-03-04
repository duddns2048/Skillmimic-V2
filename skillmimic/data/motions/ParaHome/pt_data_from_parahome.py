######################################## Data Description ############################################
# bone_vectors: [body 22/lhand 24/rhand 24][njoints] ('pHipOrigin', 'jRightHip'): 0.09663602709770203
# body_global_transform.pkl (nframes, 4, 4)
# body_joint_orientations.pkl (nframes, 23, 6)
# joint_positions.pkl (nframes, 73, 3)
# joint_states.pkl [nobjects](nframes, 1) # 旋转部分用弧度表示(eg. laptop)，棱柱部分用米表示(draw)
# hand_joint_orientations.pkl (nframes, 40, 6)
# head_tips.pkl (nframes, 3)
# object_transformations.pkl [object_name] (nframes, 4, 4)
######################################################################################################

import os
import pickle
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


scene_number = 110
start_frame = 200
end_frame = 355
motion_name = 'move_chair'
data_pair = [
    {'scene_number': 11, 'scene_name': 'move_book', 'start_frame': 2640, 'end_frame': 2790},
    {'scene_number': 29, 'scene_name': 'move_pot', 'start_frame': 2200, 'end_frame': 2420},
    {'scene_number': 29, 'scene_name': 'move_kettle', 'start_frame': 4080, 'end_frame': 4240},
    {'scene_number': 79, 'scene_name': 'move_cup', 'start_frame': 150, 'end_frame': 330},
    {'scene_number': 79, 'scene_name': 'move_chair', 'start_frame': 330, 'end_frame': 551},
    {'scene_number': 79, 'scene_name': 'pour_kettle', 'start_frame': 660, 'end_frame': 840},
    {'scene_number': 79, 'scene_name': 'move_cup', 'start_frame': 1290, 'end_frame': 1440},
    {'scene_number': 89, 'scene_name': 'drink_cup', 'start_frame': 840, 'end_frame': 1209},
    {'scene_number': 98, 'scene_name': 'move_bowl', 'start_frame': 3372, 'end_frame': 3532},
    {'scene_number': 99, 'scene_name': 'move_cup', 'start_frame': 270, 'end_frame': 450},
    {'scene_number': 108, 'scene_name': 'move_kettle', 'start_frame': 3660, 'end_frame': 3840},
    {'scene_number': 110, 'scene_name': 'move_kettle', 'start_frame': 200, 'end_frame': 355},
    {'scene_number': 110, 'scene_name': 'pour_kettle', 'start_frame': 450, 'end_frame': 600},
    {'scene_number': 110, 'scene_name': 'drink_cup', 'start_frame': 3010, 'end_frame': 3210},
    {'scene_number': 113, 'scene_name': 'move_chair', 'start_frame': 1033, 'end_frame': 1300},
    {'scene_number': 114, 'scene_name': 'move_pan', 'start_frame': 2700, 'end_frame': 2880},
    {'scene_number': 117, 'scene_name': 'move_pan', 'start_frame': 4320, 'end_frame': 4530},
    {'scene_number': 117, 'scene_name': 'move_kettle', 'start_frame': 4860, 'end_frame': 5148},
    {'scene_number': 149, 'scene_name': 'move_book', 'start_frame': 3810, 'end_frame': 3930},
    {'scene_number': 184, 'scene_name': 'move_cup', 'start_frame': 240, 'end_frame': 420},]

idx = 3
scene_number = data_pair[idx]['scene_number']
motion_name = data_pair[idx]['scene_name']
start_frame = data_pair[idx]['start_frame']
end_frame = data_pair[idx]['end_frame']
obj_names = [motion_name.split('_')[1]]

root_path = f'/home/kimyw/github/sim/InterMimic/intermimic/data/assets/parahome/seq/s{scene_number}'
with open(f'{root_path}/bone_vectors.pkl', 'rb') as f:
    bone_vectors = pickle.load(f)
with open(f'{root_path}/body_global_transform.pkl', 'rb') as f:
    body_global_transform = pickle.load(f)
with open(f'{root_path}/body_joint_orientations.pkl', 'rb') as f:
    body_joint_orientations = pickle.load(f)
with open(f'{root_path}/hand_joint_orientations.pkl', 'rb') as f:
    hand_joint_orientations = pickle.load(f)
with open(f'{root_path}/joint_positions.pkl', 'rb') as f:
    joint_positions = pickle.load(f)
with open(f'{root_path}/object_transformations.pkl', 'rb') as f:
    object_transformations = pickle.load(f)
local_body_rot = np.load(f'{root_path}/local_body_rot.npy')[start_frame:end_frame]

# settings
# 0-22: body motion (23)
# 23-47: left hands (25)
# 48-72: right hands (25)
body_joint_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 'jLeftT4Shoulder': 11, 'jLeftShoulder': 12, 'jLeftElbow': 13, 'jLeftWrist': 14, 'jRightHip': 15, 'jRightKnee': 16, 'jRightAnkle': 17, 'jRightBallFoot': 18, 'jLeftHip': 19, 'jLeftKnee': 20, 'jLeftAnkle': 21, 'jLeftBallFoot': 22}
lhand_joint_order = {'jLeftWrist': 0, 'jLeftFirstCMC': 1, 'jLeftSecondCMC': 2, 'jLeftThirdCMC': 3, 'jLeftFourthCMC': 4, 'jLeftFifthCMC': 5, 'jLeftFifthMCP': 6, 'jLeftFifthPIP': 7, 'jLeftFifthDIP': 8, 'pLeftFifthTip': 9, 'jLeftFourthMCP': 10, 'jLeftFourthPIP': 11, 'jLeftFourthDIP': 12, 'pLeftFourthTip': 13, 'jLeftThirdMCP': 14, 'jLeftThirdPIP': 15, 'jLeftThirdDIP': 16, 'pLeftThirdTip': 17, 'jLeftSecondMCP': 18, 'jLeftSecondPIP': 19, 'jLeftSecondDIP': 20, 'pLeftSecondTip': 21, 'jLeftFirstMCP': 22, 'jLeftIP': 23, 'pLeftFirstTip': 24}
rhand_joint_order = {'jRightWrist': 0, 'jRightFirstCMC': 1, 'jRightSecondCMC': 2, 'jRightThirdCMC': 3, 'jRightFourthCMC': 4, 'jRightFifthCMC': 5, 'jRightFifthMCP': 6, 'jRightFifthPIP': 7, 'jRightFifthDIP': 8, 'pRightFifthTip': 9, 'jRightFourthMCP': 10, 'jRightFourthPIP': 11, 'jRightFourthDIP': 12, 'pRightFourthTip': 13, 'jRightThirdMCP': 14, 'jRightThirdPIP': 15, 'jRightThirdDIP': 16, 'pRightThirdTip': 17, 'jRightSecondMCP': 18, 'jRightSecondPIP': 19, 'jRightSecondDIP': 20, 'pRightSecondTip': 21, 'jRightFirstMCP': 22, 'jRightIP': 23, 'pRightFirstTip': 24}

motor_order = {'pHipOrigin': 0, 'jL5S1': 1, 'jL4L3': 2, 'jL1T12': 3, 'jT9T8': 4, 'jT1C7': 5, 'jC1Head': 6, 
            'jRightT4Shoulder': 7, 'jRightShoulder': 8, 'jRightElbow': 9, 'jRightWrist': 10, 
            'jRightFirstCMC': 11, 'jRightFirstMCP': 12, 'jRightIP': 13, 
            'jRightSecondCMC': 14, 'jRightSecondMCP': 15, 'jRightSecondPIP': 16, 'jRightSecondDIP': 17, 
            'jRightThirdCMC': 18, 'jRightThirdMCP': 19, 'jRightThirdPIP': 20, 'jRightThirdDIP': 21, 
            'jRightFourthCMC': 22, 'jRightFourthMCP': 23, 'jRightFourthPIP': 24, 'jRightFourthDIP': 25, 
            'jRightFifthCMC': 26, 'jRightFifthMCP': 27, 'jRightFifthPIP': 28, 'jRightFifthDIP': 29, 
            'jLeftT4Shoulder': 30, 'jLeftShoulder': 31, 'jLeftElbow': 32, 'jLeftWrist': 33, 
            'jLeftFirstCMC': 34, 'jLeftFirstMCP': 35, 'jLeftIP': 36, 
            'jLeftSecondCMC': 37, 'jLeftSecondMCP': 38, 'jLeftSecondPIP': 39, 'jLeftSecondDIP': 40, 
            'jLeftThirdCMC': 41, 'jLeftThirdMCP': 42, 'jLeftThirdPIP': 43, 'jLeftThirdDIP': 44, 
            'jLeftFourthCMC': 45, 'jLeftFourthMCP': 46, 'jLeftFourthPIP': 47, 'jLeftFourthDIP': 48, 
            'jLeftFifthCMC': 49, 'jLeftFifthMCP': 50, 'jLeftFifthPIP': 51, 'jLeftFifthDIP': 52, 
            'jRightHip': 53, 'jRightKnee': 54, 'jRightAnkle': 55, 'jRightBallFoot': 56, 
            'jLeftHip': 57, 'jLeftKnee': 58, 'jLeftAnkle': 59, 'jLeftBallFoot': 60}
num_motors = len(motor_order) - 1
num_joints = 71

# human
# root_pos(3) + root_rot(3) + root_pos_vel(3) +dof_pos(60*3) + joint_pos(71*3) + obj_pos(3) + obj_rot(3) + contact_graph(1)
motion_dim = 6 + 3 + num_motors*3 + num_joints*3 + 6*len(obj_names) + 1 
motion = torch.zeros(end_frame-start_frame, motion_dim)
root_pos = np.stack([body_global_transform[i][:3, 3] for i in range(start_frame, end_frame)])
motion[:, 0:3] = torch.tensor(root_pos) # root_pos
root_rot = R.from_matrix([body_global_transform[i][:3, :3] for i in range(start_frame, end_frame)])
motion[:, 3:6] = torch.tensor(root_rot.as_rotvec()) # root_rot_3d
start_ind = 9
motion[:, start_ind:start_ind+num_motors*3] = torch.tensor(local_body_rot).view(-1, num_motors*3) # dof_pos

joint_pos = torch.zeros(end_frame-start_frame, num_joints*3)
joint_pos[:,0:3] = torch.tensor(joint_positions[start_frame:end_frame, 0]) # pHipOrigin
for key in motor_order:
    ind = motor_order[key]
    if key in body_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, body_joint_order[key]])
    elif key in lhand_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, 23+lhand_joint_order[key]])
    elif key in rhand_joint_order:
        joint_pos[:, 3*ind:3*ind+3] = torch.tensor(joint_positions[start_frame:end_frame, 48+rhand_joint_order[key]])
start_ind += num_motors * 3
motion[:, start_ind:start_ind+num_joints*3] = joint_pos

# obejct
start_ind += num_joints*3
for obj_name in obj_names:
    obj_pos = np.stack([object_transformations[i][f'{obj_name}_base'][:3, 3] for i in range(start_frame, end_frame)])
    obj_rot = R.from_matrix([object_transformations[i][f'{obj_name}_base'][:3, :3] for i in range(start_frame, end_frame)])
    motion[:, start_ind:start_ind+3] = torch.tensor(obj_pos) # obj_pos
    start_ind += 3
    motion[:, start_ind:start_ind+3] = torch.tensor(obj_rot.as_rotvec()) # obj_rot
    start_ind += 3


save_path = f'skillmimic/data/motions/ParaHome/wristmimic/s{scene_number}_{motion_name}/s{scene_number}_{motion_name}.pt'

os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(motion.to('cuda'), save_path)
print(f'Save motion into {save_path}')
print(f'Motion shape: {motion.shape}')