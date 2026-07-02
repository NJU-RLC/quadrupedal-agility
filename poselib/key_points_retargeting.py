import copy
import math
import json
import os.path

import numpy as np
from isaacgym.torch_utils import *
from kinematics import Kinematics
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.skeleton.backend.fbx.fbx_read_wrapper import fbx_to_array
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# get key points
# transforms:(len, 27, 4, 4)
# ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_End', 'LeftShoulder', 'LeftArm',
#  'LeftForeArm', 'LeftHand', 'LeftHand_End', 'RightShoulder', 'RightArm', 'RightForeArm',
#  'RightHand', 'RightHand_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End',
#  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 'Tail', 'Tail1', 'Tail1_End']

# ['trunk', 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot', 'RR_hip',
# 'RR_thigh', 'RR_calf', 'RR_foot', 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']

# source file path
json_dir = "clip_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

root_dir = 'data/npy_data'
output_dir = 'data/clipped_data'
output_file = '{}_{}_{}_{}'.format(file_name, clip[0], clip[1], remarks)

body_length = 0.3868
body_wide = 0.093
hip_length = 0.0955
thigh_length = 0.213
calf_length = 0.213

# target robot parameters (meter)
kinematic = Kinematics(body_length=body_length, body_wide=body_wide, hip_length=hip_length, thigh_length=thigh_length,
                       calf_length=calf_length)

# load and visualize source motion sequence
source_motion = SkeletonMotion.from_file(
    os.path.join(root_dir, '{}.npy'.format(file_name)))  # generate from fbx_importer

# plot_skeleton_motion_interactive(source_motion)

# zoom according to source t-pose and target t-pose
source_tpose = SkeletonState.from_file('data/dog_tpose.npy')
# plot_skeleton_state(source_tpose)
target_tpose = SkeletonState.from_file('data/go2_tpose.npy')
# plot_skeleton_state(target_tpose)
skeleton_s = source_tpose.skeleton_tree
skeleton_t = target_tpose.skeleton_tree
source_length = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 0] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 0])
source_wide = torch.abs(source_tpose.global_translation[skeleton_s.index('LeftHand_End'), 1] -
                        source_tpose.global_translation[skeleton_s.index('RightHand_End'), 1])
source_height = torch.abs(source_tpose.global_translation[skeleton_s.index('Hips'), 2] -
                          source_tpose.global_translation[skeleton_s.index('LeftFoot_End'), 2])

target_length = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 0] -
                          target_tpose.global_translation[skeleton_t.index('RL_foot'), 0])
target_wide = torch.abs(target_tpose.global_translation[skeleton_t.index('FL_foot'), 1] -
                        target_tpose.global_translation[skeleton_t.index('FR_foot'), 1])
target_height = torch.abs(target_tpose.global_translation[skeleton_t.index('trunk'), 2] -
                          target_tpose.global_translation[skeleton_t.index('FL_foot'), 2])

zoom_x = (target_length / source_length) * 1.  # height
zoom_y = target_wide / source_wide
zoom_z = target_height / source_height
zoom_scale = min([zoom_x, zoom_y, zoom_z])

skeleton = source_motion.skeleton_tree
local_translation = skeleton.local_translation  # (27, 3)
root_translation = source_motion.root_translation  # (len, 3)

print('zoom_x: {}, zoom_y: {}, zoom_z: {}'.format(zoom_x, zoom_y, zoom_z))
for i in range(local_translation.shape[0]):
    local_translation[i, 0] *= zoom_scale
    local_translation[i, 1] *= zoom_scale
    local_translation[i, 2] *= zoom_scale

root_translation[:, 0] *= zoom_scale
root_translation[:, 1] *= zoom_scale
root_translation[:, 2] *= zoom_scale

# plot_skeleton_motion_interactive(source_motion)

# global_root_translation 在source_motion.root_translation改变后自动更新
# global_root_translation = root_translation
global_trans = source_motion.global_transformation  # (len， 27, 7)
global_root_translation = global_trans[:, skeleton.node_names.index("Hips"), -3:]  # (len, 7)

# global_left_shoulder_trans = global_trans[:, skeleton.node_names.index("LeftShoulder"), -3:]
# global_right_shoulder_trans = global_trans[:, skeleton.node_names.index("RightShoulder"), -3:]  # (len, 3)
# global_left_up_leg_trans = global_trans[:, skeleton.node_names.index("LeftUpLeg"), -3:]  # (len, 3)
# global_right_up_leg_trans = global_trans[:, skeleton.node_names.index("RightUpLeg"), -3:]  # (len, 3)

global_left_hand_end_trans = global_trans[:, skeleton.node_names.index("LeftHand_End"), -3:]  # (len, 3)
global_right_hand_end_trans = global_trans[:, skeleton.node_names.index("RightHand_End"), -3:]  # (len, 3)
global_left_foot_end_trans = global_trans[:, skeleton.node_names.index("LeftFoot_End"), -3:]  # (len, 3)
global_right_foot_end_trans = global_trans[:, skeleton.node_names.index("RightFoot_End"), -3:]  # (len, 3)

global_root_rotation_quat = global_trans[:, skeleton.node_names.index("Hips"), :4]

global_root_rotation_quat = global_root_rotation_quat.numpy()
global_root_rotation_euler = [kinematic.quaternion2rpy(q) for q in
                              global_root_rotation_quat]
global_root_rotation_euler = np.array(global_root_rotation_euler)

# inverse kinematic
g_root_trans = global_root_translation.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_root_rot_euler = global_root_rotation_euler[clip[0]:clip[1], :]  # (len, 3)
g_lh_trans = global_left_hand_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_rh_trans = global_right_hand_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_lf_trans = global_left_foot_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)
g_rf_trans = global_right_foot_end_trans.numpy()[clip[0]:clip[1], :]  # (len, 3)

for i in range(g_root_rot_euler.shape[0]):
    if g_root_rot_euler[i, 2] <= np.pi / 2:
        g_root_rot_euler[i, 2] = 2 * np.pi + g_root_rot_euler[i, 2]
g_root_rot_euler[:, 2] = g_root_rot_euler[:, 2] - np.pi

g_root_rot_euler[:, 0] = np.pi / 2 - g_root_rot_euler[:, 0]
for i in range(g_root_rot_euler.shape[0]):
    for j in range(3):
        if g_root_rot_euler[i, j] >= np.pi:
            g_root_rot_euler[i, j] = g_root_rot_euler[i, j] - 2 * np.pi
        elif g_root_rot_euler[i, j] <= -np.pi:
            g_root_rot_euler[i, j] = 2 * np.pi + g_root_rot_euler[i, j]
g_root_rot_euler[:, 0] = -g_root_rot_euler[:, 0]
g_root_rot_euler[:, 1] = -g_root_rot_euler[:, 1]

rot_matrix = kinematic.rot_matrix_ba([0, 0, -g_root_rot_euler[0, -1]])
g_root_trans = np.array([rot_matrix @ t for t in g_root_trans])
g_root_rot_euler[:, -1] -= g_root_rot_euler[0, -1]
g_lh_trans = np.array([rot_matrix @ t for t in g_lh_trans])
g_rh_trans = np.array([rot_matrix @ t for t in g_rh_trans])
g_lf_trans = np.array([rot_matrix @ t for t in g_lf_trans])
g_rf_trans = np.array([rot_matrix @ t for t in g_rf_trans])

g_foot_trans = np.hstack([g_rh_trans, g_lh_trans, g_rf_trans, g_lf_trans])

root_new_local = np.array([-body_length / 3.0, 0, 0, 1])
g_root_trans = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])

joint_rot_euler = np.zeros((g_root_trans.shape[0], 12))
for i in range(g_root_trans.shape[0]):
    ang = kinematic.inverse_kinematics(g_root_trans[i, :], g_root_rot_euler[i, :], g_foot_trans[i, :])
    joint_rot_euler[i, :] = ang
if np.isnan(joint_rot_euler).any():
    raise ValueError('Angle can not be nan!')

hip_limit = [-0.2, 0.2]
thigh_limit_h = [-1.5708, 3.4907]
thigh_limit_f = [-0.5236, 3.4907]
calf_limit = [-2.7227, -0.83776]
for i in range(joint_rot_euler.shape[0]):
    for j in range(joint_rot_euler.shape[1]):
        if j == 0 or j == 3 or j == 6 or j == 9:
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], hip_limit[0], hip_limit[1])
        elif j == 1 or j == 4:  # hand
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], thigh_limit_h[0], thigh_limit_h[1])
        elif j == 7 or j == 10:  # foot
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], thigh_limit_f[0], thigh_limit_f[1])
        elif j == 2 or j == 5 or j == 8 or j == 11:
            joint_rot_euler[i, j] = np.clip(joint_rot_euler[i, j], calf_limit[0], calf_limit[1])

g_root_rot_euler = torch.from_numpy(g_root_rot_euler)
g_root_rot_quat = quat_from_euler_xyz(g_root_rot_euler[:, 0], g_root_rot_euler[:, 1],
                                      g_root_rot_euler[:, 2])  # (len. 4)

joint_rot_euler = torch.from_numpy(joint_rot_euler)  # (190, 12)
skeleton_target = target_tpose.skeleton_tree
local_rotation = torch.zeros((joint_rot_euler.shape[0], 17, 4))  # (len, 17, 4)

for i in range(joint_rot_euler.shape[0]):
    joint_rot_quat = []
    for j in range(joint_rot_euler.shape[1]):
        if j == 0 or j == 3 or j == 6 or j == 9:
            # joint_quat = quat_from_euler_xyz(joint_rot_euler[i, j], torch.tensor(0), torch.tensor(0))
            joint_quat = kinematic.rpy2quaternion([joint_rot_euler[i, j], 0, 0])
        else:
            # joint_quat = quat_from_euler_xyz(torch.tensor(0), joint_rot_euler[i, j], torch.tensor(0))
            joint_quat = kinematic.rpy2quaternion([0, joint_rot_euler[i, j], 0])
        joint_quat = torch.from_numpy(np.array(joint_quat))
        joint_rot_quat.append(joint_quat)
    local_rotation[i, 0, :] = g_root_rot_quat[i, :]
    local_rotation[i, 1, :] = joint_rot_quat[0]
    local_rotation[i, 2, :] = joint_rot_quat[1]
    local_rotation[i, 3, :] = joint_rot_quat[2]
    local_rotation[i, 5, :] = joint_rot_quat[3]
    local_rotation[i, 6, :] = joint_rot_quat[4]
    local_rotation[i, 7, :] = joint_rot_quat[5]
    local_rotation[i, 9, :] = joint_rot_quat[6]
    local_rotation[i, 10, :] = joint_rot_quat[7]
    local_rotation[i, 11, :] = joint_rot_quat[8]
    local_rotation[i, 13, :] = joint_rot_quat[9]
    local_rotation[i, 14, :] = joint_rot_quat[10]
    local_rotation[i, 15, :] = joint_rot_quat[11]

root_new_local = np.array([body_length / 2, 0, 0, 1])
root_trans_w = np.array([list((kinematic.trans_matrix_ba(m, t) @ root_new_local)[:3])
                         for m, t in zip(g_root_trans, g_root_rot_euler)])

root_translation = torch.from_numpy(root_trans_w)
root_translation[:, 0] = root_translation[:, 0] - root_translation[0, 0]
root_translation[:, 1] = root_translation[:, 1] - root_translation[0, 1]
# root_translation[:, 2] = root_translation[:, 2] - torch.min(global_trans[0, :, 2])

# foot above the ground
root_height_offset = 0.02
root_translation += root_height_offset
skeleton_state = SkeletonState.from_rotation_and_root_translation(target_tpose.skeleton_tree, local_rotation,
                                                                  root_translation, is_local=True)
target_motion = SkeletonMotion.from_skeleton_state(skeleton_state, fps=source_motion.fps)
target_motion.to_file(os.path.join(output_dir, '{}.npy'.format(output_file)))

# target_motion = SkeletonMotion.from_file(os.path.join(output_dir, '{}.npy'.format(output_file)))
# plot_skeleton_motion_interactive(target_motion)
