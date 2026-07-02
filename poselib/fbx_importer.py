import re
import os
import json
import copy
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# ['Hips', 'Spine', 'Spine1', 'Neck', 'Head', 'Head_End', 'LeftShoulder', 'LeftArm',
#  'LeftForeArm', 'LeftHand', 'LeftHand_End', 'RightShoulder', 'RightArm', 'RightForeArm',
#  'RightHand', 'RightHand_End', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End',
#  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 'Tail', 'Tail1', 'Tail1_End']

# source file path
json_dir = "clip_config.json"
with open(json_dir, "r") as f:
    motion_json = json.load(f)
    file_name = motion_json["file_name"]
    clip = motion_json["clip"]
    remarks = motion_json["remarks"]

# source fbx file path
root_dir = "data/fbx_data"
npy_dir = "data/npy_data"

file_name = "{}.fbx".format(file_name)

print('File: ', file_name)
fbx_file = os.path.join(root_dir, file_name)
base_name, extension = os.path.splitext(file_name)
npy_file = os.path.join(npy_dir, '{}.npy'.format(base_name))

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    fps=60
)

skeleton = motion.skeleton_tree

for i in range(len(skeleton.node_names)):
    rgx = r'^.+:(.+)$'
    skeleton.node_names[i] = re.search(rgx, skeleton.node_names[i]).group(1)

local_rotation = motion.local_rotation
for i in range(local_rotation.shape[0]):
    local_rotation[i, skeleton.node_names.index("Hips"), :] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
        local_rotation[i, skeleton.node_names.index("Hips"), :]
    )

local_rotation = motion.local_rotation
for i in range(local_rotation.shape[0]):
    local_rotation[i, skeleton.node_names.index("Hips"), :] = quat_mul(
        quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
        local_rotation[i, skeleton.node_names.index("Hips"), :]
    )

root_translation = motion.root_translation
root_translation_copy = copy.deepcopy(root_translation[:, 2])
root_translation[:, 2] = root_translation[:, 0]
root_translation[:, 0] = root_translation_copy

root_translation = motion.root_translation
root_translation_copy = copy.deepcopy(root_translation[:, 2])
root_translation[:, 2] = root_translation[:, 1]
root_translation[:, 1] = root_translation_copy

root_translation[:root_translation.shape[0], 0] -= copy.deepcopy(root_translation[0, 0])
root_translation[:root_translation.shape[0], 1] -= copy.deepcopy(root_translation[0, 1])
global_trans = motion.global_translation
root_translation[:, 2] -= global_trans[0, skeleton.node_names.index("RightFoot_End"), 2]

repeat_num = 2
local_rotation_out = []  # (len, 27, 4)
root_translation_out = []  # (len, 3)
for i in range(local_rotation.shape[0]):
    if i != 0 and i % repeat_num == 0:
        continue
    local_rotation_out.append(local_rotation[i, ...])
    root_translation_out.append(root_translation[i, ...])
local_rotation_out = torch.stack(local_rotation_out)
root_translation_out = torch.stack(root_translation_out)

new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, local_rotation_out,
                                                                root_translation_out, is_local=True)
target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps / repeat_num)

# visualize motion
plot_skeleton_motion_interactive(target_motion)
