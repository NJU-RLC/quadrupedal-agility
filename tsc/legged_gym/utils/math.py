import math
import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


# @ torch.jit.script
def torch_rand_int(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return ((upper - lower) * torch.rand(*shape, device=device).squeeze(1) + lower).long().float()


def sample_unit_vector(n, dim, device):
    tensor = torch.randn(n, dim, device=device)
    unit_vector = tensor / torch.norm(tensor, dim=-1, keepdim=True)
    return unit_vector


def rot_matrix(t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    return r


def rot_matrix_inv(t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    return r.T


def trans_matrix(m, t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    trans = np.hstack([r, np.array(m)[:, np.newaxis]])
    trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
    return trans


def trans_matrix_inv(m, t):
    r = np.array([[np.cos(t[2]) * np.cos(t[1]),
                   np.cos(t[2]) * np.sin(t[1]) * np.sin(t[0]) - np.sin(t[2]) * np.cos(t[0]),
                   np.cos(t[2]) * np.sin(t[1]) * np.cos(t[0]) + np.sin(t[2]) * np.sin(t[0])],
                  [np.sin(t[2]) * np.cos(t[1]),
                   np.sin(t[2]) * np.sin(t[1]) * np.sin(t[0]) + np.cos(t[2]) * np.cos(t[0]),
                   np.sin(t[2]) * np.sin(t[1]) * np.cos(t[0]) - np.cos(t[2]) * np.sin(t[0])],
                  [-np.sin(t[1]), np.cos(t[1]) * np.sin(t[0]), np.cos(t[1]) * np.cos(t[0])]])
    trans = np.hstack([r.T, -np.dot(r.T, np.array(m))[:, np.newaxis]])
    trans = np.vstack([trans, np.array([[0, 0, 0, 1]])])
    return trans


def quaternion2rpy(q):
    x, y, z, w = q[0], q[1], q[2], q[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [roll, pitch, yaw]


def rpy2quaternion(r):
    roll, pitch, yaw = r[0], r[1], r[2]
    x = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    y = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
        yaw / 2)
    z = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
        yaw / 2)
    w = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
        yaw / 2)
    return [x, y, z, w]
