import copy
import random
import numpy as np
import torch
from numpy.random import choice
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym import terrain_utils
from legged_gym.utils.math import *
from skimage.draw import polygon


class Obstacle:
    def __init__(self, cfg: LeggedRobotCfg.obstacle, num_envs) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_robots = num_envs
        self.num_obstacles = num_envs
        self.num_cols = int(np.floor(np.sqrt(num_envs)))
        self.num_rows = int(np.ceil(num_envs / self.num_cols))
        self.env_length = cfg.env_length
        self.env_width = cfg.env_width
        self.num_links_per_obst = cfg.num_obstacle_links
        self.num_joints_per_obst = cfg.num_obstacle_joints
        self.proportions = [np.sum(cfg.obstacle_proportions[:i + 1]) for i in range(len(cfg.obstacle_proportions))]
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        self.num_goals = cfg.num_goals
        self.obst_types = list(cfg.obstacle_dict.keys())
        self.num_obst_per_env = cfg.num_obst_per_env
        self.frame_pos = np.array(cfg.frame_pos)
        self.frame_ang = np.radians(np.array(cfg.frame_ang))
        self.random_yaw = np.radians(cfg.random_yaw)
        self.seesaw_dof_pos = -np.arcsin(0.25 / 1.5)
        self.curriculum = cfg.curriculum
        self.curr_step = cfg.curr_step
        self.curr_threshold = cfg.curr_threshold
        self.bar_jump_joint_bias = -1
        self.tire_jump_joint_bias = -10

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(cfg.border_size / self.horizontal_scale)
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.x_edge_mask = np.zeros((self.tot_rows, self.tot_cols), dtype=bool)
        self.bar_jump_mask = None
        self.tire_jump_mask = None

        self.env_origins = np.zeros((num_envs, 3))
        # create a grid of obstacles
        xx, yy = np.meshgrid(np.arange(self.num_rows), np.arange(self.num_cols))
        self.spacing_x = cfg.env_length
        self.spacing_y = cfg.env_width
        self.env_boarder = cfg.env_boarder
        self.env_center = np.array([self.spacing_x, self.spacing_y]) / 2
        self.env_origins[:, 0] = self.spacing_x * xx.flatten()[:num_envs]
        self.env_origins[:, 1] = self.spacing_y * yy.flatten()[:num_envs]
        self.env_origins[:, 2] = 0.

        self.obstacle_origins = np.zeros((num_envs, self.num_obst_per_env, 3))
        self.obstacle_yaws = np.zeros((num_envs, self.num_obst_per_env))
        self.obstacle_types = np.zeros((num_envs, self.num_obst_per_env), dtype=int)
        self.obstacle_joint_pos = np.zeros((num_envs, self.num_obst_per_env)) - 1  # -1: no joint

        self.last_goal_repeat = cfg.last_goal_repeat
        self.env_goals = np.zeros((num_envs, self.num_obst_per_env, cfg.num_goals, 3))
        self.bar_jump_goal_mask = None
        self.tire_jump_goal_mask = None
        env_ids = list(range(self.num_envs))
        self._create_obstacle(env_ids)

    def _create_obstacle(self, env_ids):
        for i in env_ids:
            obst_types = copy.deepcopy(self.obst_types)
            random.shuffle(obst_types)
            terrain_set = terrain_utils.SubTerrain("terrain",
                                                   width=self.length_per_env_pixels,
                                                   length=self.width_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)
            terrain_set.x_edge_mask = np.zeros_like(terrain_set.height_field_raw)
            terrain_set.goals = np.zeros((self.num_obst_per_env, self.num_goals, 3))
            for j, obst_type in enumerate(obst_types):
                obst_idx = self.obst_types.index(obst_type)
                terrain = terrain_utils.SubTerrain("terrain",
                                                   width=self.length_per_env_pixels,
                                                   length=self.width_per_env_pixels,
                                                   vertical_scale=self.cfg.vertical_scale,
                                                   horizontal_scale=self.cfg.horizontal_scale)
                random_x = self.cfg.random_x[obst_type]
                noise_x = np.random.uniform(random_x[0], random_x[1])
                noise_y = np.random.uniform(self.cfg.random_y[0], self.cfg.random_y[1])
                noise_pos = np.array([noise_x, noise_y])
                noise_yaw = np.random.uniform(self.random_yaw[0], self.random_yaw[1])
                frame_pos = (self.frame_pos[j][1] - self.frame_pos[j][0]) / 2 + self.frame_pos[j][0]
                frame_yaw = self.frame_ang[j]
                obst_pos = frame_pos + noise_pos
                obst_pos_z = 0
                obst_pos_bias = [0, 0]
                obst_yaw = frame_yaw + noise_yaw
                joint_pos = -1
                env_center = copy.deepcopy(self.env_center)
                if obst_idx == 0:  # bar_jump
                    if self.curriculum:
                        joint_pos = np.random.uniform(self.cfg.bar_jump_init_range[0], self.cfg.bar_jump_init_range[1])
                    else:
                        joint_pos = np.random.uniform(self.cfg.bar_jump_range[0], self.cfg.bar_jump_range[1])
                    joint_pos += self.bar_jump_joint_bias
                    self.bar_jump_obstacle(terrain, env_center, joint_pos)
                elif obst_idx == 1:  # frame
                    self.frame_obstacle(terrain, env_center)
                elif obst_idx == 2:  # poles
                    obst_pos_bias = [-1.5, 0]
                    obst_pos += obst_pos_bias
                    env_center += obst_pos_bias
                    self.poles_obstacle(terrain, env_center)
                elif obst_idx == 3:  # seesaw
                    obst_pos_z = 0.26
                    joint_pos = self.seesaw_dof_pos
                    self.seesaw_obstacle(terrain, env_center, joint_pos)
                elif obst_idx == 4:  # tire_jump
                    if self.curriculum:
                        joint_pos = np.random.uniform(self.cfg.tire_jump_init_range[0], self.cfg.tire_jump_init_range[1])
                    else:
                        joint_pos = np.random.uniform(self.cfg.tire_jump_range[0], self.cfg.tire_jump_range[1])
                    joint_pos += self.tire_jump_joint_bias
                    self.tire_jump_obstacle(terrain, env_center, joint_pos)
                elif obst_idx == 5:  # tunnel
                    obst_pos_bias = [-1.0, 0]
                    obst_pos += obst_pos_bias
                    env_center += obst_pos_bias
                    self.tunnel_obstacle(terrain, env_center)

                # ####### Frame transformation ########
                pos_org = (env_center - obst_pos_bias) / self.horizontal_scale
                obst_org = (obst_pos - obst_pos_bias) / self.horizontal_scale
                rot_mat = np.array([[np.cos(obst_yaw), -np.sin(obst_yaw)], [np.sin(obst_yaw), np.cos(obst_yaw)]])
                new_mat = np.zeros_like(terrain.height_field_raw)
                new_mat_edge = np.zeros_like(terrain.x_edge_mask)
                mat_size = new_mat.shape
                # Rect transformation
                rect_points = terrain.rect_points
                for m in range(rect_points.shape[0]):
                    for n in range(rect_points.shape[1]):
                        terrain.rect_points[m, n] = rot_mat @ (rect_points[m, n] - pos_org) + obst_org
                    # Fill the transformed rectangular area with polygon
                    transformed_points = terrain.rect_points[m]
                    rr, cc = polygon(transformed_points[:, 0], transformed_points[:, 1], mat_size)
                    rr_original = np.clip(np.round(
                        (rr - obst_org[0]) * np.cos(obst_yaw) + (cc - obst_org[1]) * np.sin(obst_yaw) + pos_org[0]).
                                          astype(int), 0, mat_size[0] - 1)
                    cc_original = np.clip(np.round(
                        (cc - obst_org[1]) * np.cos(obst_yaw) - (rr - obst_org[0]) * np.sin(obst_yaw) + pos_org[1]).
                                          astype(int), 0, mat_size[1] - 1)
                    new_mat[rr, cc] = terrain.height_field_raw[rr_original, cc_original]
                # Rect_edge transformation
                rect_edge_points = terrain.rect_edge_points
                for m in range(rect_edge_points.shape[0]):
                    for n in range(rect_edge_points.shape[1]):
                        terrain.rect_edge_points[m, n] = rot_mat @ (rect_edge_points[m, n] - pos_org) + obst_org
                    # Fill the transformed rectangular area with polygon
                    transformed_points = terrain.rect_edge_points[m]
                    rr, cc = polygon(transformed_points[:, 0], transformed_points[:, 1], mat_size)
                    rr_original = np.clip(np.round(
                        (rr - obst_org[0]) * np.cos(obst_yaw) + (cc - obst_org[1]) * np.sin(obst_yaw) + pos_org[0]).
                                          astype(int), 0, mat_size[0] - 1)
                    cc_original = np.clip(np.round(
                        (cc - obst_org[1]) * np.cos(obst_yaw) - (rr - obst_org[0]) * np.sin(obst_yaw) + pos_org[1]).
                                          astype(int), 0, mat_size[1] - 1)
                    new_mat_edge[rr, cc] = terrain.x_edge_mask[rr_original, cc_original]
                # Goals transformation
                goals = terrain.goals[:, :2]
                pos_org *= self.horizontal_scale
                obst_org *= self.horizontal_scale
                for m in range(goals.shape[0]):
                    terrain.goals[m, :2] = rot_mat @ (goals[m] - pos_org) + obst_org

                self.obstacle_types[i, j] = obst_idx
                self.obstacle_origins[i, j] = self.env_origins[i] + np.append(obst_pos, obst_pos_z)
                self.obstacle_yaws[i, j] = obst_yaw
                if obst_idx == 0:  # bar_jump
                    joint_pos -= self.bar_jump_joint_bias
                elif obst_idx == 4:  # tire_jump
                    joint_pos -= self.tire_jump_joint_bias
                self.obstacle_joint_pos[i, j] = joint_pos
                self.add_border(new_mat)
                terrain_set.height_field_raw = terrain_set.height_field_raw | new_mat
                terrain_set.x_edge_mask = terrain_set.x_edge_mask | new_mat_edge
                terrain_set.goals[j] = terrain.goals

            self.add_terrain_to_map(terrain_set, i)

        self.bar_jump_mask = (self.height_field_raw > int(-5 / self.vertical_scale)) & (self.height_field_raw < 0)
        self.tire_jump_mask = (self.height_field_raw < int(-5 / self.vertical_scale))
        self.bar_jump_goal_mask = (self.env_goals > int(-5 / self.vertical_scale)) & (self.env_goals < 0)
        self.tire_jump_goal_mask = (self.env_goals < int(-5 / self.vertical_scale))
        self.height_field_raw[self.bar_jump_mask] -= int(self.bar_jump_joint_bias / self.vertical_scale)
        self.height_field_raw[self.tire_jump_mask] -= int(self.tire_jump_joint_bias / self.vertical_scale)
        self.env_goals[self.bar_jump_goal_mask] -= self.bar_jump_joint_bias
        self.env_goals[self.tire_jump_goal_mask] -= self.tire_jump_joint_bias

    def refresh_seesaw(self):
        seesaw_ids = np.where(self.obstacle_types == 3)[0]
        self.refresh_index(seesaw_ids)


    def add_terrain_to_map(self, terrain, i):
        x = (self.env_origins[i, 0]) / self.horizontal_scale
        y = (self.env_origins[i, 1]) / self.horizontal_scale
        # map coordinate system
        start_x = self.border + x
        end_x = self.border + x + self.length_per_env_pixels
        start_y = self.border + y
        end_y = self.border + y + self.width_per_env_pixels
        self.height_field_raw[int(start_x):int(end_x), int(start_y):int(end_y)] = terrain.height_field_raw
        self.x_edge_mask[int(start_x):int(end_x), int(start_y):int(end_y)] = terrain.x_edge_mask
        self.env_goals[i] = terrain.goals + np.array([self.env_origins[i, 0], self.env_origins[i, 1], 0])

    def add_border(self, mat):
        width = int(10.0 / self.horizontal_scale)
        length = int(7.0 / self.horizontal_scale)
        height = int(2.0 / self.vertical_scale)
        thickness = int(0.1 / self.horizontal_scale)

        rects = [[0, 0, length, thickness],
                 [0, 0, thickness, width],
                 [length - thickness, 0, thickness, width],
                 [0, width - thickness, length, thickness]]
        for rect in rects:
            mat[rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]] = height

    def bar_jump_obstacle(self, terrain, obst_pos, joint_pos):
        width_1 = int(1.2 / self.horizontal_scale)
        length_1 = int(0.2 / self.horizontal_scale)
        height_1 = int(joint_pos / self.vertical_scale)
        width_2 = int(2.04 / self.horizontal_scale)
        length_2 = int(0.5 / self.horizontal_scale)
        height_2 = int(0.42 / self.vertical_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)

        # (x_start, y_start, width, height)
        rects = [[int(pos_x - length_1 / 2), int(pos_y - width_1 / 2), length_1, width_1],
                 [int(pos_x - length_2 / 2), int(pos_y - width_2 / 2), length_2, int((width_2 - width_1) / 2)],
                 [int(pos_x - length_2 / 2), int(pos_y + width_1 / 2), length_2, int((width_2 - width_1) / 2)]]
        rect_values = [height_1, height_2, height_2]
        rect_points = []
        rect_edge = []
        rect_edge_points = []

        for rect, v in zip(rects, rect_values):
            terrain.height_field_raw[rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]] = v
            # Record in counterclockwise order
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        x_edge_mask = copy.deepcopy(terrain.height_field_raw)
        x_edge_mask = x_edge_mask.astype(bool)

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)
        terrain.x_edge_mask = np.array(x_edge_mask)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 1.8
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        goals[0] = np.array([pos_x - goal_x_step, pos_y, goal_z_bias])
        goals[1] = np.array([pos_x - goal_x_step / 2, pos_y, goal_z_bias])
        goals[2] = np.array([pos_x, pos_y, joint_pos + goal_z_bias])
        goals[3] = np.array([pos_x + goal_x_step / 2, pos_y, goal_z_bias])
        terrain.goals = goals

    def frame_obstacle(self, terrain, obst_pos):
        width = int(0.6 / self.horizontal_scale)
        length = int(1.4625 / self.horizontal_scale)
        height = int(0.333 / self.vertical_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)
        slope = (height * self.vertical_scale) / (length * self.horizontal_scale)
        ratio = self.horizontal_scale / self.vertical_scale

        rects = []
        x_range = np.array(range(int(pos_x - length), int(pos_x) + 1))
        y_range = np.array(range(int(pos_y - width / 2), int(pos_y + width / 2) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile((x_range[:] - x_range[0]) * slope * ratio, (len(y_range), 1)).T)
        rects.append([x_range[0], y_range[0], length, width])

        x_range = np.array(range(int(pos_x), int(pos_x + length) + 1))
        y_range = np.array(range(int(pos_y - width / 2), int(pos_y + width / 2) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile(np.flip(x_range[:] - x_range[0]) * slope * ratio, (len(y_range), 1)).T)
        rects.append([x_range[0], y_range[0], length, width])

        rect_edge = []
        edge_size = 1
        x_edge_mask = np.zeros_like(terrain.height_field_raw)
        x_edge_mask[int(pos_x - length):int(pos_x + length) + 1,
        int(pos_y - width / 2):int(pos_y - width / 2) + edge_size + 1] = 1
        rect_edge.append([int(pos_x - length), int(pos_y - width / 2), 2 * length, edge_size])
        x_edge_mask[int(pos_x - length):int(pos_x + length) + 1,
        int(pos_y + width / 2) - edge_size:int(pos_y + width / 2) + 1] = 1
        rect_edge.append([int(pos_x - length), int(pos_y + width / 2) - edge_size, 2 * length, edge_size])
        terrain.x_edge_mask = x_edge_mask.astype(bool)

        rect_points = []
        rect_edge_points = []
        for rect in rects:
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])
        for rect in rect_edge:
            rect_edge_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                     [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 0.7
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        length *= self.horizontal_scale
        height *= self.vertical_scale
        goals[0] = np.array([pos_x - length - goal_x_step, pos_y, goal_z_bias])
        goals[1] = np.array([pos_x - length, pos_y, goal_z_bias])
        goals[2] = np.array([pos_x, pos_y, height + goal_z_bias])
        goals[3] = np.array([pos_x + length, pos_y, goal_z_bias])
        terrain.goals = goals

    def poles_obstacle(self, terrain, obst_pos):
        radius = int((0.2 / 2) / self.horizontal_scale)
        height = int(1.0 / self.vertical_scale)
        x_spacing = int(1.0 / self.horizontal_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)
        num_poles = 4
        rects = []
        rect_points = []
        for i in range(num_poles):
            rect = [int(pos_x - radius) + i * x_spacing, int(pos_y - radius), 2 * radius, 2 * radius]
            rects.append(rect)
        rects = np.array(rects)
        for rect in rects:
            terrain.height_field_raw[rect[0]:rect[0] + rect[2], rect[1]:rect[1] + rect[3]] = height
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        rect_edge_points = []
        x_edge_mask = np.zeros_like(terrain.height_field_raw)
        x_edge_mask = x_edge_mask.astype(bool)

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)
        terrain.x_edge_mask = np.array(x_edge_mask)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 1.0
        goal_y_bias = 0.5
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        x_spacing *= self.horizontal_scale
        for i in range(num_poles):
            if i % 2 == 0:
                goals[i] = np.array([pos_x + i * x_spacing, pos_y - goal_y_bias, goal_z_bias])
            else:
                goals[i] = np.array([pos_x + i * x_spacing, pos_y + goal_y_bias, goal_z_bias])
        terrain.goals = goals

    def seesaw_obstacle(self, terrain, obst_pos, joint_pos):
        width = int(0.6 / self.horizontal_scale)
        length = int(1.5 / self.horizontal_scale)
        height = int(0.26 / self.vertical_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)
        slope = (height * self.vertical_scale) / (length * self.horizontal_scale)
        ratio = self.horizontal_scale / self.vertical_scale

        rects = []
        x_range = np.array(range(int(pos_x - length), int(pos_x) + 1))
        y_range = np.array(range(int(pos_y - width / 2), int(pos_y + width / 2) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile((x_range[:] - x_range[0]) * slope * ratio, (len(y_range), 1)).T)
        rects.append([x_range[0], y_range[0], length, width])

        x_range = np.array(range(int(pos_x), int(pos_x + length) + 1))
        y_range = np.array(range(int(pos_y - width / 2), int(pos_y + width / 2) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile(np.flip(x_range[:] - x_range[0]) * slope * ratio, (len(y_range), 1)).T)
        rects.append([x_range[0], y_range[0], length, width])

        rect_edge = []
        edge_size = 1
        x_edge_mask = np.zeros_like(terrain.height_field_raw)
        x_edge_mask[int(pos_x - length):int(pos_x + length) + 1,
        int(pos_y - width / 2):int(pos_y - width / 2) + edge_size + 1] = 1
        rect_edge.append([int(pos_x - length), int(pos_y - width / 2), 2 * length, edge_size])
        x_edge_mask[int(pos_x - length):int(pos_x + length) + 1,
        int(pos_y + width / 2) - edge_size:int(pos_y + width / 2) + 1] = 1
        rect_edge.append([int(pos_x - length), int(pos_y + width / 2) - edge_size, 2 * length, edge_size])
        terrain.x_edge_mask = x_edge_mask.astype(bool)

        rect_points = []
        rect_edge_points = []
        for rect in rects:
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])
        for rect in rect_edge:
            rect_edge_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                     [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 0.7
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        length *= self.horizontal_scale
        height *= self.vertical_scale
        goals[0] = np.array([pos_x - length - goal_x_step, pos_y, goal_z_bias])
        goals[1] = np.array([pos_x - length, pos_y, goal_z_bias])
        goals[2] = np.array([pos_x, pos_y, height + goal_z_bias])
        goals[3] = np.array([pos_x + length, pos_y, goal_z_bias])
        terrain.goals = goals

    def tire_jump_obstacle(self, terrain, obst_pos, joint_pos):
        radius = int((0.8 / 2) / self.horizontal_scale)
        width = int(1.5 / self.horizontal_scale)
        length_1 = int(0.2 / self.horizontal_scale)
        length_2 = int(0.6 / self.horizontal_scale)
        height_1 = int(joint_pos / self.vertical_scale)
        height_2 = int(1.5 / self.vertical_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)
        ratio = self.horizontal_scale / self.vertical_scale

        rects = []
        x_range = np.array(range(int(pos_x - length_1 / 2), int(pos_x + length_1 / 2) + 1))
        y_range = np.array(range(int(pos_y - radius), int(pos_y + radius) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile(self.get_circle_height((y_range[:] - y_range[0])) * ratio + height_1, (len(x_range), 1)))
        terrain.height_field_raw[int(pos_x - length_2 / 2):int(pos_x + length_2 / 2) + 1,
        int(pos_y - width / 2):int(pos_y - radius) + 1] = height_2
        terrain.height_field_raw[int(pos_x - length_2 / 2):int(pos_x + length_2 / 2) + 1,
        int(pos_y + radius):int(pos_y + width / 2) + 1] = height_2

        rects = [[int(pos_x - length_1 / 2), int(pos_y - radius), length_1, 2 * radius],
                 [int(pos_x - length_2 / 2), int(pos_y - width / 2), length_2, width / 2 - radius],
                 [int(pos_x - length_2 / 2), int(pos_y + radius), length_2, width / 2 - radius]]

        rect_points = []
        for rect in rects:
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        x_edge_mask = copy.deepcopy(terrain.height_field_raw)
        terrain.x_edge_mask = x_edge_mask.astype(bool)
        rect_edge_points = []

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 1.8
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        goals[0] = np.array([pos_x - goal_x_step, pos_y, goal_z_bias])
        goals[1] = np.array([pos_x - goal_x_step / 2, pos_y, goal_z_bias])
        goals[2] = np.array([pos_x, pos_y, joint_pos])
        goals[3] = np.array([pos_x + goal_x_step / 2, pos_y, goal_z_bias])
        terrain.goals = goals

    def tunnel_obstacle(self, terrain, obst_pos):
        radius = int((0.8 / 2) / self.horizontal_scale)
        length = int(2.0 / self.horizontal_scale)
        pos_x = int(obst_pos[0] / self.horizontal_scale)
        pos_y = int(obst_pos[1] / self.horizontal_scale)
        ratio = self.horizontal_scale / self.vertical_scale

        x_range = np.array(range(pos_x, int(pos_x + length) + 1))
        y_range = np.array(range(int(pos_y - radius), int(pos_y + radius) + 1))
        terrain.height_field_raw[x_range[0]:x_range[-1] + 1, y_range[0]:y_range[-1] + 1] = (
            np.tile((self.get_circle_height((y_range[:] - y_range[0])) + radius) * ratio, (len(x_range), 1)))

        rects = [[pos_x, int(pos_y - radius), length, 2 * radius]]

        x_edge_mask = np.zeros_like(terrain.height_field_raw)
        terrain.x_edge_mask = x_edge_mask.astype(bool)

        rect_points = []
        rect_edge_points = []
        for rect in rects:
            rect_points.append([[rect[0], rect[1]], [rect[0] + rect[2], rect[1]],
                                [rect[0] + rect[2], rect[1] + rect[3]], [rect[0], rect[1] + rect[3]]])

        terrain.rect_points = np.array(rect_points)
        terrain.rect_edge_points = np.array(rect_edge_points)

        goals = np.zeros((self.num_goals, 3))
        goal_x_step = 0.5
        goal_z_bias = 0.3
        pos_x *= self.horizontal_scale
        pos_y *= self.horizontal_scale
        goals[0] = np.array([pos_x - 2 * goal_x_step, pos_y, goal_z_bias])
        goals[1] = np.array([pos_x - goal_x_step, pos_y, goal_z_bias])
        goals[2] = np.array([pos_x + length * self.horizontal_scale / 2, pos_y, goal_z_bias])
        goals[3] = np.array([pos_x + length * self.horizontal_scale + goal_x_step, pos_y, goal_z_bias])
        terrain.goals = goals

    @staticmethod
    def get_circle_height(x_range):
        n = len(x_range) - 1
        h = -np.sqrt((n / 2) ** 2 - (x_range - n / 2) ** 2)
        return h
