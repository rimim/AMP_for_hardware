# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import glob
import os

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

#MOTION_FILES = glob.glob("datasets/bdx/placo_moves_faster/*")
#MOTION_FILES = ["datasets/go_bdx/placo_moves/bdx_walk_forward.txt"]
#MOTION_FILES = ["datasets/go_bdx/placo_moves/bdx_stand.txt"]
# MOTION_FILES = ["datasets/bdx/placo_moves_faster/bdx_walk_forward.txt"]
MOTION_FILES = [
     # "datasets/go_bdx/placo_moves/bdx_stand.txt",
     # "datasets/go_bdx/placo_moves/bdx_step_left.txt",
     # "datasets/go_bdx/placo_moves/bdx_step_right.txt",
     "datasets/go_bdx/placo_moves/bdx_walk_forward.txt",
     # "datasets/go_bdx/placo_moves/bdx_walk_backward.txt",
     # "datasets/go_bdx/placo_moves/bdx_turn_left.txt",
     # "datasets/go_bdx/placo_moves/bdx_turn_right.txt",
]


class GOBDXAMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 54  # 3+3+3+16+16+16
        num_privileged_obs = 60  # 3+3+4+3+16+16+16
        num_actions = 16
        env_spacing = 1.5
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        ee_names = ["left_foot", "right_foot"]
        get_commands_from_joystick = False
        get_commands_from_keyboard = False
        episode_length_s = 8  # episode length in seconds
        debug_save_obs = False
        debug_zero_action = False

    class init_state(LeggedRobotCfg.init_state):
        # pos = [0.0, 0.0, 0.3]  # x,y,z [m]
        pos = [0.0, 0.0, 0.0]  # x,y,z [m]
        if os.getenv('GYM_PLOT_COMMAND_ACTION_REF') is not None:
            pos = [0.0, 0.0, 0.3]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "left_hip_yaw": 0.0,  # [rad]
            "left_hip_roll": 0.0,  # [rad]
            "left_hip_pitch": 0.0,  # [rad]
            "left_knee": 0.0,  # [rad]
            "left_ankle": 0.0,  # [rad]
            "neck_pitch": 0.0,  # [rad]
            "head_pitch": 0.0,  # [rad]
            "head_yaw": 0.0,  # [rad]
            "head_roll": 0.0,  # [rad]
            "left_antenna": 0.0,  # [rad]
            "right_antenna": 0.0,  # [rad]
            "right_hip_yaw": 0.0,  # [rad]
            "right_hip_roll": 0.0,  # [rad]
            "right_hip_pitch": 0.0,  # [rad]
            "right_knee": 0.0,  # [rad]
            "right_ankle": 0.0,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        override_effort = False
        dof_friction = 0.01

        stiffness = {
            "left_hip_yaw": 40,
            "left_hip_roll": 40,
            "left_hip_pitch": 40,
            "left_knee": 35,
            "left_ankle": 30,
            "neck_pitch": 40,
            "head_pitch": 15,
            "head_yaw": 15,
            "head_roll": 15,
            "left_antenna": 3,
            "right_antenna": 3,
            "right_hip_yaw": 40,
            "right_hip_roll": 40,
            "right_hip_pitch": 40,
            "right_knee": 35,
            "right_ankle": 30,
        }
        damping = {
            "left_hip_yaw": 1.3,
            "left_hip_roll": 1.3,
            "left_hip_pitch": 1.3,
            "left_knee": 1.3,
            "left_ankle": 1.6,
            "neck_pitch": 1.3,
            "head_pitch": 1.0,
            "head_yaw": 1.0,
            "head_roll": 1.0,
            "left_antenna": 0.2,
            "right_antenna": 0.2,
            "right_hip_yaw": 1.3,
            "right_hip_roll": 1.3,
            "right_hip_pitch": 1.3,
            "right_knee": 1.3,
            "right_ankle": 1.6,
        }

        action_scale = 1.0

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        static_friction = 5.0  # 5
        dynamic_friction = 5.0  # 5

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go_bdx/go_bdx.urdf"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = [
            "pelvis",
            "head_body_roll",
            "left_thigh",
            "right_thigh",
            "left_shin",
            "right_shin",
            "left_hip",
            "right_hip",
        ]
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # default_dof_drive_mode = 0  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        disable_gravity = False
        fix_base_link = False  # fixe the base of the robot
        if os.getenv('GYM_PLOT_COMMAND_ACTION_REF') is not None:
            fix_base_link = True  # fixe the base of the robot

    # class normalization(LeggedRobotCfg.normalization):
    #     class obs_scales:
    #         lin_vel = 2.
    #         ang_vel = 1.
    #         dof_pos = 1.
    #         dof_vel = 0.05
    #         quat = 1.
    #         height_measurements = 5.0
    #     clip_observations = 5.0
    #     clip_actions = 1.0

    class sim(LeggedRobotCfg.sim):
        dt = 0.004
        substeps = 2

    class domain_rand:
        randomize_friction = False
        friction_range = [0.25, 1.05]
        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_torques = False
        torque_multiplier_range = [0.90, 1.1]
        randomize_com = False
        com_range = [-0.01, 0.01]

    class noise:
        add_noise = False
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.01
            lin_vel = 0.01
            ang_vel = 0.01
            gravity = 0.01
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.0

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.5 * 1.0 / (0.004 * 6)
            tracking_ang_vel = 0.5 * 1.0 / (0.004 * 6)
            # tracking_lin_vel = 0
            # tracking_ang_vel = 0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -1.0
            stand_still = 0.0
            dof_pos_limits = 0.0

    class commands:
        curriculum = False  # False
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        minimum_command_size = 0.1 # was 0.01

        class ranges:
            lin_vel_x = [0, 0.6]  # [0.4, 0.4] #[-0.2, 1.0]  # min max [m/s]
            lin_vel_y = [0, 0]  # [-0.3, 0.3] #[-0.1836, 0.1836]  # min max [m/s]
            ang_vel_yaw = [0, 0]  # [-1.57, 1.57]  # min max [rad/s]
            # ang_vel_yaw = [-0.4, 0.4]  # [-1.57, 1.57]  # min max [rad/s]
            heading = [0, 0]
            # lin_vel_x = [0.1, 0.2]  # min max [m/s]
            # lin_vel_y = [0.0, 0.0]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [-3.14, 3.14]

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [0, 0, 1]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]


class GOBDXAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "AMPOnPolicyRunner"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        disc_coef = 5  # TUNE ?

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "go_bdx_amp"
        algorithm_class_name = "AMPPPO"
        policy_class_name = "ActorCritic"
        max_iterations = 100000  # number of policy updates

        amp_reward_coef = 2.0  # 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.2  # 0.3
        amp_discr_hidden_dims = [1024, 512]

        # disc_grad_penalty = 1  # original 10 # TUNE ?
        # smaller penalty is needed for high-dynamic mocap
        disc_grad_penalty = 0.1  # original 10 # TUNE ?

        # min_normalized_std = [0.05, 0.02, 0.05] * 4

        min_normalized_std = [0.02] * 16  # WARNING TOTALLY PIFFED

        pass
