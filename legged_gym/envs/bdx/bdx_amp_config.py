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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob("datasets/bdx/new_placo_moves/*")
MOTION_FILES = [
    "datasets/bdx/new_placo_moves/bdx_walk_forward.txt",
    # "datasets/bdx/placo_moves/bdx_turn_left.txt",
    # "datasets/bdx/placo_moves/bdx_turn_right.txt",
]

NO_FEET = False  # Do not use feet in the amp observations and data


class BDXAMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 8
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 51  # TODO what ?
        num_privileged_obs = 57
        num_actions = 15
        env_spacing = 1.0
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        ee_names = ["left_foot", "right_foot"]
        get_commands_from_joystick = False
        get_commands_from_keyboard = False
        episode_length_s = 8  # episode length in seconds
        debug_save_obs = False
        no_feet = NO_FEET

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.16]  # x,y,z [m]
        # pos = [0.0, 0.0, 0.3]  # x,y,z [m]
        rot = [0, -0.08, 0, 1]

        # default_joint_angles = {
        #     "left_hip_yaw": -0.0038368461980583203,
        #     "left_hip_roll": 0.014431210831554308,
        #     "left_hip_pitch": 1.133082668956462,
        #     "left_knee": -1.4620636255950707,
        #     "left_ankle": 0.590808248377727,
        #     "neck_pitch": -0.17453292519943295,
        #     "head_pitch": -0.17453292519943295,
        #     "head_yaw": 0,
        #     "left_antenna": 0,
        #     "right_antenna": 0,
        #     "right_hip_yaw": 0.0019449075083605783,
        #     "right_hip_roll": 0.007371299077063522,
        #     "right_hip_pitch": 1.1351700102241071,
        #     "right_knee": -1.4585281663702696,
        #     "right_ankle": 0.5851648236531114,
        # }
        default_joint_angles = {
            "left_hip_yaw": -0.002853397830292128,
            "left_hip_roll": 0.01626303761810685,
            "left_hip_pitch": 1.0105624704499077,
            "left_knee": -1.4865015965817336,
            "left_ankle": 0.6504953719748071,
            "neck_pitch": -0.17453292519943295,
            "head_pitch": -0.17453292519943295,
            "head_yaw": 0,
            "left_antenna": 0,
            "right_antenna": 0,
            "right_hip_yaw": 0.001171696610228082,
            "right_hip_roll": 0.006726989242258406,
            "right_hip_pitch": 1.0129772861831692,
            "right_knee": -1.4829304760981399,
            "right_ankle": 0.6444901047812701,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        override_effort = True
        effort = 0.93  # Nm
        # effort = 0.52  # Nm

        # dof_friction = 0.01  # 0.01

        stiffness_all = 10  # 10 [N*m/rad]
        damping_all = 0.03  # 0.03
        stiffness = {
            "left_hip_yaw": stiffness_all,
            "left_hip_roll": stiffness_all,
            "left_hip_pitch": stiffness_all,
            "left_knee": stiffness_all,
            "left_ankle": stiffness_all,
            "neck_pitch": stiffness_all,
            "head_pitch": stiffness_all,
            "head_yaw": stiffness_all,
            "left_antenna": 1,
            "right_antenna": 1,
            "right_hip_yaw": stiffness_all,
            "right_hip_roll": stiffness_all,
            "right_hip_pitch": stiffness_all,
            "right_knee": stiffness_all,
            "right_ankle": stiffness_all,
        }

        damping = {
            "left_hip_yaw": damping_all,
            "left_hip_roll": damping_all,
            "left_hip_pitch": damping_all,
            "left_knee": damping_all,
            "left_ankle": damping_all,
            "neck_pitch": damping_all,
            "head_pitch": damping_all,
            "head_yaw": damping_all,
            "left_antenna": 0,
            "right_antenna": 0,
            "right_hip_yaw": damping_all,
            "right_hip_roll": damping_all,
            "right_hip_pitch": damping_all,
            "right_knee": damping_all,
            "right_ankle": damping_all,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # 0.25
        # action_scale = 0.5  # 0.25
        # action_scale = 1.0  # 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4  # 4

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0, 1.0, 0, 0, 0.0]
        # trimesh only:
        # slope_treshold = (
        #     0.75  # slopes above this threshold will be corrected to vertical surfaces
        # )
        # vertical_scale = 0.001  # [m]

        # mesh_type = "plane"
        measure_heights = False
        static_friction = 5.0  # 5
        dynamic_friction = 5.0  # 5

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bdx/urdf/bdx.urdf"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = [
            "body_module",
            "head",
            "left_antenna",
            "right_antenna",
            # "leg_module",
            # "leg_module_2",
        ]
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        # default_dof_drive_mode = 0  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        disable_gravity = False
        fix_base_link = False  # fix the base of the robot

        # angular_damping = 0.05
        # linear_damping = 0.0
        # armature = 0.0
        # thickness = 0.01

    # class normalization(LeggedRobotCfg.normalization):
    #     clip_observations = 5.0
    #     clip_actions = 1.0

    class sim(LeggedRobotCfg.sim):
        # dt = 0.004  # 0.004
        dt = 0.004
        substeps = 2  # 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.8, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.2, 0.2]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.5  # 0.3
        randomize_gains = False
        stiffness_multiplier_range = [0.95, 1.05]
        damping_multiplier_range = [0.95, 1.05]
        randomize_torques = True
        torque_multiplier_range = [0.8, 1.2]
        randomize_com = True
        com_range = [-0.04, 0.04]

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.05  # 1.5
            lin_vel = 0.05
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.16
        tracking_sigma = 0.05  # tracking reward = exp(-error^2/sigma)

        class scales(LeggedRobotCfg.rewards.scales):
            termination = 0.0
            tracking_lin_vel = 1.5 * 1.0 / (0.004 * 4)
            tracking_ang_vel = 0.5 * 1.0 / (0.004 * 4)
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = -0.000025  # -0.000025
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0
            feet_air_time = 0.0
            collision = 0.0
            feet_stumble = 0.0
            action_rate = -1.0
            stand_still = 0.0
            dof_pos_limits = 0.0
            # motion_imitation = 5.0

    class commands:
        curriculum = False  # False
        max_curriculum = 0.2
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [0.0, 0.1]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]
            heading = [0, 0]
            # lin_vel_x = [0.1, 0.2]  # min max [m/s]
            # lin_vel_y = [0.0, 0.0]  # min max [m/s]
            # ang_vel_yaw = [0.0, 0.0]  # min max [rad/s]
            # heading = [-3.14, 3.14]

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [0, 0, 1]  # [m]
        lookat = [11.0, 5, 1.0]  # [m]


class BDXAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "AMPOnPolicyRunner"

    # class policy(LeggedRobotCfgPPO.policy):
    #     actor_hidden_dims = [1024, 512]  # [512, 256, 128]
    #     critic_hidden_dims = [1024, 512]  # [512, 256, 128]
    #     activation = "relu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01  # 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4
        disc_coef = 5  # TUNE ?
        # bounds_loss_coef = 10

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "bdx_amp"
        algorithm_class_name = "AMPPPO"
        policy_class_name = "ActorCritic"
        max_iterations = 500000  # number of policy updates

        no_feet = NO_FEET

        amp_reward_coef = 2.0  # 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.2  # 0.3
        amp_discr_hidden_dims = [1024, 512]

        disc_grad_penalty = 0.1  # original 10 # TUNE ?

        # min_normalized_std = [0.05, 0.02, 0.05] * 4

        min_normalized_std = [0.02] * 15  # WARNING TOTALLY PIFFED

        pass
