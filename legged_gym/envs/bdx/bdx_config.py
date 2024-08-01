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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BDXRoughCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        # num_envs = 5480
        num_envs = 16
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 51
        num_privileged_obs = 57
        num_actions = 15
        env_spacing = 1.0
        reference_state_initialization = False
        # reference_state_initialization_prob = 0.85
        # amp_motion_files = MOTION_FILES

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.175]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "right_hip_yaw": -0.03676731090962078,  # [rad]
            "right_hip_roll": -0.030315211140564333,  # [rad]
            "right_hip_pitch": 0.4065815100399598,  # [rad]
            "right_knee": -1.0864064934571644,  # [rad]
            "right_ankle": 0.5932324840794684,  # [rad]
            "left_hip_yaw": -0.03485756878823724,  # [rad]
            "left_hip_roll": 0.052286054888550475,  # [rad]
            "left_hip_pitch": 0.36623601032755765,  # [rad]
            "left_knee": -0.964204465274923,  # [rad]
            "left_ankle": 0.5112970996901808,  # [rad]
            "neck_pitch": -0.17453292519943295,  # [rad]
            "head_pitch": -0.17453292519943295,  # [rad]
            "head_yaw": 0,  # [rad]
            "left_antenna": 0.0,  # [rad]
            "right_antenna": 0.0,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = "P"
        stiffness = {
            "left_hip_yaw": 10.0,
            "left_hip_roll": 10.0,
            "left_hip_pitch": 10.0,
            "left_knee": 10.0,
            "left_ankle": 10.0,
            "right_hip_yaw": 10.0,
            "right_hip_roll": 10.0,
            "right_hip_pitch": 10.0,
            "right_knee": 10.0,
            "right_ankle": 10.0,
            "neck_pitch": 10.0,
            "head_pitch": 10.0,
            "head_yaw": 10.0,
            "left_antenna": 10.0,
            "right_antenna": 10.0,
        }  # [N*m/rad]

        damping = {
            "left_hip_yaw": 0.5,
            "left_hip_roll": 0.5,
            "left_hip_pitch": 0.5,
            "left_knee": 0.5,
            "left_ankle": 0.5,
            "right_hip_yaw": 0.5,
            "right_hip_roll": 0.5,
            "right_hip_pitch": 0.5,
            "right_knee": 0.5,
            "right_ankle": 0.5,
            "neck_pitch": 0.5,
            "head_pitch": 0.5,
            "head_yaw": 0.5,
            "left_antenna": 0.5,
            "right_antenna": 0.5,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        # action_scale = 1
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1
        # decimation = 6

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bdx/urdf/bdx.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["left_knee" "right_knee"]
        terminate_after_contacts_on = ["body_module"]
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class BDXRoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_bdx"
