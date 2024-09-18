import argparse
import pickle
import time
import sys

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import onnxruntime
import pygame

import cv2  # OpenCV is required for this

def make_action_dict(action, joints_order):
    action_dict = {}
    for i, a in enumerate(action):
        if "antenna" not in joints_order[i]:
            action_dict[joints_order[i]] = a

    return action_dict

class OnnxInfer:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        self.ort_session = onnxruntime.InferenceSession(
            self.onnx_model_path, providers=["CPUExecutionProvider"]
        )

    def infer(self, inputs):
        # outputs = self.ort_session.run(None, {"obs": [inputs]})
        # return outputs[0][0]
        outputs = self.ort_session.run(None, {"obs": inputs.astype("float32")})
        return outputs[0]

def action_to_pd_targets(action, pd_action_offset, pd_action_scale):
    return pd_action_offset + pd_action_scale * action

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("--duration", type=int, default=8, help="Number of seconds for video")
parser.add_argument('--width', type=int, help='Width of the viewer window (optional)')
parser.add_argument('--height', type=int, help='Height of the viewer window (optional)')
parser.add_argument('--video-width', type=int, help='Width of the output video in pixels')
parser.add_argument('--video-height', type=int, help='Height of the output video in pixels')
parser.add_argument("--video", type=str, required=False)
parser.add_argument('--hide-menu', action='store_true', help='Hide the viewer menu')
parser.add_argument("--saved_obs", type=str, required=False)
parser.add_argument("--saved_actions", type=str, required=False)
parser.add_argument("--xbox", action="store_true", default=False)
args = parser.parse_args()

if args.saved_obs is not None:
    saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
if args.saved_actions is not None:
    saved_actions = pickle.loads(open("saved_actions.pkl", "rb").read())

if args.xbox:
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        print("No joystick connected!")
        sys.exit()

    # Select the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    # Print some information about the controller
    print(f"Joystick Name: {joystick.get_name()}")
    print(f"Number of Axes: {joystick.get_numaxes()}")
    print(f"Number of Buttons: {joystick.get_numbuttons()}")

# Params
# dt = 0.002
dt = 0.004
linearVelocityScale = 1.0
angularVelocityScale = 1.0
dof_pos_scale = 1.0
dof_vel_scale = 0.1
action_scale = 0.85
num_actions = 16

model = mujoco.MjModel.from_xml_path("resources/robots/go_bdx/scene.xml")
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
if args.width and args.height:
    viewer = mujoco_viewer.MujocoViewer(model, data, width=args.width, height=args.height, hide_menus=args.hide_menu)
else:
    viewer = mujoco_viewer.MujocoViewer(model, data, hide_menus=args.hide_menu)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
fps = 33  # Frames per second
if args.video_width and args.video_height:
    frame_width = args.video_width
    frame_height = args.video_height
else:
    frame_width = viewer.viewport.width
    frame_height = viewer.viewport.height
#model.opt.gravity[:] = [0, 0, 0]  # no gravity
if args.hide_menu:
    viewer._hide_menu = True  # Assuming the MujocoViewer supports this attribute

video_writer = None
if args.video is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.video, fourcc, fps, (frame_width, frame_height))

policy = OnnxInfer(args.onnx_model_path)

class ImuDelaySimulator:
    def __init__(self, delay_ms):
        self.delay_ms = delay_ms
        self.rot = []
        self.ang_rot = []
        self.t0 = None

    def push(self, rot, ang_rot, t):
        self.rot.append(rot)
        self.ang_rot.append(ang_rot)
        if self.t0 is None:
            self.t0 = t

    def get(self):
        if time.time() - self.t0 < self.delay_ms / 1000:
            return [0, 0, 0, 0], [0, 0, 0]
        rot = self.rot.pop(0)
        ang_rot = self.ang_rot.pop(0)

        return rot, ang_rot

def quat_rotate_inverse(q, v):
    q = np.array(q)
    v = np.array(v)

    q_w = q[-1]
    q_vec = q[:3]

    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0

    return a - b + c

def get_obs(data, action, commands, imu_delay_simulator: ImuDelaySimulator):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)

    base_lin_vel = (
        data.sensor("linear-velocity").data.astype(np.double) * linearVelocityScale
    )
    # print(f"base_lin_vel: {base_lin_vel}")

    base_quat = data.qpos[3 : 3 + 4].copy()
    base_quat = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]

    # # Remove yaw component
    # rot_euler = R.from_quat(base_quat).as_euler("xyz", degrees=False)
    # rot_euler[2] = 0
    # base_quat = R.from_euler("xyz", rot_euler, degrees=False).as_quat()

    base_ang_vel = (
        data.sensor("angular-velocity").data.astype(np.double) * angularVelocityScale
    )

    dof_pos = data.qpos[7 : 7 + 16].copy()

    dof_pos_scaled = dof_pos * dof_pos_scale

    dof_vel = data.qvel[6 : 6 + 16].copy()
    dof_vel_scaled = list(np.array(dof_vel) * dof_vel_scale)

    imu_delay_simulator.push(base_quat, base_ang_vel, time.time())
    base_quat, base_ang_vel = imu_delay_simulator.get()

    projected_gravity = quat_rotate_inverse(base_quat, [0, 0, -1])

    obs = np.concatenate(
        [
            projected_gravity,
            commands,
            dof_pos_scaled,
            dof_vel_scaled,
            action,
        ]
    )

    return (q, dq, obs)

def reset_robot():
    new_base_position = [0.0, 0.0, 0.0]  # x, y, z coordinates
    new_base_orientation = [0.0, 0.0, 0.0, 0.0]  # quaternion [w, x, y, z]

    data.qpos[0:3] = new_base_position  # First 3 elements represent position (x, y, z)
    data.qpos[3:7] = new_base_orientation  # Next 4 elements represent orientation (quaternion)

    # Optionally reset the joint positions (after the floating base state)
    data.qpos[7:] = initial_joint_positions  # Replace with joint positions if needed

    data.qvel[:] = 0  # Reset all velocities (base and joint velocities)

class default_command_velocities:
    lin_vel = 0.5
    ang_vel = 0.2

class obs_scales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0

prev_action = np.zeros(16)
commands = [default_command_velocities.lin_vel * obs_scales.lin_vel, 0.0, 0.0]
if args.xbox:
    commands = [0.0, 0.0, 0.0]
# prev = time.time()
# last_control = time.time()
prev = data.time
last_control = data.time
control_freq = 60  # hz
i = 0
data.qpos[3 : 3 + 4] = [1, 0, 0, 0]
cutoff_frequency = 20
initial_joint_positions = data.qpos[7:].copy()

data.qpos[7 : 7 + 16] = np.zeros(16)
data.ctrl[:] = np.zeros(16)

bdx_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bdx")

if bdx_index == -1:
    print("Body 'bdx' not found in model. Bodies in the model:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"Body ID {i}: {body_name}")

saved_obs = []
saved_actions = []
command_value = []
imu_delay_simulator = ImuDelaySimulator(1)
start = time.time()
sim_step = 0
sim_maxsteps = int(args.duration * fps)  # Number of frames to simulate
viewer.cam.azimuth += 90  # Rotate the camera 90 degrees counterclockwise
viewer.cam.elevation = -10  # Set the camera to be level with the ground
decimation = 2
running = True

# # Fix the base body position and disable dynamics
# data.qpos[:3] = np.array([0, 0, 1])  # Position the base 1 meter above the ground
# data.qvel[:3] = np.zeros(3)  # Zero out velocities to prevent movement

# # Optionally disable gravity if you don't want any downward force
# model.opt.gravity[:] = 0


# Get the gravity vector (assuming negative z-axis gravity)
gravity = np.linalg.norm(model.opt.gravity)

# Initialize total mass to 0
total_mass = 0.0

# Sum the masses of all bodies in the model
for body_id in range(model.nbody):
    total_mass += model.body_mass[body_id]

# Compute the weight (mass * gravity)
robot_weight = total_mass * gravity
buttons_pressed_mask = 0

# Print the total weight
print(f"Total robot weight: {robot_weight:.2f} N")

left_antenna_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_antenna") - 1
right_antenna_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_antenna") - 1
neck_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch") - 1
head_pitch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch") - 1
head_roll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_yaw") - 1

# left_antenna_qpos_adr = model.jnt_qposadr[left_antenna_id]
# right_antenna_qpos_adr = model.jnt_qposadr[right_antenna_id]
# neck_pitch_qpos_adr = model.jnt_qposadr[neck_pitch_id]
# head_roll_qpos_adr = model.jnt_qposadr[head_roll_id]

move_ears_back = False
move_ears_forward = False
move_head_down = False
move_head_up = False
turn_head_left = False
turn_head_right = False
ear_action_scale = 0.2
head_action_scale = 0.1
ears_back_position = 1.0
head_target_position = 1.0

try:
    start = time.time()
    while running:
        # t = time.time()
        t = data.time
        if time.time() - start < 1:
            last_control = t
        q, dq, obs = get_obs(data, prev_action, commands, imu_delay_simulator)
        q = q[-num_actions:]
        dq = dq[-num_actions:]
        if sim_step % decimation == 0:
            # saved_obs.append(obs)

            # if args.saved_obs is not None:
            #     obs = saved_obs[i]  # works with saved obs

            action = policy.infer(obs)
            if args.saved_actions is not None:
                action = saved_actions[i][0]
            prev_action_left_ear = prev_action[left_antenna_id]
            prev_action_right_ear = prev_action[right_antenna_id]
            prev_action_head_roll = prev_action[head_roll_id]
            prev_action_head_pitch = prev_action[head_pitch_id]
            prev_action_neck_pitch = prev_action[neck_pitch_id]
            prev_action = action.copy()
            if move_ears_back:
                prev_action_left_ear += ear_action_scale * (ears_back_position - prev_action_left_ear)
                prev_action_right_ear += ear_action_scale * (ears_back_position - prev_action_right_ear)
                prev_action[left_antenna_id] = prev_action_left_ear
                prev_action[right_antenna_id] = prev_action_right_ear
                action[left_antenna_id] = prev_action[left_antenna_id]
                action[right_antenna_id] = prev_action[right_antenna_id]
            elif move_ears_forward:
                prev_action_left_ear += ear_action_scale * (-ears_back_position - prev_action_left_ear)
                prev_action_right_ear += ear_action_scale * (-ears_back_position - prev_action_right_ear)
                prev_action[left_antenna_id] = prev_action_left_ear
                prev_action[right_antenna_id] = prev_action_right_ear
                action[left_antenna_id] = prev_action[left_antenna_id]
                action[right_antenna_id] = prev_action[right_antenna_id]
            elif turn_head_left:
                prev_action_head_roll += head_action_scale * (head_target_position - prev_action_head_roll)
                prev_action[head_roll_id] = prev_action_head_roll
                action[head_roll_id] = prev_action[head_roll_id]
            elif turn_head_right:
                prev_action_head_roll += head_action_scale * (-head_target_position - prev_action_head_roll)
                prev_action[head_roll_id] = prev_action_head_roll
                action[head_roll_id] = prev_action[head_roll_id]
            elif move_head_down:
                prev_action_head_pitch += head_action_scale * (head_target_position - prev_action_head_pitch)
                prev_action_neck_pitch += head_action_scale * (head_target_position - prev_action_neck_pitch)
                prev_action[head_pitch_id] = prev_action_head_pitch
                prev_action[neck_pitch_id] = prev_action_neck_pitch
                action[head_pitch_id] = prev_action[head_pitch_id]
                action[neck_pitch_id] = prev_action[neck_pitch_id]
            elif move_head_up:
                prev_action_head_pitch += head_action_scale * (-head_target_position - prev_action_head_pitch)
                prev_action_neck_pitch += head_action_scale * (-head_target_position - prev_action_neck_pitch)
                prev_action[head_pitch_id] = prev_action_head_pitch
                prev_action[neck_pitch_id] = prev_action_neck_pitch
                action[head_pitch_id] = prev_action[head_pitch_id]
                action[neck_pitch_id] = prev_action[neck_pitch_id]

            action = action * action_scale

            last_control = t

            target_q = action

            data.ctrl[:] = action.copy()
            # saved_actions.append(action)

            # command_value.append([data.ctrl.copy(), data.qpos[7:].copy()])

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.xpos[bdx_index]
        sim_step = sim_step + 1
        if video_writer is not None:
            print(f"sim_step: {sim_step}:{sim_maxsteps}")
            sys.stdout.flush() 

        viewer.render()

        if args.xbox:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYAXISMOTION:
                    # # Axis motion event (analog stick)
                    # for i in range(joystick.get_numaxes()):
                    #     axis_value = joystick.get_axis(i)
                    #     print(f"Axis {i}: {axis_value:.2f}")
                    lin_vel_x = joystick.get_axis(0) * default_command_velocities.lin_vel
                    lin_vel_y = 0
                    ang_vel = joystick.get_axis(1) * default_command_velocities.ang_vel
                    lin_vel_x = round(lin_vel_x,1) * obs_scales.lin_vel
                    lin_vel_y = round(lin_vel_y,1)
                    ang_vel = round(ang_vel,1)
                    if abs(lin_vel_x) != 0 or abs(ang_vel) != 0:
                        print(f"lin_vel_x: {lin_vel_x} ang_vel: {ang_vel}")
                    commands[0] = lin_vel_x
                    commands[1] = lin_vel_y
                    commands[2] = ang_vel #* obs_scales.ang_vel

                elif event.type == pygame.JOYBUTTONDOWN:
                    # Button press event
                    for i in range(joystick.get_numbuttons()):
                        if joystick.get_button(i):
                            buttons_pressed_mask = buttons_pressed_mask | (1 << i)
                            if i == 0:
                                turn_head_left = True
                            elif i == 1:
                                move_head_up = True
                            elif i == 3:
                                turn_head_right = True
                            elif i == 4:
                                move_head_down = True
                            elif i == 6:
                                move_ears_forward = True
                            elif i == 7:
                                move_ears_back = True
                elif event.type == pygame.JOYBUTTONUP:
                    # Button release event
                    for i in range(joystick.get_numbuttons()):
                        if (buttons_pressed_mask & (1 << i)) >> i == 1:
                            buttons_pressed_mask = buttons_pressed_mask & ~(1 << i)
                            if i == 0:
                                turn_head_left = False
                            elif i == 1:
                                move_head_up = False
                            elif i == 2:
                                reset_robot()
                            elif i == 3:
                                turn_head_right = False
                            elif i == 4:
                                move_head_down = False
                            elif i == 6:
                                move_ears_forward = False
                            elif i == 7:
                                move_ears_back = False
                            else:
                                print(f"Button #{i} pressed")

            # Exit on keypress
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                running = False

        if video_writer is not None:
            img = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            viewport = mujoco.MjrRect(0, 0, frame_width, frame_height)
            mujoco.mjr_readPixels(img, None, viewport, context)  # Correct arguments with manual context

            img = np.flipud(img)

            # Convert the image to BGR (OpenCV uses BGR format)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Write the frame to the video file
            video_writer.write(img)
            if sim_step >= sim_maxsteps:
                break

        # Check for collisions after the step
        for i in range(data.ncon):  # Iterate over all contacts
            contact = data.contact[i]
            
            geom1 = contact.geom1  # Geom ID 1 involved in the contact
            geom2 = contact.geom2  # Geom ID 2 involved in the contact
            
            # Get the body IDs associated with the geoms
            body1 = model.geom_bodyid[geom1]
            body2 = model.geom_bodyid[geom2]
            
            # Get the body names
            body_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
            body_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
            
            # Handle case where body name is None
            if body_name1 is None:
                body_name1 = f"Body {body1}"
            if body_name2 is None:
                body_name2 = f"Body {body2}"
            if body_name1 == "floor" and body_name2 == "pelvis":
                print(f"Collision detected between {body_name1} and {body_name2}")
                reset_robot()
            if body_name1 != "floor" and body_name2 != "floor":            
                print(f"Collision detected between {body_name1} and {body_name2}")

        prev = t
except KeyboardInterrupt:
    data = {
        "config": {},
        "mujoco": command_value,
    }
    # pickle.dump(data, open("mujoco_command_value.pkl", "wb"))
    # pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))
    # pickle.dump(saved_actions, open("mujoco_saved_actions.pkl", "wb"))

if video_writer is not None:
    video_writer.release()
    cv2.destroyAllWindows()
