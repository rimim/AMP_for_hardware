import argparse
import pickle
import time

import mujoco
import mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import onnxruntime

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
args = parser.parse_args()

if args.saved_obs is not None:
    saved_obs = pickle.loads(open("saved_obs.pkl", "rb").read())
if args.saved_actions is not None:
    saved_actions = pickle.loads(open("saved_actions.pkl", "rb").read())

# Params
# dt = 0.002
dt = 0.005
linearVelocityScale = 1.0
angularVelocityScale = 0.25
dof_pos_scale = 1.0
dof_vel_scale = 0.1
action_scale = 0.75
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
# model.opt.gravity[:] = [0, 0, 0]  # no gravity
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

prev_action = np.zeros(16)
#commands = [0.38, 0.0, 0.0]
commands = [0.0, 0.0, 0.0]
# prev = time.time()
# last_control = time.time()
prev = data.time
last_control = data.time
control_freq = 85  # hz
i = 0
data.qpos[3 : 3 + 4] = [1, 0, 0, 0]
cutoff_frequency = 20

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
try:
    start = time.time()
    while True:
        # t = time.time()
        t = data.time
        if time.time() - start < 1:
            last_control = t
        q, dq, obs = get_obs(data, prev_action, commands, imu_delay_simulator)
        q = q[-num_actions:]
        dq = dq[-num_actions:]
        if sim_step % decimation == 0:
            saved_obs.append(obs)

            if args.saved_obs is not None:
                obs = saved_obs[i]  # works with saved obs

            action = policy.infer(obs)
            if args.saved_actions is not None:
                action = saved_actions[i][0]
            prev_action = action.copy()

            action = action * action_scale

            last_control = t

            target_q = action

            data.ctrl[:] = action.copy()
            saved_actions.append(action)

            command_value.append([data.ctrl.copy(), data.qpos[7:].copy()])

        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.xpos[bdx_index]
        sim_step = sim_step + 1

        viewer.render()

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

        prev = t
except KeyboardInterrupt:
    data = {
        "config": {},
        "mujoco": command_value,
    }
    pickle.dump(data, open("mujoco_command_value.pkl", "wb"))
    pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))
    pickle.dump(saved_actions, open("mujoco_saved_actions.pkl", "wb"))

if video_writer is not None:
    video_writer.release()
    cv2.destroyAllWindows()
