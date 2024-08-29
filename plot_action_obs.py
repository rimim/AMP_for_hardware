import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Plot DOF poses and actions.')
parser.add_argument('-png', action='store_true', help='Save the plot as a PNG file')
parser.add_argument('--width', type=int, default=1024, help='Width of the output image in pixels')
parser.add_argument('--height', type=int, default=768, help='Height of the output image in pixels')
args = parser.parse_args()

# Calculate figsize based on the specified width and height
dpi = 96
figsize = (args.width / dpi, args.height / dpi)

obses = pickle.load(open("saved_obs.pkl", "rb"))
num_dofs = 16
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[6 : 6 + 16][i])
        actions[i].append(obs[-16:][i])

isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

# Plot action vs dof pos
nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True, figsize=figsize)  # Set figure size

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos")
        axs[i, j].legend()
        axs[i, j].set_title(f"{isaac_joints_order[i * nb_cols + j]}")

# Reduce margin
plt.tight_layout()

if args.png:
    plt.savefig("plot.png", dpi=dpi)  # Use specified width and height
else:
    plt.show()
