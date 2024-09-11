import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
import json

# Argument parser
parser = argparse.ArgumentParser(description='Plot x and y data from CSV or JSON files.')
parser.add_argument('directory', type=str, help='Directory containing CSV or JSON files')
parser.add_argument('-png', action='store_true', help='Save the plot as a PNG file')
parser.add_argument('--width', type=int, default=1024, help='Width of the output image in pixels')
parser.add_argument('--height', type=int, default=768, help='Height of the output image in pixels')
args = parser.parse_args()

# Calculate figsize based on the specified width and height
dpi = 96
figsize = (args.width / dpi, args.height / dpi)

# Get all CSV and JSON files in the specified directory
csv_files = glob.glob(os.path.join(args.directory, "*.csv"))
json_files = glob.glob(os.path.join(args.directory, "*.json"))
all_files = sorted(csv_files + json_files)  # Sort file names alphabetically

# Determine the number of plots needed
nb_files = len(all_files)
nb_cols = 4
nb_rows = int(np.ceil(len(all_files) / nb_cols))

# Create subplots
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)

for idx, file in enumerate(all_files):
    print(f"Parsing: {file}")
    try:
        if file.endswith(".csv"):
            data = np.loadtxt(file, delimiter=',', skiprows=1)  # Assuming data starts from the second row
            x = data[:, 1]
            y = data[:, 2]
        elif file.endswith(".json"):
            with open(file, 'r') as f:
                data = json.load(f)
            x = np.array([entry[1] for entry in data])  # Extracting y from the JSON arrays
            y = np.array([entry[2] for entry in data])  # Extracting y from the JSON arrays
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

    row = idx // nb_cols
    col = idx % nb_cols
    axs[row, col].plot(x, y, color='orange')  # Use orange color to match the screenshot
    axs[row, col].set_xlim([0, 500])  # Ensure x-axis goes from 0 to 500
    axs[row, col].set_ylim([min(y), max(y)])

    # Set the title as the filename without the extension
    file_title = os.path.splitext(os.path.basename(file))[0]
    axs[row, col].set_title(file_title)
    axs[row, col].set_xlabel('Time')

# Hide any unused subplots
for i in range(nb_files, nb_rows * nb_cols):
    fig.delaxes(axs.flatten()[i])

for ax in axs.flat:
    ax.label_outer()

# Reduce margin
plt.tight_layout()

# Save or show the plot
if args.png:
    plt.savefig("tensorboard.png", dpi=dpi)  # Use specified width and height
else:
    plt.show()
