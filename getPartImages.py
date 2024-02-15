# Virtual Environment name: capstone
# getPartImages.py FUNCTION -- Obtain top-down transparent images of a part

# https://trevorsandy.github.io/lpub3d/assets/docs/ldview/Help.html

import subprocess
import os

# Define your part number and part name as variables
part_number = "57585"
part_name = "4502595"
part_color = "0x8c8c8c" # black 0x333333 | red 0xff0000 | white 0xd6d6d6 | dark grey 0x8c8c8c | tan 0xd2b48c | blue 0x2d6fcb | yellow 0xf9da4e | green 0x2fac56
orientation = 2 # 1 for 180 (symmetrical), 2 for 360 (asymmetrical)

# Path to LDView application bundle on a Mac
ldview_app_path = "LDView.app/Contents/MacOS/LDView"

# Path to your LDraw file
ldraw_file = f"/Users/kyvang/Documents/Capstone/data/{part_number} ({part_name})/{part_number}.dat"

# Path to save snapshots
output_dir = f"/Users/kyvang/Documents/Capstone/data/{part_number} ({part_name})/images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through DefaultLatitude/Longitude values
count = 0

if orientation == 1: 
    for i in range(0, 171, 10):
        ldview_command = [
            ldview_app_path,
            ldraw_file,
            f"-DefaultColor3={part_color}", 
            f"-DefaultLatitude=90",
            f"-DefaultLongitude={i}",
            f"-SaveAlpha=True",
            f"-AutoCrop=True",
            f"-SaveSnapshot={output_dir}/{part_number}-{count:04d}.png"
        ]
        count += 1
        # Execute the LDView command
        subprocess.run(ldview_command)
elif orientation == 2: 
    for i in range(0, 351, 20):
        ldview_command = [
            ldview_app_path,
            ldraw_file,
            f"-DefaultColor3={part_color}", 
            f"-DefaultLatitude=90",
            f"-DefaultLongitude={i}",
            f"-SaveAlpha=True",
            f"-AutoCrop=True",
            f"-SaveSnapshot={output_dir}/{part_number}-{count:04d}.png"
        ]
        count += 1
        # Execute the LDView command
        subprocess.run(ldview_command)

print("Snapshot generation complete.")