import os
import sys

base_folder = "/home/mgambino"
folder = f"{base_folder}/reduce_db/runs"
code_base_folder = f'{base_folder}/vg/images_selection/geo_class_v3/reduce_panoramas'

if not os.path.abspath(os.curdir) == folder: 
    print(f"Not in the right folder!!!!\ncd {folder}")
    sys.exit()

if not os.path.isdir(f'{base_folder}/descriptors/positives/r50_512'):
    os.makedirs(f'{base_folder}/descriptors/positives/r50_512')

if not os.path.isdir(f'{base_folder}/panoramas/r50_512'):
    os.makedirs(f'{base_folder}/panoramas/r50_512')

CONTENT = \
f"""#!/bin/bash
#SBATCH --job-name EXP_NAME
#SBATCH --cpus-per-task 24
#SBATCH --mem MEMORYGB
#SBATCH --time 240:00:00
#SBATCH --output {folder}/out_job/out_EXP_NAME.txt
#SBATCH --error {folder}/out_job/err_EXP_NAME.txt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition global
source /home/gmberton/anaconda3/bin/activate
python {code_base_folder}/clean.py --method EXP_NAME PARAMS
"""
mem = 64
for method in ["descriptor_similarity", "distance","base"]:
    for conf in range(1,4): 
        exp_name = f"reduce_{method}_{conf}"
        filename = f"{folder}/jobs/{exp_name}.job"
        params = f"""--log-folder {folder}/{method}_{conf} \\
                    --recalls 1 5 10 15 20 \\
                    --descriptors-path {base_folder}/descriptors/37.75/r50_512/db_descs.pth \\
                    --queries-path {base_folder}/descriptors/37.75/r50_512/queries.pth \\
                    --positives-per-query-path {base_folder}/descriptors/positives/r50_512/ \\
                    --config {code_base_folder}/config/conf{conf}.json \\
                    --num-workers 24 \\
                    --out-file {folder}/{method}_{conf}/filtered_panoramas.json \\
                    --panoramas-path {base_folder}/panoramas/r50_512/ \\
                    --queries-zones 37.75 37.76 \\
                    --extra-descriptors {base_folder}/descriptors/37.76/r50_512"""

        content = CONTENT.replace("MEMORY",str(mem)).replace("EXP_NAME", method)\
                            .replace("PARAMS", params)
        with open(filename, "w") as file:
            _ = file.write(content)
        _ = os.system(f"sbatch {filename}")
        print(f"sbatch {filename}")