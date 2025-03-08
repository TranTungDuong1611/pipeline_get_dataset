import glob
import gzip
import os
import shutil
import subprocess
import sys

import gdown
import pandas as pd
import torch
from PIL import Image

gan_control_repo = 'https://github.com/amazon-science/gan-control.git'
gan_control_dir = 'gan-control'

# clone gan-control repo
if not os.path.exists(gan_control_dir):
    subprocess.run(["git", "clone", gan_control_repo], check=True)
else:
    print(f"Repository '{gan_control_dir}' is already exist!")

# down load weights
if glob.glob(os.path.join(os.getcwd(), '*.part*')) is None:
    url = 'https://drive.google.com/file/d/19v0lX69fV6zQv2HbbYUVr9gZ8ZKvUzHq/view'
    gdown.download(url, os.getcwd(), fuzzy=True, quiet=False)

    # unzip the weight
    # path to the devided file
    file_parts = glob.glob(os.path.join(os.getcwd(), '*.part*'))

    # path to the output file
    output_file = os.path.join(os.getcwd(), 'merged.tar.gz')

    # merge file
    with open(output_file, 'wb') as outfile:
        for part in file_parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())

    print(f"Merged and save into {output_file}")

    # path to file .gz and output file
    input_file = "merged.tar.gz"
    output_file = "merged.tar"

    # unzip file .gz
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Unziped {input_file} into {output_file}")

    # path to tar file
    tar_file = "merged.tar"
    save_folder = os.path.join(os.getcwd(), 'gan-control/resources', 'gan_models')

    os.makedirs(save_folder, exist_ok=True)

    # run tar
    subprocess.run(["tar", "-xvf", tar_file, "-C", save_folder], check=True)

# change the folder weight name
extract_name = './gan-control/resources/gan_models/controller_age015id025exp02hai04ori02gam15_temp'
changed_name = './gan-control/resources/gan_models/controller_age015id025exp02hai04ori02gam15'

if os.path.exists(extract_name):
    os.rename(extract_name, changed_name)

print('The weight has been setted up')