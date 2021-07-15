import os
import os.path as osp
import pickle

import zipfile
import patoolib

import pandas as pd
import numpy as np

from dataloading.imports import GESTURE_NAMES

def extract_files(path_in: str, path_extract: str):
    """Extract the uWave .zip file and each contained .rar file.

    Args:
        path_in (str): Input filepath
        path_extract (str): Output folder path
    """
    assert osp.isfile(path_in)

    with zipfile.ZipFile(path_in, 'r') as f:
        f.extractall(path_extract)

    filenames = [f for f in os.listdir(path_extract) if f.endswith('.rar')]

    for filename in filenames:
        dirname = filename.replace('.rar', "")
        print(osp.join(path_extract, filename))
        
        os.mkdir(osp.join(path_extract, dirname))
        patoolib.extract_archive(osp.join(path_extract, filename), outdir=osp.join(path_extract, dirname))

        os.remove(osp.join(path_extract, filename))

def load_files(path):
    """Load all files extracted with extract_files() into a dictionary.

    Args:
        path (str): Input folder path

    Returns:
        dict: Data dictionary with {gesture_name: [data0, data1, ...]}
    """

    all_data = {g: [] for g in GESTURE_NAMES} # Data as {gesture_name: [series0, series1, ...]}

    count = 0
    folders = [f for f in os.listdir(path) if osp.isdir(osp.join(path, f))]
    for folder in folders:
        filenames = [f for f in os.listdir(osp.join(path, folder)) if f.endswith('.txt')]
        
        for filename in filenames:
            data = pd.read_csv(osp.join(path, folder, filename), delimiter=" ", header=None)
            data = data.to_numpy()

            # Skip other files
            if sum([g in filename for g in GESTURE_NAMES]) != 1:
                continue
            
            for gesture_name in GESTURE_NAMES:
                if gesture_name in filename:
                    all_data[gesture_name].append(data)
                    count += 1
                    break
    print(f'Loaded {count} files.')

    return all_data    
    