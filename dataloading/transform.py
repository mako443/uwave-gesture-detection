import os
import os.path as osp

import pickle
import numpy as np

from dataloading.imports import GESTURE_NAMES
from dataloading.visualize import visualize_samples

def normalize_vectors(data):
    data_normalized = {gesture_name: [] for gesture_name in data}
    for gesture_name in data:
        for i in range(len(data[gesture_name])):
            normalized = data[gesture_name][i] / np.linalg.norm(data[gesture_name][i], axis=-1).reshape((-1, 1))
            normalized[np.isnan(normalized)] = 0 # Some entries might be zero, resulting in nan during normalization. Therefore, setting to 0 again is okay here.
            data_normalized[gesture_name].append(normalized)
    return data_normalized

def integrate_acceleration(data):
    data_integrated = {gesture_name: [] for gesture_name in data}
    for gesture_name in data:
        for i in range(len(data[gesture_name])):
            integrated = np.cumsum(data[gesture_name][i], axis=0)
            data_integrated[gesture_name].append(integrated)
    return data_integrated

def center_scale(data, scale_per_axis=False):
    data_scaled = {gesture_name: [] for gesture_name in data}
    for gesture_name in data:
        for i in range(len(data[gesture_name])):
            centered_scaled = data[gesture_name][i]
            centered_scaled -= np.mean(centered_scaled, axis=0)
            if scale_per_axis:
                centered_scaled /= np.abs(np.max(centered_scaled, axis=0))
            else:
                centered_scaled /= np.abs(np.max(centered_scaled))
            centered_scaled[np.isnan(centered_scaled)] = 0 # Some entries might be zero, resulting in nan during centering. Therefore, setting to 0 again is okay here.
            data_scaled[gesture_name].append(centered_scaled)
    return data_scaled

if __name__ == '__main__':
    with open('./data/uwave/uwave.pkl', 'rb') as f:
        data = pickle.load(f)

    # name, idx = 'hook', 0
    # name, idx = 'box', 1
    # name, idx = 'left2right', 2
    # name, idx = 'top2bot', 5
    name, idx = 'counter-clw', 7

    num_visualize = 9
    gestures = [GESTURE_NAMES[idx] for i in range(num_visualize)]
    indices = np.random.randint(500, size=num_visualize)

    data_normalized = normalize_vectors(data)
    print(data_normalized[gestures[0]][0].dtype)
    quit()

    # data_normalized_integrated = integrate_acceleration(data_normalized)
    # data_integrated = integrate_acceleration(data)
    # data_normalized_integrated_scaled = center_scale(data_normalized_integrated)
    # data_integrated_scaled = center_scale(data_integrated)

    visualize_samples([data_normalized_integrated_scaled[gestures[i]][indices[i]] for i in range(num_visualize)], axes=(0, 1), save_path=f"plots/{name}_norm_integ_scale_01.png") 
    visualize_samples([data_integrated_scaled[gestures[i]][indices[i]] for i in range(num_visualize)], axes=(0, 1), save_path=f"plots/{name}_integ_scale_01.png") 

    # Assumption: axes 0-1 are the important ones, y-axis flipped for plotting, actual gesture is at the end of time-series