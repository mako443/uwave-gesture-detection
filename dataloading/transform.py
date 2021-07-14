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

def polyfit_timeseries(data, deg=3):
    data_poly = {gesture_name: [] for gesture_name in data}
    for gesture_name in data:
        for i_sample, sample in enumerate(data[gesture_name]):
            poly = np.polynomial.polynomial.polyfit(np.arange(len(sample)), sample, deg=deg)
            data_poly[gesture_name].append(poly.T.flatten()) # [x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3]

    return data_poly

def aggregate_data(data):
    num_samples = np.sum([len(data[g]) for g in GESTURE_NAMES])
    dim = len(data[GESTURE_NAMES[0]][0])
    X = np.zeros((num_samples, dim), dtype=np.float)
    y = np.zeros(num_samples, dtype=np.int)
    
    idx = 0
    for i_class, gesture_name in enumerate(GESTURE_NAMES):
        for sample in data[gesture_name]:
            X[idx, :] = sample
            y[idx] = i_class
            idx += 1

    return X, y

def normalize_X(X, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X)
        
    X_normed = (X - mean) / std
    return X_normed, mean, std

def random_split(X, y, train_fraction=0.7):
    assert len(X) == len(y)
    test_fraction = 1.0 - train_fraction
    indices_test = []
    
    # Sample the same fraction from each class to prevent (further) imbalance
    for i_class in np.unique(y):
        possible_indices = np.argwhere(y == i_class).flatten()
        selected_indices = np.random.choice(possible_indices, size=int(test_fraction*len(possible_indices)), replace=False) # Select test indices to require less random samples
        indices_test.extend(list(selected_indices))

    # Get the "inverse" indices for training
    slice_train = np.ones(len(X), dtype=np.bool)
    slice_train[indices_test] = False
    
    return X[slice_train], y[slice_train], X[indices_test], y[indices_test]    

if __name__ == '__main__':
    pass