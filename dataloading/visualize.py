import numpy as np
import matplotlib.pyplot as plt

def visualize_samples(samples, axes=(0, 1), save_path=None, invert_y=True, figsize=(6.4, 4.8)):
    """Visualize multiple samples in a combined figure

    Args:
        samples (list): List of samples
        axes (tuple, optional): Which axes to plot. Defaults to (0, 1).
        save_path (str, optional): If set, the figure is saved to this path, otherwise it is shown. Defaults to None.
        invert_y (bool, optional): Invert the y-axis to be coherent with the gesture sketches. Defaults to True.
        figsize (tuple, optional): Figure size. Defaults to (6.4, 4.8).
    """

    fig = plt.figure(figsize=figsize)
    num_x = int(np.sqrt(len(samples)))
    num_y = int(np.ceil(len(samples) / num_x))
    # num_x = 1
    # num_y = len(samples)
    for i_sample, sample in enumerate(samples):
        plt.subplot(num_x, num_y, i_sample + 1, aspect='equal')
        # plt.plot(sample[axes[0]], sample[axes[1]], c=np.arange(len(sample)), cmap=plt.cm.coolwarm)
        if invert_y:
            plt.scatter(sample[:, axes[0]], -1.0 * sample[:, axes[1]], marker='.', c=np.arange(len(sample)), cmap=plt.cm.coolwarm)
        else:
            plt.scatter(sample[:, axes[0]], sample[:, axes[1]], marker='.', c=np.arange(len(sample)), cmap=plt.cm.coolwarm)
    
    fig

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')