import numpy as np


def rough_one(item_space):
    x_mean = np.mean(item_space)
    abs_matr = np.abs(item_space - x_mean)
    return np.max(abs_matr)
