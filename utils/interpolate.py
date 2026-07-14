import numpy as np


def interpolate_stress(x, x0, y0):
    ndim = y0.shape[1]
    y = np.empty((len(x), ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            y[:,i,j] = np.interp(x, x0, y0[:,i,j])
    return y