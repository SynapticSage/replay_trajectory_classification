import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
#from sklearn.decomposition import PCA
#from sklearn.neighbors import KernelDensity
#from numba import cuda
import cupy as cp

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4
ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (cp.sqrt(5) - 1.0) / 2.0
TRANSITION_TO_CATEGORY = {
    'identity': 'stationary',
    'uniform': 'fragmented',
    'random_walk': 'continuous',
    'w_track_1D_random_walk': 'continuous',
}

PROBABILITY_THRESHOLD = 0.8
SQRT_2PI = cp.float64(cp.sqrt(2.0 * cp.pi))

# Plotting Colors
STATE_COLORS = {
    'stationary': '#9f043a',
    'fragmented': '#ff6944',
    'continuous': '#521b65',
    'stationary-continuous-mix': '#61c5e6',
    'fragmented-continuous-mix': '#2a586a',
    '': '#c7c7c7',
}


def cupy_gaussian_pdf(x, mean, sigma):
    '''Compute the value of a Gaussian probability density function at x with
    given mean and sigma.

    Parameters
    ----------
    x : float
    mean : float
    sigma : float

    Returns
    -------
    pdf : float

    '''
    return cp.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)



def cupy_kde(eval_points, samples, bandwidths):
    '''

    Parameters
    ----------
    eval_points : np.ndarray, shape (n_eval_points, n_bandwidths)
    samples : np.ndarray, shape (n_samples, n_bandwidths)
    bandwidths : np.ndarray, shape (n_bandwidths,)


    Returns
    -------
    kernel_density_estimate : np.ndarray, shape (n_eval_points, n_samples)

    '''

    eval_points = cp.array(eval_points)
    n_eval_points, n_bandwidths = eval_points.shape

    result      = cp.zeros((n_eval_points,))
    bandwidths  = cp.array(bandwidths)
    samples = cp.array(samples)

    n_samples = len(samples)

    for i in cp.arange(n_eval_points):
        for j in cp.arange(n_samples):
            product_kernel = 1.0
            for k in cp.arange(n_bandwidths):
                bandwidth  = bandwidths[k]
                eval_point = eval_points[i, k]
                sample     = samples[j, k]
                product_kernel *= (cp.exp(
                    -0.5 * ((eval_point - sample) / bandwidth)**2) /
                    (bandwidth * SQRT_2PI)) / bandwidth
            result[i] += product_kernel
        result[i] /= n_samples

    return result.get()


class GpuKDE_cupy(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y=None, sample_weight=None):
        self.training_data = X
        return self

    def score_samples(self, testing_data):
        return np.log(cupy_kde(testing_data, 
                               self.training_data,
                                self.bandwidth[-testing_data.shape[1]:]))
