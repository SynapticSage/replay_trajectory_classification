import math

import numba
import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from numba import cuda

# Figure Parameters
MM_TO_INCHES = 1.0 / 25.4
ONE_COLUMN = 89.0 * MM_TO_INCHES
ONE_AND_HALF_COLUMN = 140.0 * MM_TO_INCHES
TWO_COLUMN = 178.0 * MM_TO_INCHES
PAGE_HEIGHT = 247.0 * MM_TO_INCHES
GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2.0
TRANSITION_TO_CATEGORY = {
    'identity': 'stationary',
    'uniform': 'fragmented',
    'random_walk': 'continuous',
    'w_track_1D_random_walk': 'continuous',
}

PROBABILITY_THRESHOLD = 0.8

# Plotting Colors
STATE_COLORS = {
    'stationary': '#9f043a',
    'fragmented': '#ff6944',
    'continuous': '#521b65',
    'stationary-continuous-mix': '#61c5e6',
    'fragmented-continuous-mix': '#2a586a',
    '': '#c7c7c7',
}

SQRT_2PI = np.float64(math.sqrt(2.0 * math.pi))


class WhitenedKDE(BaseEstimator, DensityMixin):
    def __init__(self, **kwargs):
        self.kde = KernelDensity(**kwargs)
        self.pre_whiten = PCA(whiten=True)

    def fit(self, X, y=None, sample_weight=None):
        self.kde.fit(self.pre_whiten.fit_transform(X))
        return self

    def score_samples(self, X):
        return self.kde.score_samples(self.pre_whiten.transform(X))


@numba.njit(nogil=True, cache=True, parallel=True, error_model='numpy')
def numba_kde(eval_points, samples, bandwidths):
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
    n_eval_points, n_bandwidths = eval_points.shape
    result = np.zeros((n_eval_points,))
    n_samples = len(samples)

    for i in numba.prange(n_eval_points):
        for j in range(n_samples):
            product_kernel = 1.0
            for k in range(n_bandwidths):
                bandwidth = bandwidths[k]
                eval_point = eval_points[i, k]
                sample = samples[j, k]
                product_kernel *= (np.exp(
                    -0.5 * ((eval_point - sample) / bandwidth)**2) /
                    (bandwidth * SQRT_2PI)) / bandwidth
            result[i] += product_kernel
        result[i] /= n_samples

    return result


class NumbaKDE(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y=None, sample_weight=None):
        self.training_data = X
        return self

    def score_samples(self, X):
        return np.log(numba_kde(X, self.training_data,
                                self.bandwidth[-X.shape[1]:]))

## RYAN Testing cuda KDE codes

@cuda.jit(device=True)
def gaussian_pdf(x, mean, sigma):
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
    return math.exp(-0.5 * ((x - mean) / sigma)**2) / (sigma * SQRT_2PI)

@cuda.jit
def numba_kde_cuda2(eval_points, samples, bandwidths, out):
    '''

    Parameters
    ----------
    eval_points : ndarray, shape (n_eval, n_bandwidths)
    samples : ndarray, shape (n_samples, n_bandwidths)
    out : ndarray, shape (n_eval,)

    '''
    thread_id1, thread_id2 = cuda.grid(2)
    stride1, stride2 = cuda.gridsize(2)

    (n_eval, n_bandwidths), n_samples = eval_points.shape, samples.shape[0]

    for eval_ind in range(thread_id1, n_eval, stride1):
        for sample_ind in range(thread_id2, n_samples, stride2):
            product_kernel = 1.0
            for bandwidth_ind in range(n_bandwidths):
                product_kernel *= (
                    gaussian_pdf(eval_points[eval_ind, bandwidth_ind],
                                 samples[sample_ind, bandwidth_ind],
                                 bandwidths[bandwidth_ind])
                    / bandwidths[bandwidth_ind])
            product_kernel /= n_samples
            cuda.atomic.add(out, eval_ind, product_kernel)


class GpuKDE(BaseEstimator, DensityMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y=None, sample_weight=None):
        self.training_data = X
        return self

    def score_samples(self, testing_data):

        threads_per_block = 8, 8
        n_test, n_train = testing_data.shape[0], self.training_data.shape[0]
        blocks_per_grid_x = np.min((
            math.ceil(n_test / threads_per_block[0]), 65535))
        blocks_per_grid_y = np.min((
            math.ceil(n_train / threads_per_block[1]), 65535))
        blocks_per_grid = blocks_per_grid_x, blocks_per_grid_y

        results = np.zeros((testing_data.shape[0],), dtype='float32')
        numba_kde_cuda2[blocks_per_grid, threads_per_block]( testing_data, 
            self.training_data, self.bandwidth[-testing_data.shape[1]:], results)
        results = np.log(results)

        return results
