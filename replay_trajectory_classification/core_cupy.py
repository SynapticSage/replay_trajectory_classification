import numpy as np
import cupy as cp
from numba import njit
from tqdm import tqdm


def logsumexp(a):
    a_max = cp.max(a)
    out = cp.log(cp.sum(cp.exp(a - a_max)))
    out += a_max
    return out


def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / cp.nansum(distribution)


def _causal_decode(initial_conditions, state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_bins,)
    state_transition : ndarray, shape (n_bins, n_bins)
    likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    posterior : ndarray, shape (n_time, n_bins)

    '''
    from tqdm import tqdm
 
    n_time = likelihood.shape[0]
    posterior          = cp.zeros_like(likelihood)
    state_transition   = cp.array(state_transition)
    initial_conditions = cp.array(initial_conditions)
    likelihood         = cp.array(likelihood)

    posterior[0] = normalize_to_probability(
        initial_conditions * likelihood[0])

    del initial_conditions

    for time_ind in tqdm(cp.arange(1, n_time), desc="Causal filtering"):
        prior = state_transition.T @ posterior[time_ind - 1]
        posterior[time_ind] = normalize_to_probability(
            prior * likelihood[time_ind])

    return posterior.get()


def _acausal_decode(causal_posterior, state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_bins, 1)
    state_transition : ndarray, shape (n_bins, n_bins)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_bins, 1)

    '''
    causal_posterior      = cp.array(causal_posterior)
    acausal_posterior     = cp.zeros_like(causal_posterior)
    acausal_posterior[-1] = cp.array(causal_posterior[-1])
    state_transition      = cp.array(state_transition)
    n_time, n_bins = causal_posterior.shape[0], causal_posterior.shape[-2]
    weights = cp.zeros((n_bins, 1))
    eta     = cp.array(np.spacing(1))

    for time_ind in tqdm(cp.arange(n_time - 2, -1, -1), desc='Causal filtering'):
        acausal_prior = (
            state_transition.T @ causal_posterior[time_ind])
        log_ratio = (
            cp.log(acausal_posterior[time_ind + 1, ..., 0] + eta) -
            cp.log(acausal_prior[..., 0] + eta))
        weights[..., 0] = cp.exp(log_ratio) @ state_transition

        acausal_posterior[time_ind] = normalize_to_probability(
            weights * causal_posterior[time_ind])

    return acausal_posterior.get()


#@njit(nogil=True, error_model='numpy', cache=True)
def _causal_classify(initial_conditions, continuous_state_transition,
                     discrete_state_transition, likelihood):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)
    likelihood : ndarray, shape (n_time, n_states, n_bins, 1)

    Returns
    -------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    n_time, n_states, n_bins, _ = likelihood.shape
    posterior          = cp.zeros_like(likelihood)
    initial_conditions = cp.array(initial_conditions)

    posterior[0] = normalize_to_probability(
        initial_conditions * likelihood[0])

    del initial_conditions

    for k in cp.arange(1, n_time):
        prior = cp.zeros((n_states, n_bins, 1))
        for state_k in cp.arange(n_states):
            for state_k_1 in cp.arange(n_states):
                prior[state_k, :] += (
                    discrete_state_transition[state_k_1, state_k] *
                    continuous_state_transition[state_k_1, state_k].T @
                    posterior[k - 1, state_k_1])
        posterior[k] = normalize_to_probability(prior * likelihood[k])

    return posterior


#@njit(nogil=True, error_model='numpy', cache=True)
def _acausal_classify(causal_posterior, continuous_state_transition,
                      discrete_state_transition):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    causal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)
    continuous_state_transition : ndarray, shape (n_states, n_states,
                                                  n_bins, n_bins)
    discrete_state_transition : ndarray, shape (n_states, n_states)

    Return
    ------
    acausal_posterior : ndarray, shape (n_time, n_states, n_bins, 1)

    '''
    acausal_posterior = cp.zeros_like(causal_posterior)
    acausal_posterior[-1] = cp.array(causal_posterior[-1])
    n_time, n_states, n_bins, _ = causal_posterior.shape
    eta = cp.array(np.spacing(1))

    for k in cp.arange(n_time - 2, -1, -1):
        # Prediction Step -- p(x_{k+1}, I_{k+1} | y_{1:k})
        prior = cp.zeros((n_states, n_bins, 1))
        for state_k_1 in cp.arange(n_states):
            for state_k in cp.arange(n_states):
                prior[state_k_1, :] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1].T @
                    causal_posterior[k, state_k])

        # Backwards Update
        weights = cp.zeros((n_states, n_bins, 1))
        ratio = cp.exp(
            cp.log(acausal_posterior[k + 1] + eta) -
            cp.log(prior + eta))
        for state_k in cp.arange(n_states):
            for state_k_1 in np.arange(n_states):
                weights[state_k] += (
                    discrete_state_transition[state_k, state_k_1] *
                    continuous_state_transition[state_k, state_k_1] @
                    ratio[state_k_1])

        acausal_posterior[k] = normalize_to_probability(
            weights * causal_posterior[k])

    return acausal_posterior.get()


def scaled_likelihood(log_likelihood, axis=1):
    '''
    Parameters
    ----------
    log_likelihood : ndarray, shape (n_time, n_bins)

    Returns
    -------
    scaled_log_likelihood : ndarray, shape (n_time, n_bins)

    '''
    max_log_likelihood = np.nanmax(log_likelihood, axis=axis, keepdims=True)
    # np.exp(posterior - logsumexp(posterior, axis=axis)) ?
    likelihood = np.exp(log_likelihood - max_log_likelihood)
    likelihood[np.isnan(likelihood)] = 0.0
    return likelihood


def mask(value, is_track_interior):
    try:
        value[..., ~is_track_interior] = np.nan
    except IndexError:
        value[..., ~is_track_interior, :] = np.nan
    return value
