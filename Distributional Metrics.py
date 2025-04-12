import numpy as np
import pandas as pd
import scipy as sp


def compute_returns(data, horizon):
  data = np.array(data)
  return (data[horizon:]- data[:-horizon])

def EMDCalculate(horizon, synthetic_returns, historic_returns):
  synthetic_returns = compute_returns(synthetic_returns, horizon)
  historic_returns = compute_returns(historic_returns, horizon)
  EMD_j = sp.stats.wasserstein_distance(synthetic_returns, historic_returns)
  return EMD_j


def compute_dy_metric(horizon, historic_returns, synthetic_returns):
    synthetic_returns = compute_returns(synthetic_returns, horizon)
    historic_returns = compute_returns(historic_returns, horizon)
    num_bins = len(historic_returns) / 5

    hist_counts, bin_edges = np.histogram(historic_returns, bins=num_bins, density=True)
    gen_counts, _ = np.histogram(synthetic_returns, bins=bin_edges, density=True)

    hist_probs = np.where(hist_counts > 0, hist_counts, 1e-10)
    gen_probs = np.where(gen_counts > 0, gen_counts, 1e-10)

    dy_value = np.sum(np.abs(np.log(hist_probs) - np.log(gen_probs))) 
    return dy_value

