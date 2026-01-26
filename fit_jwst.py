import os
import sys
import glob
import csv
import pickle
from functools import partial

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.stats import norm
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.7
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from plotting import plot_map_fits, plot_map_residuals, plot_transmission_spectrum, plot_wavelength_offset_summary
import argparse
import yaml
import jaxopt
import arviz as az
from createdatacube import SpectroData, process_spectroscopy_data
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.gridspec as gridspec
from jaxoplanet.experimental import calc_poly_coeffs
import tinygp
import matplotlib.cm as cm
from models.core import _to_f64, _tree_to_f64, compute_transit_model, get_I_power2
from models.trends import (
    spot_crossing, compute_lc_linear, compute_lc_quadratic, compute_lc_cubic,
    compute_lc_quartic, compute_lc_linear_discontinuity, compute_lc_explinear,
    compute_lc_spot, compute_lc_2spot, compute_lc_none
)
from models.gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_cubic_gp_mean, compute_lc_quartic_gp_mean, compute_lc_explinear_gp_mean
)
from models.builder import create_whitelight_model, create_vectorized_model, COMPUTE_KERNELS

TREND_PARAMS = [
    'c', 'v', 'v2', 'v3', 'v4', 
    'A', 'tau', 
    'spot_amp', 'spot_mu', 'spot_sigma', 
    'spot_amp2', 'spot_mu2', 'spot_sigma2',
    't_jump', 'jump', 
    'A_gp', 'A_spot', 'A_spot2', 'A_jump'
]

def _param_at(params, name, idx=None):
    if name not in params:
        return None
    val = params[name]
    return val[idx] if idx is not None else val

def _poly_trend_np(params, t_shift, order, idx=None):
    trend = _param_at(params, "c", idx) + _param_at(params, "v", idx) * t_shift
    if order >= 2:
        trend = trend + _param_at(params, "v2", idx) * t_shift**2
    if order >= 3:
        trend = trend + _param_at(params, "v3", idx) * t_shift**3
    if order >= 4:
        trend = trend + _param_at(params, "v4", idx) * t_shift**4
    return trend

_JUMP_WIDTH_DAYS = 1e-4

def _soft_step_np(t, t_jump, width=_JUMP_WIDTH_DAYS):
    return 0.5 * (1.0 + np.tanh((t - t_jump) / width))

def _trend_from_params_np(detrend_type, time, params, idx=None, gp_trend=None, spot_trend=None, spot_trend2=None, jump_trend=None):
    t_shift = time - np.min(time)
    if detrend_type == 'none':
        return np.ones_like(time)

    poly_order = 0
    if 'quartic' in detrend_type:
        poly_order = 4
    elif 'cubic' in detrend_type:
        poly_order = 3
    elif 'quadratic' in detrend_type:
        poly_order = 2
    elif 'linear' in detrend_type or detrend_type in {'spot', '2spot'}:
        poly_order = 1

    if 'gp_spectroscopic' in detrend_type:
        trend = _param_at(params, "c", idx) if poly_order == 0 else _poly_trend_np(params, t_shift, poly_order, idx)
        if 'explinear' in detrend_type:
            trend = trend + _param_at(params, "A", idx) * np.exp(-t_shift / _param_at(params, "tau", idx))
        return trend + _param_at(params, "A_gp", idx) * gp_trend

    if detrend_type == '2spot_spectroscopic':
        return (_param_at(params, "c", idx)
                + _param_at(params, "A_spot", idx) * spot_trend
                + _param_at(params, "A_spot2", idx) * spot_trend2)
    if detrend_type == 'spot_spectroscopic':
        return _param_at(params, "c", idx) + _param_at(params, "A_spot", idx) * spot_trend
    if detrend_type == 'linear_discontinuity_spectroscopic':
        return _param_at(params, "c", idx) + _param_at(params, "A_jump", idx) * jump_trend

    trend = _param_at(params, "c", idx) if poly_order == 0 else _poly_trend_np(params, t_shift, poly_order, idx)
    if detrend_type == 'linear_discontinuity':
        if jump_trend is None:
            jump_trend = _param_at(params, "jump", idx) * _soft_step_np(time, _param_at(params, "t_jump", idx))
        trend = trend + jump_trend
    elif detrend_type == 'explinear':
        trend = trend + _param_at(params, "A", idx) * np.exp(-t_shift / _param_at(params, "tau", idx))
    elif detrend_type == 'spot':
        if spot_trend is None:
            spot_trend = spot_crossing(time, _param_at(params, "spot_amp", idx), _param_at(params, "spot_mu", idx), _param_at(params, "spot_sigma", idx))
        trend = trend + spot_trend
    elif detrend_type == '2spot':
        if spot_trend is None:
            spot_trend = spot_crossing(time, _param_at(params, "spot_amp", idx), _param_at(params, "spot_mu", idx), _param_at(params, "spot_sigma", idx))
            spot_trend = spot_trend + spot_crossing(time, _param_at(params, "spot_amp2", idx), _param_at(params, "spot_mu2", idx), _param_at(params, "spot_sigma2", idx))
        trend = trend + spot_trend
    return trend

def _align_trend_to_time(trend, trend_time, target_time):
    trend = np.asarray(trend)
    target_time = np.asarray(target_time)
    trend_time = np.asarray(trend_time)
    if len(trend) == len(target_time):
        return trend
    return np.interp(target_time, trend_time, trend)

def _spectro_detrend_type(detrending_type):
    """Map WL detrending type to spectroscopic equivalent where applicable."""
    detrend_type = detrending_type
    if 'gp' in detrend_type and 'gp_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('gp', 'gp_spectroscopic')
    if 'linear_discontinuity' in detrend_type and 'linear_discontinuity_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('linear_discontinuity', 'linear_discontinuity_spectroscopic')
    if 'spot' in detrend_type and 'spot_spectroscopic' not in detrend_type:
        detrend_type = detrend_type.replace('spot', 'spot_spectroscopic')
    return detrend_type

def _bin_time_series_numpy(time, y, yerr=None, dt_seconds=120.0, t0=None, method="weighted"):
    """
    Bin a time-series (or stack of time-series) onto a fixed cadence.

    Parameters
    ----------
    time : (N,) array
        Time array (assumed to be in DAYS if it's MJD/BJD-style; we convert dt_seconds->dt_days).
    y : (N,) or (C, N) array
        Flux (or residuals, etc.). If 2D, bins along the last axis.
    yerr : (N,) or (C, N) array or None
        Uncertainties. If None, uses unweighted mean and returns None for binned errors.
    dt_seconds : float
        Bin width in seconds.
    t0 : float or None
        Bin reference start time. If None, uses min(time).
    method : {"weighted","mean"}
        "weighted": inverse-variance weighted mean (needs yerr)
        "mean": simple mean; error propagated as sqrt(sum(err^2))/N if yerr provided

    Returns
    -------
    t_b : (M,) array
    y_b : (M,) or (C, M) array
    yerr_b : same shape as y_b or None
    counts : (M,) array
    """
    time = np.asarray(time)
    y = np.asarray(y)

    if t0 is None:
        t0 = np.nanmin(time)

    dt_days = float(dt_seconds) / 86400.0
    if dt_days <= 0:
        raise ValueError("dt_seconds must be > 0")

    idx = np.floor((time - t0) / dt_days).astype(np.int64)

    valid = np.isfinite(time)
    if y.ndim == 1:
        valid &= np.isfinite(y)
    else:
        valid &= np.all(np.isfinite(y), axis=0)

    if yerr is not None:
        yerr = np.asarray(yerr)

        if y.ndim == 1 and yerr.ndim == 2:
            if yerr.shape == (1, y.shape[0]):
                yerr = yerr[0]
            elif yerr.shape == (y.shape[0], 1):
                yerr = yerr[:, 0]

        if yerr.ndim == 0:
            yerr = np.full_like(y, float(yerr))

        elif y.ndim == 2 and yerr.ndim == 1:
            if yerr.shape[0] == y.shape[1]:
                yerr = np.broadcast_to(yerr[None, :], y.shape)
            elif yerr.shape[0] == y.shape[0]:
                yerr = np.broadcast_to(yerr[:, None], y.shape)

        if yerr.ndim == 1:
            valid &= np.isfinite(yerr) & (yerr > 0)
        else:
            valid &= np.all(np.isfinite(yerr) & (yerr > 0), axis=0)


    time_v = time[valid]
    idx_v = idx[valid]

    idx0 = idx_v.min()
    idx_v = idx_v - idx0
    nbins = int(idx_v.max()) + 1

    counts = np.bincount(idx_v, minlength=nbins).astype(float)
    keep = counts > 0

    t_sum = np.bincount(idx_v, weights=time_v, minlength=nbins)
    t_b_full = np.full(nbins, np.nan)
    t_b_full[keep] = t_sum[keep] / counts[keep]
    t_b = t_b_full[keep]
    counts_b = counts[keep]

    def _bin_1d(y1, e1):
        if method == "weighted" and e1 is not None:
            w = 1.0 / (e1 * e1)
            wsum = np.bincount(idx_v, weights=w, minlength=nbins)
            ysum = np.bincount(idx_v, weights=w * y1, minlength=nbins)
            yb_full = np.full(nbins, np.nan)
            eb_full = np.full(nbins, np.nan)
            yb_full[keep] = ysum[keep] / wsum[keep]
            eb_full[keep] = np.sqrt(1.0 / wsum[keep])
            yb = yb_full[keep]
            eb = eb_full[keep]
            return yb, eb
        else:
            ysum = np.bincount(idx_v, weights=y1, minlength=nbins)
            yb_full = np.full(nbins, np.nan)
            yb_full[keep] = ysum[keep] / counts[keep]
            yb = yb_full[keep]
            if e1 is None:
                return yb, None
            esum2 = np.bincount(idx_v, weights=e1 * e1, minlength=nbins)
            eb = np.sqrt(esum2[keep]) / counts_b
            return yb, eb

    if y.ndim == 1:
        y_v = y[valid]
        e_v = None if yerr is None else (yerr[valid] if yerr.ndim == 1 else yerr[:, valid])
        y_b, e_b = _bin_1d(y_v, e_v if (e_v is not None and np.ndim(e_v) == 1) else None)
        return t_b, y_b, e_b, counts_b

    C = y.shape[0]
    y_b_list = []
    e_b_list = [] if yerr is not None else None

    for c in range(C):
        y_v = y[c, valid]
        e_v = None if yerr is None else (yerr[c, valid] if yerr.ndim == 2 else None)
        yb, eb = _bin_1d(y_v, e_v)
        y_b_list.append(yb)
        if e_b_list is not None:
            e_b_list.append(eb)

    y_b = np.stack(y_b_list, axis=0)
    e_b = None if e_b_list is None else np.stack(e_b_list, axis=0)
    return t_b, y_b, e_b, counts_b


def bin_spectrodata_in_time(data, dt_seconds=120.0, method="weighted", bin_whitelight=True, bin_spectroscopic=True):
    """
    In-place time-binning for SpectroData:
      - wl_time, wl_flux, wl_flux_err
      - time, flux_lr, flux_err_lr
      - time, flux_hr, flux_err_hr
    """
    n0_wl = len(data.wl_time)
    n0_spec = len(data.time)

    def _timebin_debug(label, time, wl_time, flux_lr, flux_hr, wl_flux):
        def _summarize_time(t):
            if t is None or len(t) == 0:
                return "n=0"
            dt_med = np.nanmedian(np.diff(t)) if len(t) > 1 else np.nan
            return f"n={len(t)}, range=[{np.nanmin(t):.6f}, {np.nanmax(t):.6f}], dt_med={dt_med:.6e}"

        def _shape_or_none(arr):
            return None if arr is None else arr.shape

        def _nan_count(arr):
            return None if arr is None else int(np.isnan(arr).sum())

        print(f"=== TIME BINNING DEBUG ({label}) ===")
        print(f"wl_time: {_summarize_time(wl_time)}")
        print(f"time: {_summarize_time(time)}")
        print(f"wl_flux shape: {_shape_or_none(wl_flux)} nan_count: {_nan_count(wl_flux)}")
        print(f"flux_lr shape: {_shape_or_none(flux_lr)} nan_count: {_nan_count(flux_lr)}")
        print(f"flux_hr shape: {_shape_or_none(flux_hr)} nan_count: {_nan_count(flux_hr)}")
        if flux_lr is not None:
            print(f"n_channels_lr: {flux_lr.shape[0]}")
        if flux_hr is not None:
            print(f"n_channels_hr: {flux_hr.shape[0]}")

    _timebin_debug("pre", data.time, data.wl_time, data.flux_lr, data.flux_hr, data.wl_flux)

    t_b_wl = data.wl_time
    if bin_whitelight:
        t_b_wl, wl_b, wlerr_b, _ = _bin_time_series_numpy(
            data.wl_time, data.wl_flux, data.wl_flux_err,
            dt_seconds=dt_seconds, method=method
        )
        data.wl_time = t_b_wl
        data.wl_flux = wl_b
        data.wl_flux_err = wlerr_b

    if bin_spectroscopic:
        t0_ref = np.nanmin(data.wl_time)
        t_b_lr, flr_b, flrerr_b, _ = _bin_time_series_numpy(
            data.time, data.flux_lr, data.flux_err_lr,
            dt_seconds=dt_seconds, method=method, t0=t0_ref
        )
        t_b_hr, fhr_b, fhrerr_b, _ = _bin_time_series_numpy(
            data.time, data.flux_hr, data.flux_err_hr,
            dt_seconds=dt_seconds, method=method, t0=t0_ref
        )
        data.time = t_b_lr
        data.flux_lr = flr_b
        data.flux_err_lr = flrerr_b
        data.flux_hr = fhr_b
        data.flux_err_hr = fhrerr_b

    if bin_whitelight:
        print(f"[time_binning wl] {n0_wl} -> {len(data.wl_time)} points (dt={dt_seconds:.1f}s, method={method})")
    if bin_spectroscopic:
        print(f"[time_binning spec] {n0_spec} -> {len(data.time)} points (dt={dt_seconds:.1f}s, method={method})")
    _timebin_debug("post", data.time, data.wl_time, data.flux_lr, data.flux_hr, data.wl_flux)
    return data

DTYPE = jnp.float64

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def jax_bin_lightcurve(time, flux, duration, points_per_transit=20):
    dt = duration / points_per_transit
    t_min = jnp.min(time)
    t_max = jnp.max(time)
    total_range = t_max - t_min
    num_bins = jnp.ceil(total_range / dt).astype(int) + 1
    bin_indices = jnp.clip(((time - t_min) / dt).astype(int), 0, num_bins - 1)
    flux_sums = jnp.zeros(num_bins)
    time_sums = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)
    flux_sums = flux_sums.at[bin_indices].add(flux)
    time_sums = time_sums.at[bin_indices].add(time)
    counts = counts.at[bin_indices].add(1.0)
    binned_flux = jnp.where(counts > 0, flux_sums / counts, jnp.nan)
    binned_time = jnp.where(counts > 0, time_sums / counts, jnp.nan)
    return binned_time, binned_flux
    
def _noise_binning_stats(residuals, n_bins=30, max_bin=None):
    """
    residuals: array-like, shape (n_channels, n_times) or (n_times,)
    Returns: bins, median_sigma, p16, p84, expected_white_median, expected_white_per_channel
    """
    residuals = np.array(residuals)
    if residuals.ndim == 1:
        residuals = residuals[None, :]
    n_channels, n_times = residuals.shape
    if max_bin is None:
        max_bin = max(1, n_times // 4)
    bins = np.unique(np.round(np.logspace(0, np.log10(max_bin), n_bins)).astype(int))
    sigma_b_channels = np.full((n_channels, len(bins)), np.nan)
    for i in range(n_channels):
        r = residuals[i, :]
        for j, b in enumerate(bins):
            m = (n_times // b) * b
            if m < b:
                sigma_b_channels[i, j] = np.nan
                continue
            r_trunc = r[:m].reshape(-1, b)
            means = r_trunc.mean(axis=1)
            if means.size >= 2:
                sigma_b_channels[i, j] = np.std(means, ddof=0)
            else:
                sigma_b_channels[i, j] = np.nan

    sigma_med = np.nanmedian(sigma_b_channels, axis=0)
    sigma_16 = np.nanpercentile(sigma_b_channels, 16, axis=0)
    sigma_84 = np.nanpercentile(sigma_b_channels, 84, axis=0)
    sigma1_channels = np.nanstd(residuals, axis=1, ddof=0)
    sigma1_med = np.nanmedian(sigma1_channels)
    expected_white_per_channel = sigma1_channels[:, None] / np.sqrt(bins)
    expected_white_median = sigma1_med / np.sqrt(bins)
    return bins, sigma_med, sigma_16, sigma_84, expected_white_median, expected_white_per_channel

def plot_noise_binning(residuals, outpath, title=None, to_ppm=True, show_per_channel_expected=False):
    bins, sigma_med, sigma_16, sigma_84, expected_white_med, expected_white_pc = _noise_binning_stats(residuals)
    factor = 1e6 if to_ppm else 1.0
    plt.figure(figsize=(6,4))
    plt.loglog(bins, sigma_med * factor, 'k-o', label='Measured (median across channels)')
    plt.fill_between(bins, sigma_16 * factor, sigma_84 * factor, color='gray', alpha=0.3, label='16-84%')
    plt.loglog(bins, expected_white_med * factor, 'r--', label='White-noise expectation (σ₁/√N)')
    if show_per_channel_expected and expected_white_pc is not None:
        for ch in range(expected_white_pc.shape[0]):
            plt.loglog(bins, expected_white_pc[ch, :] * factor, color='r', alpha=0.12)
    plt.xlabel('Bin size (number of points)')
    ylabel = 'RMS (ppm)' if to_ppm else 'RMS'
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"Saved noise-binning plot to {outpath}")

def get_samples(model, key, t, yerr, indiv_y, init_params, **model_kwargs):
    t = _to_f64(t)
    yerr = _to_f64(yerr)
    indiv_y = _to_f64(indiv_y)
    init_params = _tree_to_f64(init_params)
    model_kwargs = _tree_to_f64(model_kwargs)

    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            regularize_mass_matrix=False,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
            target_accept_prob=0.9
        ),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=True,
        jit_model_args=True
    )
    mcmc.run(key, t, yerr, y=indiv_y, **model_kwargs)
    return mcmc.get_samples()

def _slice_by_channel(value, sl, num_lcs):
    if value is None:
        return None
    if isinstance(value, (np.ndarray, jnp.ndarray)) and value.ndim > 0 and value.shape[0] == num_lcs:
        return value[sl]
    return value

def get_samples_chunked(model, key, t, yerr, indiv_y, init_params, chunk_size, output_dir=None, checkpoint_prefix=None, **model_kwargs):
    """
    Run MCMC in chunks with checkpoint support for wall-time resilience.

    Parameters:
    -----------
    output_dir : str, optional
        Directory to save checkpoint files. If None, no checkpointing is used.
    checkpoint_prefix : str, optional
        Prefix for checkpoint filenames (e.g., 'WASP-39_PRISM_native').
        Checkpoints saved as: {output_dir}/chunks/{checkpoint_prefix}_chunk_{start}_{end}.pkl
    """
    num_lcs = indiv_y.shape[0]
    print(f"Running chunked MCMC: {num_lcs} channels in blocks of {chunk_size}")

    # Set up checkpointing
    use_checkpoints = (output_dir is not None and checkpoint_prefix is not None)
    if use_checkpoints:
        checkpoint_dir = os.path.join(output_dir, 'chunks')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Checkpoint directory: {checkpoint_dir}")

    samples_chunks = []
    chunk_ranges = []  # Track (start, end) for each chunk

    for start in range(0, num_lcs, chunk_size):
        end = min(start + chunk_size, num_lcs)
        chunk_ranges.append((start, end))
        sl = slice(start, end)

        # Check if this chunk already exists
        if use_checkpoints:
            chunk_file = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_chunk_{start}_{end}.pkl")
            if os.path.exists(chunk_file):
                print(f"  chunk {start}:{end} - LOADING from checkpoint")
                with open(chunk_file, 'rb') as f:
                    samples_chunk = pickle.load(f)
                samples_chunks.append(samples_chunk)
                continue  # Skip computation for this chunk

        # Compute this chunk
        key, key_chunk = jax.random.split(key)
        init_chunk = {k: _slice_by_channel(v, sl, num_lcs) for k, v in init_params.items()}
        kwargs_chunk = {k: _slice_by_channel(v, sl, num_lcs) for k, v in model_kwargs.items()}
        yerr_chunk = yerr[sl]
        y_chunk = indiv_y[sl]
        print(f"  chunk {start}:{end} - COMPUTING ({end - start} channels)")
        samples_chunk = get_samples(model, key_chunk, t, yerr_chunk, y_chunk, init_chunk, **kwargs_chunk)
        samples_chunk = jax.device_get(samples_chunk)
        samples_chunks.append(samples_chunk)

        # Save checkpoint immediately after computation
        if use_checkpoints:
            chunk_file = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_chunk_{start}_{end}.pkl")
            with open(chunk_file, 'wb') as f:
                pickle.dump(samples_chunk, f)
            print(f"  chunk {start}:{end} - SAVED checkpoint")

    # Concatenate all chunks
    print(f"\nConcatenating {len(samples_chunks)} chunks...")
    samples = {}
    for key in samples_chunks[0].keys():
        arrays = [chunk[key] for chunk in samples_chunks]
        samples[key] = np.concatenate(arrays, axis=1)

    print(f"All chunks complete! Final shape: {list(samples.values())[0].shape}")
    return samples

def compute_aic(n, residuals, k):
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10
    aic = 2*k + n * np.log(rss/n)
    return aic

def get_asym_errors(data, axis=0):
    p16, p50, p84 = jnp.nanpercentile(data, jnp.array([16, 50, 84]), axis=axis)
    err_low = p50 - p16
    err_high = p84 - p50
    return p50, err_low, err_high

def get_limb_darkening(sld, wavelengths, wavelength_err, instrument, order=None, ld_profile='quadratic'):
    if instrument == 'NIRSPEC/G395H':
        mode = "JWST_NIRSpec_G395H"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/G235H':
        mode = "JWST_NIRSpec_G235H"
        wl_min, wl_max = 17000.0, 30600.0
    elif instrument == 'NIRSPEC/G140H':
        mode = "JWST_NIRSpec_G140H-f100"
        wl_min, wl_max = 10000.0, 18000.0
    elif instrument == 'NIRSPEC/G395M':
        mode = "JWST_NIRSpec_G395M"
        wl_min, wl_max = 28700.0, 51700.0
    elif instrument == 'NIRSPEC/PRISM':
        mode = "JWST_NIRSpec_Prism"
        wl_min, wl_max = 5000.0, 55000.0
    elif instrument == 'NIRISS/SOSS':
        mode = f"JWST_NIRISS_SOSSo{order}"
        wl_min, wl_max = 8300.0, 28100.0
    elif instrument == 'MIRI/LRS':
        mode = f'JWST_MIRI_LRS'
        wl_min, wl_max = 50000.0, 120000.0

    wavelengths = np.array(wavelengths)
    wavelength_err = np.array(wavelength_err) if hasattr(wavelength_err, '__len__') else wavelength_err

    U_mu = []

    if hasattr(wavelength_err, '__len__'):
        for i in range(len(wavelengths)):
            wl_angstrom = wavelengths[i] * 1e4
            err_angstrom = wavelength_err[i] * 1e4
            intended_min = wl_angstrom - err_angstrom
            intended_max = wl_angstrom + err_angstrom

            if intended_max > wl_max:
                if err_angstrom > 0:
                    range_min = max(wl_max - err_angstrom * 2, wl_min)
                    range_max = wl_max
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            elif intended_min < wl_min:
                if err_angstrom > 0:
                    range_min = wl_min
                    range_max = min(wl_min + err_angstrom * 2, wl_max)
                print(f"Using boundary range for {wavelengths[i]:.4f} μm: [{range_min:.1f}, {range_max:.1f}] Å")
            else:
                range_min = intended_min
                range_max = intended_max

            if ld_profile == 'quadratic':
                U_mu.append(sld.compute_quadratic_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=False
                ))
            elif ld_profile == 'power2':
                U_mu.append(sld.compute_power2_ld_coeffs(
                    wavelength_range=[range_min, range_max],
                    mode=mode,
                    return_sigmas=False
                ))
            else:
                raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
    else:
        wl_range_clipped = [max(min(wavelengths)*1e4, wl_min),
                           min(max(wavelengths)*1e4, wl_max)]
        if ld_profile == 'quadratic':
            U_mu = sld.compute_quadratic_ld_coeffs(
                wavelength_range=wl_range_clipped,
                mode=mode,
                return_sigmas=False
            )
        elif ld_profile == 'power2':
            U_mu = sld.compute_power2_ld_coeffs(
                wavelength_range=wl_range_clipped,
                mode=mode,
                return_sigmas=False
            )
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")
        U_mu = jnp.array(U_mu)
    return U_mu

def compute_u_from_c(c1, c2):
    POLY_DEGREE = 12
    MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
    profile = get_I_power2(c1, c2, MUS)
    return calc_poly_coeffs(MUS, profile, poly_degree=POLY_DEGREE)

def fit_polynomial(x, y, poly_orders):
    best_order = None
    best_aic = np.inf
    best_coeffs = None
    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1)
        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs
    return best_coeffs, best_order, best_aic

def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)
    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")

def save_results(wavelengths,wavelength_err,  samples, csv_filename):
    depth_chain = samples['rors']**2
    depth_median = np.nanmedian(depth_chain, axis=0)
    depth_err = np.std(depth_chain, axis=0)
    if depth_median.ndim == 1:
        depth_median = depth_median[:, np.newaxis]
        depth_err = depth_err[:, np.newaxis]
    n_planets = depth_median.shape[1]
    header_cols = ["wavelength", "wavelength_err"]
    for i in range(n_planets):
        header_cols.append(f"depth{i:02d}")
        header_cols.append(f"depth_err{i:02d}")
    header = ",".join(header_cols)
    output_cols = [wavelengths, wavelength_err]
    for i in range(n_planets):
        output_cols.append(depth_median[:, i])
        output_cols.append(depth_err[:, i])
    output_data = np.column_stack(output_cols)
    np.savetxt(csv_filename, output_data, delimiter=",", header=header, comments="")
    print(f"Transmission spectroscopy data saved to {csv_filename}")

def save_detailed_fit_results(time, flux, flux_err, wavelengths, wavelengths_err, samples, map_params,
                               transit_params, detrend_type, output_prefix,
                               total_error_fit=None, gp_trend=None, spot_trend=None, jump_trend=None):
    n_wavelengths = len(wavelengths)
    n_times = len(time)
    print(f"Saving detailed fit results to {output_prefix}_*.csv")
    param_rows = []
    
    def get_stats_local(data_slice):
        med, low, high = get_asym_errors(data_slice)
        return med, np.std(data_slice), low, high

    for i in range(n_wavelengths):
        rors_med, rors_std, rors_l, rors_h = get_stats_local(samples['rors'][:, i])
        depth_med, depth_std, depth_l, depth_h = get_stats_local(samples['rors'][:, i]**2)
        
        row = {
            'wavelength': wavelengths[i],
            'wavelength_err': wavelengths_err[i],
            'rors': rors_med,
            'rors_err': rors_std,
            'rors_err_low': rors_l,
            'rors_err_high': rors_h,
            'depth': depth_med,
            'depth_err': depth_std,
            'depth_err_low': depth_l,
            'depth_err_high': depth_h,
        }
        
        if 'u' in samples:
            u1_med, u1_std, u1_l, u1_h = get_stats_local(samples['u'][:, i, 0])
            u2_med, u2_std, u2_l, u2_h = get_stats_local(samples['u'][:, i, 1])
            row.update({
                'u1': u1_med, 'u1_err': u1_std, 'u1_err_low': u1_l, 'u1_err_high': u1_h,
                'u2': u2_med, 'u2_err': u2_std, 'u2_err_low': u2_l, 'u2_err_high': u2_h
            })

        if 'c1' in samples:
            c1_med, c1_std, c1_l, c1_h = get_stats_local(samples['c1'][:, i])
            row.update({'c1': c1_med, 'c1_err': c1_std, 'c1_err_low': c1_l, 'c1_err_high': c1_h})
            row.update({'u1': c1_med, 'u1_err': c1_std, 'u1_err_low': c1_l, 'u1_err_high': c1_h})

        if 'c2' in samples:
            c2_med, c2_std, c2_l, c2_h = get_stats_local(samples['c2'][:, i])
            row.update({'c2': c2_med, 'c2_err': c2_std, 'c2_err_low': c2_l, 'c2_err_high': c2_h})
            row.update({'u2': c2_med, 'u2_err': c2_std, 'u2_err_low': c2_l, 'u2_err_high': c2_h})

        if detrend_type != 'none':
            c_med, c_std, c_l, c_h = get_stats_local(samples['c'][:, i])
            row.update({'c': c_med, 'c_err': c_std, 'c_err_low': c_l, 'c_err_high': c_h})
            if 'v' in samples:
                v_med, v_std, v_l, v_h = get_stats_local(samples['v'][:, i])
                row.update({'v': v_med, 'v_err': v_std, 'v_err_low': v_l, 'v_err_high': v_h})
        
        for key in ['v2', 'v3', 'v4', 'A', 'tau', 't_jump', 'jump',
                    'spot_amp', 'spot_mu', 'spot_sigma', 'spot_amp2', 'spot_mu2', 'spot_sigma2',
                    'A_gp', 'A_spot', 'A_spot2', 'A_jump']:
            if key in samples:
                med, std, low, high = get_stats_local(samples[key][:, i])
                row.update({
                    key: med,
                    f'{key}_err': std,
                    f'{key}_err_low': low,
                    f'{key}_err_high': high
                })
            
        param_rows.append(row)

    params_df = pd.DataFrame(param_rows)
    params_df.to_csv(f"{output_prefix}_bestfit_params.csv", index=False)

    time = np.asarray(time)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)
    wavelengths = np.asarray(wavelengths)
    wavelengths_err = np.asarray(wavelengths_err)

    meta_path = f"{output_prefix}_wavelengths.csv"
    meta_df = pd.DataFrame({
        "wavelength_center": wavelengths,
        "wavelength_err": wavelengths_err,
    })
    meta_df.to_csv(meta_path, index=False)

    wide_path = f"{output_prefix}_lightcurves_wide.csv"
    columns = ["time"]
    data_cols = [time]
    for i in range(n_wavelengths):
        columns.append(f"flux_{i:03d}")
        columns.append(f"flux_err_{i:03d}")
        data_cols.append(flux[i])
        data_cols.append(flux_err[i])

    wide_data = np.column_stack(data_cols)
    wide_df = pd.DataFrame(wide_data, columns=columns)
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved wavelength metadata to {meta_path}")
    print(f"Saved wide light curves to {wide_path}")
    return params_df, None

def get_robust_sigma(x):
    """Helper to calculate sigma using MAD (robust to outliers)."""
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826 * mad

def calculate_beta_metrics(residuals, dt, cut_factor=5.0):
    residuals = np.array(residuals)
    ndata = len(residuals)
    
    sigma1 = get_robust_sigma(residuals)
    
    cadence_min = dt / 60.0
    
    max_bin_n = ndata // int(cut_factor) 
    
    bin_sizes_points = np.unique(np.logspace(0, np.log10(max_bin_n), 300).astype(int))
    
    measured_rms = []
    expected_rms = []
    bin_sizes_min = []
    betas = []
    
    for N in bin_sizes_points:
        cutoff = ndata - (ndata % N)
        binned_res = residuals[:cutoff].reshape(-1, N).mean(axis=1)
        
        sigma_N_measured = get_robust_sigma(binned_res)
        
        sigma_N_theory = sigma1 / np.sqrt(N)
        
        measured_rms.append(sigma_N_measured)
        expected_rms.append(sigma_N_theory)
        bin_sizes_min.append(N * cadence_min)
        
        betas.append(sigma_N_measured / sigma_N_theory)

    beta_final = np.median(betas)
    
    return beta_final, np.array(bin_sizes_min), np.array(measured_rms), np.array(expected_rms)

def run_beta_monte_carlo(residuals, dt, n_sims=500):
    ndata = len(residuals)
    
    sigma1 = get_robust_sigma(residuals)
    
    mc_betas = []
    
    _, ref_bin_sizes, _, _ = calculate_beta_metrics(residuals, dt)
    all_sim_rms = np.zeros((n_sims, len(ref_bin_sizes)))
    
    for i in range(n_sims):
        synth_res = np.random.normal(0, sigma1, ndata)
        b_sim, _, rms_sim, _ = calculate_beta_metrics(synth_res, dt)
        
        mc_betas.append(b_sim)
        
        if len(rms_sim) == len(ref_bin_sizes):
            all_sim_rms[i, :] = rms_sim
            
    rms_low_1sig = np.percentile(all_sim_rms, 16, axis=0)
    rms_high_1sig = np.percentile(all_sim_rms, 84, axis=0)
    rms_low_2sig = np.percentile(all_sim_rms, 5, axis=0)
    rms_high_2sig = np.percentile(all_sim_rms, 95, axis=0)

    return np.array(mc_betas), rms_low_1sig, rms_high_1sig, rms_low_2sig, rms_high_2sig
def main():
    parser = argparse.ArgumentParser(description="Run transit analysis with YAML config.")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    instrument = cfg['instrument']
    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        nrs = cfg['nrs']
    elif instrument == 'NIRISS/SOSS':
        order = cfg['order']
    
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    resolution = cfg.get('resolution', None)
    pixels = cfg.get('pixels', None)
    
    if resolution is None:
        if pixels is None: raise ValueError('Must Specify Resolutions or Pixels')
        bins = pixels
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)
    elif pixels is None:
        bins = resolution
        high_resolution_bins = bins.get('high', None)
        low_resolution_bins = bins.get('low', None)

    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']
    mask_integrations_start = outlier_clip.get('mask_integrations_start', None)
    mask_integrations_end = outlier_clip.get('mask_integrations_end', None)

    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    fits_file = os.path.join(input_dir, cfg.get('fits_file'))
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform(host_device)
    key_master = jax.random.PRNGKey(555)

    detrending_type = flags.get('detrending_type', 'linear')
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    fix_ld = flags.get('fix_ld', False)
    need_lowres = flags.get('need_lowres', True)
    mask_start = flags.get('mask_start', False)
    mask_end = flags.get('mask_end', False)
    spot_amp = flags.get('spot_amp', 0.0)
    spot_mu = flags.get('spot_center', 0.0)
    spot_sigma = flags.get('spot_width', 0.0)
    spot_amp2 = flags.get('spot_amp2', flags.get('spot_amp_2', 0.0))
    spot_mu2 = flags.get('spot_center2', flags.get('spot_center_2', 0.0))
    spot_sigma2 = flags.get('spot_width2', flags.get('spot_width_2', 0.0))
    t_jump_guess = flags.get('t_jump_guess', None)
    jump_guess = flags.get('jump_guess', 0.0)
    save_trace = flags.get('save_whitelight_trace', False)
    vmap_chunk = flags.get('vmap_chunk', False)
    vmap_chunk_size = None
    if isinstance(vmap_chunk, (int, float)) and not isinstance(vmap_chunk, bool):
        vmap_chunk_size = int(vmap_chunk)
    elif vmap_chunk is True:
        vmap_chunk_size = 50

    ld_profile = flags.get('ld_profile', 'quadratic')

    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    periods = jnp.atleast_1d(planet_cfg['period'])
    n_planets = len(periods)
    durations = jnp.atleast_1d(planet_cfg['duration'])
    t0s = jnp.atleast_1d(planet_cfg['t0'])
    bs = jnp.atleast_1d(planet_cfg['b'])
    rors = jnp.atleast_1d(planet_cfg['rprs'])
    depths = rors**2

    PERIOD_FIXED = periods
    PRIOR_DUR = durations
    PRIOR_T0 = t0s
    PRIOR_B = bs
    PRIOR_RPRS = rors
    PRIOR_DEPTH = depths

    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')
    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        mini_instrument = f'nrs{nrs}'
    elif instrument == 'NIRISS/SOSS':
        mini_instrument = f'order{order}'
    else:
        mini_instrument = ''

    instrument_full_str = f"{planet_str}_{instrument.replace('/', '_')}_{mini_instrument}"
    if bins == resolution:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}LR_{high_resolution_bins}HR.pkl'
    elif bins == pixels:
        spectro_data_file = output_dir + f'/{instrument_full_str}_spectroscopy_data_{low_resolution_bins}pix_{high_resolution_bins}pix.pkl'

    lr_bin_str = f'R{low_resolution_bins}' if bins == resolution else f'pix{low_resolution_bins}'
    hr_bin_str = f'R{high_resolution_bins}' if bins == resolution else f'pix{high_resolution_bins}'

    if not os.path.exists(spectro_data_file) or mask_start is not False:
        data = process_spectroscopy_data(instrument, input_dir, output_dir, planet_str, cfg, fits_file, mask_start, mask_end, mask_integrations_start, mask_integrations_end)
        data.save(spectro_data_file)
    else:
        data = SpectroData.load(spectro_data_file)
    
    print("Data loaded.")
    print(f"Data length: {data.time.shape}")
    
    timebin_cfg = cfg.get("time_binning", {})
    do_timebin = timebin_cfg.get("enabled", False) or flags.get("bin_time", False)

    if do_timebin:
        dt_seconds = (
            timebin_cfg.get("dt_seconds", None)
            or flags.get("bin_dt_seconds", None)
            or 120.0
        )
        method = (
            timebin_cfg.get("method", None)
            or flags.get("bin_method", None)
            or "weighted"
        )
        bin_whitelight = timebin_cfg.get("whitelight", True) if "whitelight" in timebin_cfg else flags.get("bin_whitelight", True)
        bin_spectroscopic = timebin_cfg.get("spectroscopic", True) if "spectroscopic" in timebin_cfg else flags.get("bin_spectroscopic", True)

        data = bin_spectrodata_in_time(
            data,
            dt_seconds=float(dt_seconds),
            method=str(method),
            bin_whitelight=bool(bin_whitelight),
            bin_spectroscopic=bool(bin_spectroscopic)
        )
        print('Finished Binning! Binned data length: ', data.time.shape)



    stringcheck = os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')

    if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'MIRI/LRS', 'NIRSPEC/G140H', 'NIRSPEC/G235H']:
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned,0.0, instrument, ld_profile=ld_profile)
    elif instrument == 'NIRISS/SOSS':
        U_mu_wl = get_limb_darkening(sld, data.wavelengths_unbinned, 0.0, instrument, order=order, ld_profile=ld_profile)
    if not stringcheck or ('gp' in detrending_type):
        if not os.path.exists(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv'):
            plt.scatter(data.wl_time, data.wl_flux, c='k', s=6, alpha=0.5)
            plt.savefig(f'{output_dir}/stuff.png')
            plt.close()
            print('Fitting whitelight for outliers and bestfit parameters')
            hyper_params_wl = {
                "duration": PRIOR_DUR,
                "t0": PRIOR_T0,
                'period': PERIOD_FIXED,
                'u': U_mu_wl
            }
            if '2spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
                hyper_params_wl['spot_guess2'] = spot_mu2
            elif 'spot' in detrending_type:
                hyper_params_wl['spot_guess'] = spot_mu
            if 'linear_discontinuity' in detrending_type:
                hyper_params_wl['t_jump_guess'] = t_jump_guess if t_jump_guess is not None else 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                hyper_params_wl['jump_guess'] = jump_guess

            hyper_params_wl['u'] = U_mu_wl

            init_params_wl = {
                'c': 1.0,
                'v': 0.0,
                'log_jitter': jnp.log(1e-4),
                'b': PRIOR_B,
                'rors': PRIOR_RPRS
            }
            init_params_wl['u'] = U_mu_wl

            for i in range(n_planets):
                init_params_wl[f'logD_{i}'] = jnp.log(PRIOR_DUR[i])
                init_params_wl[f't0_{i}'] = PRIOR_T0[i]
                init_params_wl[f'_b_{i}'] = PRIOR_B[i]
                init_params_wl[f'depths_{i}'] = PRIOR_DEPTH[i]

            if 'quadratic' in detrending_type:
                init_params_wl['v2'] = 0.0
            if 'cubic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0
            if 'quartic' in detrending_type:
                init_params_wl['v2'] = 0.0; init_params_wl['v3'] = 0.0; init_params_wl['v4'] = 0.0
            if 'explinear' in detrending_type:
                init_params_wl['A'] = 0.001; init_params_wl['tau'] = 0.5
            if 'gp' in detrending_type:
                init_params_wl['GP_log_sigma'] = jnp.log(jnp.nanmedian(data.wl_flux_err))
                init_params_wl['GP_log_rho'] = jnp.log(0.1)
            if 'linear_discontinuity' in detrending_type:
                if t_jump_guess is not None:
                    init_params_wl['t_jump'] = t_jump_guess
                else:
                    init_params_wl['t_jump'] = 0.5 * (jnp.min(data.wl_time) + jnp.max(data.wl_time))
                init_params_wl['jump'] = jump_guess
            if 'spot' in detrending_type:
                init_params_wl['spot_amp'] = spot_amp
                init_params_wl['spot_mu'] = spot_mu
                init_params_wl['spot_sigma'] = spot_sigma
            if '2spot' in detrending_type:
                init_params_wl['spot_amp2'] = spot_amp2
                init_params_wl['spot_mu2'] = spot_mu2
                init_params_wl['spot_sigma2'] = spot_sigma2

            whitelight_model_for_run = create_whitelight_model(detrend_type=detrending_type, n_planets=n_planets, ld_profile=ld_profile)
            
            if 'gp' in detrending_type:
                print("--- Running Pre-Fit with Linear Detrending to stabilize GP ---")
                whitelight_model_prefit = create_whitelight_model(detrend_type='linear', n_planets=n_planets, ld_profile=ld_profile)
                init_params_prefit = init_params_wl.copy()
                init_params_prefit.pop('GP_log_sigma', None)
                init_params_prefit.pop('GP_log_rho', None)
                soln = optimx.optimize(whitelight_model_prefit, start=init_params_prefit)(
                    key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl
                )
            else:
                soln =  optimx.optimize(whitelight_model_for_run, start=init_params_wl)(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            
            
            print("Plotting Initial vs. Optimized Model sanity check...")
            
            def _get_sanity_model(params, t_vals):
                if 'gp' in detrending_type:
                    return compute_lc_linear(params, t_vals)
                elif detrending_type == 'linear': return compute_lc_linear(params, t_vals)
                elif detrending_type == 'quadratic': return compute_lc_quadratic(params, t_vals)
                elif detrending_type == 'cubic': return compute_lc_cubic(params, t_vals)
                elif detrending_type == 'quartic': return compute_lc_quartic(params, t_vals)
                elif detrending_type == 'explinear': return compute_lc_explinear(params, t_vals)
                elif detrending_type == 'linear_discontinuity': return compute_lc_linear_discontinuity(params, t_vals)
                elif detrending_type == 'spot': return compute_lc_spot(params, t_vals)
                elif detrending_type == '2spot': return compute_lc_2spot(params, t_vals)
                elif detrending_type == 'none': return compute_lc_none(params, t_vals)
                else: return compute_lc_linear(params, t_vals)
            def _soln_to_physical_params(soln, base_params, n_planets=1):
                p = dict(base_params)
                if "u" in soln:
                    p["u"] = soln["u"]

                for k in ["c", "v", "a2", "a3", "a4", "A", "tau", "t_break", "delta",
                          "spot_amp", "spot_mu", "spot_sigma", "spot_amp2", "spot_mu2", "spot_sigma2",
                          "A_spot", "t_spot", "sigma_spot"]:
                    if k in soln:
                        p[k] = soln[k]

                def have_all(prefix):
                    return all(f"{prefix}_{i}" in soln for i in range(n_planets))

                if have_all("logD"):
                    p["duration"] = jnp.array([jnp.exp(soln[f"logD_{i}"]) for i in range(n_planets)])
                if have_all("_b"):
                    p["b"] = jnp.array([jnp.abs(soln[f"_b_{i}"]) for i in range(n_planets)])
                if have_all("depths"):
                    p["rors"] = jnp.array([jnp.sqrt(soln[f"depths_{i}"]) for i in range(n_planets)])
                if have_all("t0"):
                    p["t0"] = jnp.array([soln[f"t0_{i}"] for i in range(n_planets)])

                return p

            try:

                params_complete = hyper_params_wl.copy()
                params_complete.update(init_params_wl)

                n_planets_sanity = len(jnp.atleast_1d(PERIOD_FIXED))

                def _ensure_len(value, n):
                    arr = jnp.atleast_1d(value)
                    if arr.size == 1 and n > 1:
                        return jnp.repeat(arr, n)
                    return arr

                params_complete["period"] = _ensure_len(params_complete.get("period", PERIOD_FIXED), n_planets_sanity)
                params_complete["duration"] = _ensure_len(params_complete.get("duration", PRIOR_DUR), n_planets_sanity)
                params_complete["t0"] = _ensure_len(params_complete.get("t0", PRIOR_T0), n_planets_sanity)
                params_complete["b"] = _ensure_len(params_complete.get("b", PRIOR_B), n_planets_sanity)
                params_complete["rors"] = _ensure_len(params_complete.get("rors", PRIOR_RPRS), n_planets_sanity)

                key_sanity = jax.random.PRNGKey(0) if key_master is None else key_master
                keys = jax.random.split(key_sanity, num=3)

                stage1 = optimx.optimize(
                    whitelight_model_for_run,
                    sites=["logD_0", "t0_0", "_b_0"] if n_planets_sanity == 1 else None,
                    start=init_params_wl,
                )
                soln = stage1(keys[0], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                stage2_sites = ["depths_0"]
                if ld_profile == "quadratic":
                    stage2_sites.append("u")
                elif ld_profile == "power2":
                    stage2_sites.extend(["c1", "c2"])
                if n_planets_sanity != 1:
                    stage2_sites = None

                stage2 = optimx.optimize(
                    whitelight_model_for_run,
                    sites=stage2_sites,
                    start=soln,
                )
                soln = stage2(keys[1], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                stage3 = optimx.optimize(
                    whitelight_model_for_run,
                    start=soln,
                )
                soln = stage3(keys[2], data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)

                params_opt = _soln_to_physical_params(soln, params_complete, n_planets=n_planets_sanity)
                if "rors" not in params_opt and "depths" in params_opt:
                    params_opt["rors"] = jnp.sqrt(params_opt["depths"])

                flux_init = _get_sanity_model(params_complete, data.wl_time)
                flux_opt  = _get_sanity_model(params_opt,      data.wl_time)
                print('Optimized Init Params:', params_opt)    
                plt.figure(figsize=(10, 6))
                plt.scatter(data.wl_time, data.wl_flux, c='k', s=10, alpha=0.3, label='Data')
                
                plt.plot(data.wl_time, flux_init, color='red', linestyle='--', lw=2, alpha=0.7, 
                         label='Init Guess')
                plt.plot(data.wl_time, flux_opt, color='blue', lw=2.5, 
                         label='Optimized (Gradient)')
                
                plt.title(f'Sanity Check: {detrending_type} (GP Pre-fit: {"gp" in detrending_type})')
                plt.xlabel('Time (BJD)')
                plt.ylabel('Flux')
                plt.legend()
                plt.tight_layout()
                
                sanity_plot_path = f'{output_dir}/00_{instrument_full_str}_init_vs_opt_check.png'
                plt.savefig(sanity_plot_path)
                plt.close()
                print(f"Saved sanity check plot to: {sanity_plot_path}")
                
            except Exception as e:
                print(f"Could not create sanity plot: {e}")
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(whitelight_model_for_run, regularize_mass_matrix=False, init_strategy=numpyro.infer.init_to_value(values=soln), target_accept_prob=0.9),
                num_warmup=1000, num_samples=1000, progress_bar=True, jit_model_args=True
            )
            mcmc.run(key_master, data.wl_time, data.wl_flux_err, y=data.wl_flux, prior_params=hyper_params_wl)
            inf_data = az.from_numpyro(mcmc)
            if save_trace: az.to_netcdf(inf_data, f'whitelight_trace_{n_planets}planets.nc')
            wl_samples = mcmc.get_samples()
            print(az.summary(inf_data, var_names=None, round_to=7))

            bestfit_params_wl = {
                'period': PERIOD_FIXED,
            }
            def set_param_stats(name, data_samples, axis=0):
                med, low, high = get_asym_errors(data_samples, axis=axis)
                bestfit_params_wl[name] = med
                bestfit_params_wl[f'{name}_err_low'] = low
                bestfit_params_wl[f'{name}_err_high'] = high

            if 'c1' in wl_samples and ld_profile == 'power2':
                set_param_stats('c1', wl_samples['c1'], axis=0)
                set_param_stats('c2', wl_samples['c2'], axis=0)
                POLY_DEGREE = 12
                MUS = jnp.linspace(0.0, 1.00, 300, endpoint=True)
                power2_profile = get_I_power2(bestfit_params_wl['c1'], bestfit_params_wl['c2'], MUS)
                u_poly = calc_poly_coeffs(MUS, power2_profile, poly_degree=POLY_DEGREE)
                bestfit_params_wl['u'] = u_poly
            else:
                set_param_stats('u', wl_samples['u'], axis=0)

            durations_fit, t0s_fit, bs_fit, rors_fit = [], [], [], []
            durations_err, t0s_err, bs_err, rors_err, depths_err = [], [], [], [], []
            durations_err_low, t0s_err_low, bs_err_low, rors_err_low, depths_err_low = [], [], [], [], []
            durations_err_high, t0s_err_high, bs_err_high, rors_err_high, depths_err_high = [], [], [], [], []

            for i in range(n_planets):
                med_d, low_d, high_d = get_asym_errors(wl_samples[f'duration_{i}'])
                med_t0, low_t0, high_t0 = get_asym_errors(wl_samples[f't0_{i}'])
                med_b, low_b, high_b = get_asym_errors(wl_samples[f'b_{i}'])
                med_r, low_r, high_r = get_asym_errors(wl_samples[f'rors_{i}'])
                med_depth, low_depth, high_depth = get_asym_errors(wl_samples[f'rors_{i}']**2)

                durations_fit.append(med_d)
                t0s_fit.append(med_t0)
                bs_fit.append(med_b)
                rors_fit.append(med_r)
                
                durations_err.append(jnp.std(wl_samples[f'duration_{i}']))
                t0s_err.append(jnp.std(wl_samples[f't0_{i}']))
                bs_err.append(jnp.std(wl_samples[f'b_{i}']))
                rors_err.append(jnp.std(wl_samples[f'rors_{i}']))
                depths_err.append(jnp.std(wl_samples[f'rors_{i}']**2))

                durations_err_low.append(low_d); durations_err_high.append(high_d)
                t0s_err_low.append(low_t0); t0s_err_high.append(high_t0)
                bs_err_low.append(low_b); bs_err_high.append(high_b)
                rors_err_low.append(low_r); rors_err_high.append(high_r)
                depths_err_low.append(low_depth); depths_err_high.append(high_depth)

            bestfit_params_wl['duration'] = jnp.array(durations_fit)
            bestfit_params_wl['t0'] = jnp.array(t0s_fit)
            bestfit_params_wl['b'] = jnp.array(bs_fit)
            bestfit_params_wl['rors'] = jnp.array(rors_fit)
            bestfit_params_wl['depths'] = jnp.array(rors_fit)**2

            bestfit_params_wl['duration_err'] = jnp.array(durations_err)
            bestfit_params_wl['t0_err'] = jnp.array(t0s_err)
            bestfit_params_wl['b_err'] = jnp.array(bs_err)
            bestfit_params_wl['rors_err'] = jnp.array(rors_err)
            bestfit_params_wl['depths_err'] = jnp.array(depths_err)
            
            bestfit_params_wl['duration_err_low'] = jnp.array(durations_err_low)
            bestfit_params_wl['duration_err_high'] = jnp.array(durations_err_high)
            bestfit_params_wl['t0_err_low'] = jnp.array(t0s_err_low)
            bestfit_params_wl['t0_err_high'] = jnp.array(t0s_err_high)
            bestfit_params_wl['b_err_low'] = jnp.array(bs_err_low)
            bestfit_params_wl['b_err_high'] = jnp.array(bs_err_high)
            bestfit_params_wl['rors_err_low'] = jnp.array(rors_err_low)
            bestfit_params_wl['rors_err_high'] = jnp.array(rors_err_high)
            bestfit_params_wl['depths_err_low'] = jnp.array(depths_err_low)
            bestfit_params_wl['depths_err_high'] = jnp.array(depths_err_high)

            set_param_stats('error', wl_samples['error'])
            
            if detrending_type != 'none':
                set_param_stats('c', wl_samples['c'])
                if detrending_type != 'gp': 
                    set_param_stats('v', wl_samples['v'])
                if 'v' in wl_samples and 'v' not in bestfit_params_wl:
                     set_param_stats('v', wl_samples['v'])

            if 'v2' in wl_samples: set_param_stats('v2', wl_samples['v2'])
            if 'v3' in wl_samples: set_param_stats('v3', wl_samples['v3'])
            if 'v4' in wl_samples: set_param_stats('v4', wl_samples['v4'])
            if 'explinear' in detrending_type:
                set_param_stats('A', wl_samples['A'])
                set_param_stats('tau', wl_samples['tau'])
            if 'spot' in detrending_type:
                set_param_stats('spot_amp', wl_samples['spot_amp'])
                set_param_stats('spot_mu', wl_samples['spot_mu'])
                set_param_stats('spot_sigma', wl_samples['spot_sigma'])
            if '2spot' in detrending_type:
                set_param_stats('spot_amp2', wl_samples['spot_amp2'])
                set_param_stats('spot_mu2', wl_samples['spot_mu2'])
                set_param_stats('spot_sigma2', wl_samples['spot_sigma2'])
            if 'linear_discontinuity' in detrending_type:
                set_param_stats('t_jump', wl_samples['t_jump'])
                set_param_stats('jump', wl_samples['jump'])
    
            if 'gp' in detrending_type:
                set_param_stats('GP_log_sigma', wl_samples['GP_log_sigma'])
                set_param_stats('GP_log_rho', wl_samples['GP_log_rho'])

            spot_trend, spot_trend2, jump_trend = None, None, None
            if 'spot' in detrending_type and '2spot' not in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
            if '2spot' in detrending_type:
                spot_trend = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp"],
                    bestfit_params_wl["spot_mu"],
                    jnp.abs(bestfit_params_wl["spot_sigma"])
                )
                spot_trend2 = spot_crossing(
                    data.wl_time,
                    bestfit_params_wl["spot_amp2"],
                    bestfit_params_wl["spot_mu2"],
                    jnp.abs(bestfit_params_wl["spot_sigma2"])
                )
            if 'linear_discontinuity' in detrending_type:
                jump_trend = bestfit_params_wl["jump"] * _soft_step_np(np.array(data.wl_time), bestfit_params_wl["t_jump"])

            if 'gp' in detrending_type:
                if 'quartic' in detrending_type:
                    gp_mean_func = compute_lc_quartic_gp_mean
                elif 'cubic' in detrending_type:
                    gp_mean_func = compute_lc_cubic_gp_mean
                elif 'quadratic' in detrending_type:
                    gp_mean_func = compute_lc_quadratic_gp_mean
                elif 'explinear' in detrending_type:
                    gp_mean_func = compute_lc_explinear_gp_mean
                elif 'linear' in detrending_type:
                    gp_mean_func = compute_lc_linear_gp_mean
                else:
                    gp_mean_func = compute_lc_gp_mean

                wl_kernel = tinygp.kernels.quasisep.Matern32(
                    scale=jnp.exp(bestfit_params_wl['GP_log_rho']),
                    sigma=jnp.exp(bestfit_params_wl['GP_log_sigma']),
                )
                wl_gp = tinygp.GaussianProcess(
                    wl_kernel,
                    data.wl_time,
                    diag=bestfit_params_wl['error']**2,
                    mean=partial(gp_mean_func, bestfit_params_wl),
                )
                cond_gp = wl_gp.condition(data.wl_flux, data.wl_time).gp
                mu, var = cond_gp.loc, cond_gp.variance
                wl_transit_model = mu
                
                planet_model_only = compute_transit_model(bestfit_params_wl, data.wl_time)
                trend_flux_total = mu - planet_model_only - 1.0
                
                parametric_mean_val = gp_mean_func(bestfit_params_wl, data.wl_time)
                gp_stochastic_component = mu - parametric_mean_val

            elif detrending_type == 'linear':
                wl_transit_model = compute_lc_linear(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'quadratic':
                wl_transit_model = compute_lc_quadratic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'cubic':
                wl_transit_model = compute_lc_cubic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'quartic':
                wl_transit_model = compute_lc_quartic(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'explinear':
                wl_transit_model = compute_lc_explinear(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'none':
                wl_transit_model = compute_lc_none(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'linear_discontinuity':
                wl_transit_model = compute_lc_linear_discontinuity(bestfit_params_wl, data.wl_time)
            elif detrending_type == 'spot':
                wl_transit_model = compute_lc_spot(bestfit_params_wl, data.wl_time)
            elif detrending_type == '2spot':
                wl_transit_model = compute_lc_2spot(bestfit_params_wl, data.wl_time)
            else:
                 print('Error with model, not defined!')
                 exit()

            wl_residual = data.wl_flux - wl_transit_model
            wl_sigma = 1.4826 * jnp.nanmedian(np.abs(wl_residual - jnp.nanmedian(wl_residual)))
            wl_mad_mask = jnp.abs(wl_residual - jnp.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma
            wl_sigma_post_clip = 1.4826 * jnp.nanmedian(jnp.abs(wl_residual[~wl_mad_mask] - jnp.nanmedian(wl_residual[~wl_mad_mask])))
            spec_good_mask = (~wl_mad_mask if len(wl_mad_mask) == len(data.time)
                              else np.ones(len(data.time), dtype=bool))

            plt.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            plt.scatter(data.wl_time, data.wl_flux, s=6, c='k', zorder=1, alpha=0.5)

            plt.savefig(f"{output_dir}/11_{instrument_full_str}_whitelightmodel.png")
            plt.close()

            plt.scatter(data.wl_time, wl_residual, s=6, c='k')
            plt.title('WL Pre-outlier rejection residual')
            plt.savefig(f"{output_dir}/12_{instrument_full_str}_whitelightresidual.png")
            plt.close()

            t_masked = data.wl_time[~wl_mad_mask]
            f_masked = data.wl_flux[~wl_mad_mask]

            if 'gp' in detrending_type:
                planet_model_masked = compute_transit_model(bestfit_params_wl, t_masked)
                mu_masked = mu[~wl_mad_mask] 
                total_trend_at_points = mu_masked - planet_model_masked
                detrended_flux = f_masked - (total_trend_at_points - 1.0)
                gp_stochastic_at_masked = gp_stochastic_component[~wl_mad_mask]

            else:
                trend = _trend_from_params_np(
                    detrending_type,
                    np.array(t_masked),
                    bestfit_params_wl
                )
                detrended_flux = f_masked - trend + 1.0

            plt.scatter(t_masked, detrended_flux, c='k', s=6, alpha=0.5)
            plt.title(f'Detrended WLC: Sigma {round(wl_sigma_post_clip*1e6)} PPM')
            plt.savefig(f'{output_dir}/14_{instrument_full_str}_whitelightdetrended.png')
            plt.close()

            transit_only_model = compute_transit_model(bestfit_params_wl, t_masked) + 1.0
            residuals_detrended = detrended_flux - transit_only_model 

            
            fig = plt.figure(figsize=(26, 12))
            
            gs = gridspec.GridSpec(3, 4, figure=fig, 
                                height_ratios=[1, 1, 1.5], 
                                width_ratios=[1, 1, 1.3, 1.3], 
                                hspace=0.3, wspace=0.25)

            duration_bin = jnp.max(jnp.atleast_1d(bestfit_params_wl['duration']))
            b_time, b_flux = jax_bin_lightcurve(jnp.array(data.wl_time), 
                                                jnp.array(data.wl_flux), 
                                                duration_bin)
            
            b_time_det, b_flux_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                                        jnp.array(detrended_flux), 
                                                        duration_bin)
            b_time_det, b_res_det = jax_bin_lightcurve(jnp.array(t_masked), 
                                            jnp.array(residuals_detrended), 
                                            duration_bin)
            bin_style = dict(c='darkviolet', edgecolors='darkslateblue', s=40,  zorder=10, label='Binned (8/dur)')

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax1.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax1.set_title('Raw Light Curve', fontsize=14)
            ax1.set_ylabel('Flux', fontsize=12)
            ax1.tick_params(labelbottom=False)

            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax2.scatter(data.wl_time, data.wl_flux, c='k', s=4, alpha=0.2)
            ax2.scatter(np.array(b_time), np.array(b_flux), **bin_style)
            ax2.plot(data.wl_time, wl_transit_model, color="mediumorchid", lw=2, zorder=3)
            ax2.set_title('Raw Light Curve + Best-fit Model', fontsize=14)
            ax2.set_ylabel('Flux', fontsize=12)
            ax2.tick_params(labelbottom=False)

            gs_nested = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[2, 0], 
                height_ratios=[2, 1],
                hspace=0.0
            )

            ax3_top = fig.add_subplot(gs_nested[0], sharex=ax1)
            ax3_top.scatter(t_masked, detrended_flux, c='k', s=4, alpha=0.2, label='Detrended Data')
            ax3_top.plot(t_masked, transit_only_model, color="mediumorchid", lw=2, zorder=3, label='Transit Model')
            ax3_top.scatter(np.array(b_time_det), np.array(b_flux_det), **bin_style)
            ax3_top.set_ylabel('Normalized Flux', fontsize=12)
            ax3_top.set_title('Detrended Light Curve', fontsize=14)
            plt.setp(ax3_top.get_xticklabels(), visible=False)

            ax3_bot = fig.add_subplot(gs_nested[1], sharex=ax3_top)
            ax3_bot.scatter(t_masked, residuals_detrended * 1e6, c='k', s=4, alpha=0.2)
            ax3_bot.axhline(0, color='mediumorchid', lw=4, zorder=3, linestyle='--')
            ax3_bot.scatter(np.array(b_time_det), np.array(b_res_det) * 1e6 , **bin_style)
            ax3_bot.set_ylabel('Res. (ppm)', fontsize=10)
            ax3_bot.set_xlabel('Time (BJD)', fontsize=12)

            dt = np.median(np.diff(data.wl_time)) * 86400 
            residuals_arr = np.array(wl_residual[~wl_mad_mask])
            beta, bin_sizes_min, measured_rms, expected_rms = calculate_beta_metrics(residuals_arr, dt)
            mc_betas, rms_lo_1, rms_hi_1, rms_lo_2, rms_hi_2 = run_beta_monte_carlo(residuals_arr, dt, n_sims=500)

            mu_sim, std_sim = norm.fit(mc_betas)
            z_score = (beta - mu_sim) / std_sim

            print(f"Beta: {beta:.4f}")
            print(f"Measured Beta: {beta:.3f}")
            print(f"MC Mean Beta:  {mu_sim:.3f}")
            print(f"MC Std Dev:    {std_sim:.3f}")
            print(f"Significance:  {z_score:.2f} sigma")

            ax_rms = fig.add_subplot(gs[0:2, 1])
            ax_rms.loglog(bin_sizes_min, expected_rms * 1e6, 'k--', lw=1.5, label='Theory $1/\sqrt{N}$')
            ax_rms.fill_between(bin_sizes_min, rms_lo_2 * 1e6, rms_hi_2 * 1e6, color='gray', alpha=0.2, label='White Noise ($2\sigma$)')
            ax_rms.fill_between(bin_sizes_min, rms_lo_1 * 1e6, rms_hi_1 * 1e6, color='gray', alpha=0.4, label='White Noise ($1\sigma$)')
            ax_rms.loglog(bin_sizes_min, measured_rms * 1e6, color='teal', lw=2, marker='o', markersize=5, label=f'Data (Beta={beta:.2f})')
            ax_rms.set_xlabel('Bin Size (minutes)', fontsize=12)
            ax_rms.set_ylabel('RMS (ppm)', fontsize=12)
            ax_rms.set_title('Time-Correlated Noise', fontsize=14)
            ax_rms.grid(True, which="both", alpha=0.2)

            ax_beta = fig.add_subplot(gs[2, 1])
            n, bins, patches = ax_beta.hist(mc_betas, bins=30, color='silver', alpha=0.6, density=True, label='Simulated White Noise')
            xmin, xmax = ax_beta.get_xlim()
            x_plot = np.linspace(xmin, xmax, 100)
            p_plot = norm.pdf(x_plot, mu_sim, std_sim)
            ax_beta.plot(x_plot, p_plot, 'k--', linewidth=2, label='Gaussian Fit')
            ax_beta.axvline(beta, color='teal', lw=3, label=f'Measured: {beta:.2f}')
            sig_color = 'green' if abs(z_score) < 2.0 else ('orange' if abs(z_score) < 3.0 else 'firebrick')
            ax_beta.text(0.95, 0.85, f"Significance: {z_score:.1f}$\sigma$", 
                       transform=ax_beta.transAxes, ha='right', fontsize=14, color=sig_color, fontweight='bold')
            ax_beta.set_xlabel('Beta Factor', fontsize=12)
            ax_beta.set_ylabel('Probability Density', fontsize=12)
            ax_beta.set_title("Beta Significance Test", fontsize=14)

            gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[:, 2:], wspace=0.05)
            
            ax_hr_flux = fig.add_subplot(gs_right[0])
            ax_hr_res = fig.add_subplot(gs_right[1], sharey=ax_hr_flux) 

            n_hr_bins = data.flux_hr.shape[0]
            hr_indices = np.linspace(0, n_hr_bins-1, 10, dtype=int)
            colors = cm.turbo(np.linspace(0, 1, 10))
            
            est_depth = np.nanmedian(bestfit_params_wl['depths'])
            if est_depth < 1e-4: offset_step = 0.0025
            elif est_depth < 1e-3: offset_step = 0.0075
            else: offset_step = 0.02

            wl_time_vector = np.array(data.time[spec_good_mask])
            wl_model_vector = np.array(compute_transit_model(bestfit_params_wl, jnp.array(wl_time_vector)) + 1.0)
            time_center = float(bestfit_params_wl['t0'][0])
            res_zoom_factor = 1.0 

            for idx_i, bin_idx in enumerate(hr_indices):
                raw_flux_hr = data.flux_hr[bin_idx]
                flux_hr_masked = raw_flux_hr[spec_good_mask]
                
                baseline_norm = np.nanmedian(flux_hr_masked[:50])
                norm_flux_hr = flux_hr_masked / baseline_norm
                
                residuals_hr_check = norm_flux_hr - wl_model_vector
                
                mad_ppm = 1.4826 * np.nanmedian(np.abs(residuals_hr_check)) * 1e6

                y_offset = idx_i * offset_step
                
                ax_hr_flux.scatter(wl_time_vector - time_center, norm_flux_hr + y_offset, 
                                 color=colors[idx_i], s=5, alpha=0.6, edgecolors='none')
                
                ax_hr_flux.plot(wl_time_vector - time_center, wl_model_vector + y_offset, 
                              color='dimgray', lw=2.0, alpha=0.3, zorder=2)
                ax_hr_flux.plot(wl_time_vector - time_center, wl_model_vector + y_offset, 
                              color=colors[idx_i], lw=1.0, alpha=0.9, linestyle='-', zorder=3)
                
                wl_val = data.wavelengths_hr[bin_idx]
                annotation_y = 1.0 + y_offset + (offset_step * 0.33)
                ax_hr_flux.text(wl_time_vector.min() - time_center, annotation_y, 
                              f"{wl_val:.2f} $\mu$m", 
                              fontsize=9, fontweight='bold', color=colors[idx_i])

                res_plotted = (residuals_hr_check * res_zoom_factor) + 1.0 + y_offset
                
                ax_hr_res.scatter(wl_time_vector - time_center, res_plotted, 
                                color=colors[idx_i], s=5, alpha=0.6, edgecolors='none')
                
                ax_hr_res.text(wl_time_vector.min() - time_center, annotation_y, 
                               f"$\sigma$={int(mad_ppm)} ppm", 
                               fontsize=9, fontweight='bold', color=colors[idx_i])
                
                ax_hr_res.axhline(1.0 + y_offset, color='black', linestyle='--', lw=1, alpha=0.3)

            ax_hr_flux.set_xlabel("Time from Mid-Transit (days)", fontsize=12)
            ax_hr_res.set_xlabel("Time from Mid-Transit (days)", fontsize=12)
            
            ax_hr_flux.set_yticks([])
            ax_hr_res.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/15_{instrument_full_str}_whitelight_summary.png')
            plt.close(fig)

            
            np.save(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy', arr=wl_mad_mask)
            
            if 'gp' in detrending_type:
                df = pd.DataFrame({
                    'wl_flux': data.wl_flux, 
                    'gp_flux': mu,
                    'gp_err': jnp.sqrt(var), 
                    'gp_trend': gp_stochastic_component
                }) 
                df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            
            rows = []
            for i in range(n_planets):
                row = {
                    'planet_id': i,
                    'period': bestfit_params_wl['period'][i],
                    'duration': bestfit_params_wl['duration'][i],
                    't0': bestfit_params_wl['t0'][i],
                    'b': bestfit_params_wl['b'][i],
                    'rors': bestfit_params_wl['rors'][i],
                    'depths': bestfit_params_wl['depths'][i],
                    'duration_err': bestfit_params_wl['duration_err'][i],
                    't0_err': bestfit_params_wl['t0_err'][i],
                    'b_err': bestfit_params_wl['b_err'][i],
                    'rors_err': bestfit_params_wl['rors_err'][i],
                    'depths_err': bestfit_params_wl['depths_err'][i],
                    'duration_err_low': bestfit_params_wl['duration_err_low'][i],
                    'duration_err_high': bestfit_params_wl['duration_err_high'][i],
                    't0_err_low': bestfit_params_wl['t0_err_low'][i],
                    't0_err_high': bestfit_params_wl['t0_err_high'][i],
                    'b_err_low': bestfit_params_wl['b_err_low'][i],
                    'b_err_high': bestfit_params_wl['b_err_high'][i],
                    'rors_err_low': bestfit_params_wl['rors_err_low'][i],
                    'rors_err_high': bestfit_params_wl['rors_err_high'][i],
                    'depths_err_low': bestfit_params_wl['depths_err_low'][i],
                    'depths_err_high': bestfit_params_wl['depths_err_high'][i],
                }
                
                def add_scalar_param(name):
                    if name in bestfit_params_wl:
                        row[name] = bestfit_params_wl[name]
                        if f'{name}_err_low' in bestfit_params_wl:
                            row[f'{name}_err_low'] = bestfit_params_wl[f'{name}_err_low']
                            row[f'{name}_err_high'] = bestfit_params_wl[f'{name}_err_high']

                if detrending_type != 'none':
                    add_scalar_param('c')
                    add_scalar_param('v')
                
                if 'c1' in bestfit_params_wl:
                    add_scalar_param('c1')
                    add_scalar_param('c2')
                    row['u1'] = bestfit_params_wl['c1']
                    row['u2'] = bestfit_params_wl['c2']
                    row['u1_err_low'] = bestfit_params_wl['c1_err_low']
                    row['u1_err_high'] = bestfit_params_wl['c1_err_high']
                    row['u2_err_low'] = bestfit_params_wl['c2_err_low']
                    row['u2_err_high'] = bestfit_params_wl['c2_err_high']
                else:
                    row['u1'] = bestfit_params_wl['u'][0]
                    row['u2'] = bestfit_params_wl['u'][1]
                    if 'u_err_low' in bestfit_params_wl:
                        row['u1_err_low'] = bestfit_params_wl['u_err_low'][0]
                        row['u1_err_high'] = bestfit_params_wl['u_err_high'][0]
                        row['u2_err_low'] = bestfit_params_wl['u_err_low'][1]
                        row['u2_err_high'] = bestfit_params_wl['u_err_high'][1]
                
                add_scalar_param('v2')
                add_scalar_param('v3')
                add_scalar_param('v4')
                add_scalar_param('A')
                add_scalar_param('tau')
                add_scalar_param('spot_amp')
                add_scalar_param('spot_mu')
                add_scalar_param('spot_sigma')
                add_scalar_param('spot_amp2')
                add_scalar_param('spot_mu2')
                add_scalar_param('spot_sigma2')
                add_scalar_param('t_jump')
                add_scalar_param('jump')
                add_scalar_param('GP_log_sigma')
                add_scalar_param('GP_log_rho')
                
                rows.append(row)
        
            df = pd.DataFrame(rows)
            df.to_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv', index=False)
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')

            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
        else:
            print(f'GP trends already exist...')
            wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
            bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
            DURATION_BASE = bestfit_params_wl_df['duration'].values
            T0_BASE = bestfit_params_wl_df['t0'].values
            B_BASE = bestfit_params_wl_df['b'].values
            RORS_BASE = bestfit_params_wl_df['rors'].values
            DEPTH_BASE = RORS_BASE**2
    else:
        print(f'Whitelight outliers and bestfit parameters already exist...')
        wl_mad_mask = np.load(f'{output_dir}/{instrument_full_str}_whitelight_outlier_mask.npy')
        bestfit_params_wl_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_bestfit_params.csv')
        DURATION_BASE = bestfit_params_wl_df['duration'].values
        T0_BASE = bestfit_params_wl_df['t0'].values
        B_BASE = bestfit_params_wl_df['b'].values
        RORS_BASE = bestfit_params_wl_df['rors'].values
        DEPTH_BASE = RORS_BASE**2

    spec_good_mask = (~wl_mad_mask if len(wl_mad_mask) == len(data.time)
                      else np.ones(len(data.time), dtype=bool))
    wl_time_good = data.wl_time[~wl_mad_mask] if len(wl_mad_mask) == len(data.wl_time) else data.wl_time

    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)
    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres
    
    trend_fixed_hr = None
    ld_fixed_hr = None
    best_poly_coeffs_c, best_poly_coeffs_v = None, None
    best_poly_coeffs_u1, best_poly_coeffs_u2 = None, None
    spot_trend, spot_trend2, jump_trend = None, None, None
    if 'spot' in detrending_type and '2spot' not in detrending_type:
        if {'spot_amp', 'spot_mu', 'spot_sigma'}.issubset(bestfit_params_wl_df.columns):
            spot_amp = bestfit_params_wl_df['spot_amp'].values[0]
            spot_mu = bestfit_params_wl_df['spot_mu'].values[0]
            spot_sigma = bestfit_params_wl_df['spot_sigma'].values[0]
            if not np.isnan(spot_amp) and not np.isnan(spot_mu) and not np.isnan(spot_sigma):
                spot_trend = spot_crossing(wl_time_good, spot_amp, spot_mu, np.abs(spot_sigma))
    if '2spot' in detrending_type:
        if {'spot_amp', 'spot_mu', 'spot_sigma'}.issubset(bestfit_params_wl_df.columns):
            spot_amp = bestfit_params_wl_df['spot_amp'].values[0]
            spot_mu = bestfit_params_wl_df['spot_mu'].values[0]
            spot_sigma = bestfit_params_wl_df['spot_sigma'].values[0]
            if not np.isnan(spot_amp) and not np.isnan(spot_mu) and not np.isnan(spot_sigma):
                spot_trend = spot_crossing(wl_time_good, spot_amp, spot_mu, np.abs(spot_sigma))
        if {'spot_amp2', 'spot_mu2', 'spot_sigma2'}.issubset(bestfit_params_wl_df.columns):
            spot_amp2 = bestfit_params_wl_df['spot_amp2'].values[0]
            spot_mu2 = bestfit_params_wl_df['spot_mu2'].values[0]
            spot_sigma2 = bestfit_params_wl_df['spot_sigma2'].values[0]
            if not np.isnan(spot_amp2) and not np.isnan(spot_mu2) and not np.isnan(spot_sigma2):
                spot_trend2 = spot_crossing(wl_time_good, spot_amp2, spot_mu2, np.abs(spot_sigma2))
    if 'linear_discontinuity' in detrending_type:
        if {'t_jump', 'jump'}.issubset(bestfit_params_wl_df.columns):
            t_jump = bestfit_params_wl_df['t_jump'].values[0]
            jump = bestfit_params_wl_df['jump'].values[0]
            if not np.isnan(t_jump) and not np.isnan(jump):
                jump_trend = jump * _soft_step_np(np.array(wl_time_good), t_jump)
    valid = None
    lr_mask_path = f"{output_dir}/{instrument_full_str}_{lr_bin_str}_spectroscopic_outlier_mask.npy"
    lr_params_path = f"{output_dir}/{instrument_full_str}_{lr_bin_str}_bestfit_params.csv"
    poly_coeffs_path = f"{output_dir}/{instrument_full_str}_{lr_bin_str}_poly_coeffs.npz"
    if os.path.exists(lr_mask_path) and os.path.exists(lr_params_path):
        if interpolate_trend or interpolate_ld:
            if os.path.exists(poly_coeffs_path):
                print(f"Reusing low-res results and polynomial fits from {lr_bin_str}; skipping low-res fit.")
                time_mask = np.load(lr_mask_path)
                valid = ~time_mask
                poly_data = np.load(poly_coeffs_path)
                if interpolate_trend:
                    best_poly_coeffs_c = poly_data["coeffs_c"]
                    if "coeffs_v" in poly_data:
                        best_poly_coeffs_v = poly_data["coeffs_v"]
                if interpolate_ld:
                    best_poly_coeffs_u1 = poly_data["coeffs_u1"]
                    best_poly_coeffs_u2 = poly_data["coeffs_u2"]
            else:
                print("Low-res files found, but polynomial fits are missing; running low-res to build polynomials.")
        else:
            print(f"Reusing low-res results from {lr_bin_str}; skipping low-res fit.")
            time_mask = np.load(lr_mask_path)
            valid = ~time_mask

    if need_lowres_analysis and valid is None:
        print(f"\n--- Running Low-Resolution Analysis (Binned to {lr_bin_str}) ---")
        time_lr = jnp.array(data.time[spec_good_mask])
        flux_lr = jnp.array(data.flux_lr[:, spec_good_mask])
        flux_err_lr = jnp.array(data.flux_err_lr[:, spec_good_mask])
        num_lcs_lr = jnp.array(data.flux_err_lr.shape[0])

        if 'gp' in detrending_type:
            gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
            gp_trend_raw = gp_df['gp_trend'].values
            if len(gp_trend_raw) == len(wl_mad_mask):
                gp_trend_raw = gp_trend_raw[~wl_mad_mask]
            gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_lr)))
        else:
            gp_trend = None
        detrend_type_multiwave = _spectro_detrend_type(detrending_type)

        print(f"Low-res: {num_lcs_lr} light curves.")
        DEPTHS_BASE_LR = jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))

        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS']:
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr , instrument, ld_profile=ld_profile)
        elif instrument == 'NIRISS/SOSS':
            U_mu_lr = get_limb_darkening(sld, data.wavelengths_lr, data.wavelengths_err_lr, instrument, order=order, ld_profile=ld_profile)

        init_params_lr = {
            "u": U_mu_lr,
            "depths": jnp.tile(DEPTH_BASE, (num_lcs_lr, 1))
        }
        if detrend_type_multiwave != 'none':
            init_params_lr['c'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['c'].values[0])
            if 'v' in bestfit_params_wl_df.columns:
                 init_params_lr['v'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['v'].values[0])

        if 'explinear' in detrend_type_multiwave:
            init_params_lr['A'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['A'].values[0])
            init_params_lr['tau'] = jnp.full(num_lcs_lr, bestfit_params_wl_df['tau'].values[0])
        
        lr_trend_mode = 'free'
        lr_ld_mode = 'fixed' if flags.get('fix_ld', False) else 'free'
        
        lr_model_for_run = create_vectorized_model(
            detrend_type=detrend_type_multiwave,
            ld_mode=lr_ld_mode,
            trend_mode=lr_trend_mode,
            n_planets=n_planets,
            ld_profile=ld_profile
        )

        model_run_args_lr = {
            'mu_duration': DURATION_BASE,
            'mu_t0': T0_BASE,
            'mu_b': B_BASE,
            'mu_depths': DEPTHS_BASE_LR,
            'PERIOD': PERIOD_FIXED,
        }

        if lr_ld_mode == 'fixed': model_run_args_lr['ld_fixed'] = U_mu_lr
        if lr_ld_mode == 'free': model_run_args_lr['mu_u_ld'] = U_mu_lr
        
        if 'gp_spectroscopic' in detrend_type_multiwave:
            model_run_args_lr['gp_trend'] = gp_trend
            init_params_lr['A_gp'] = jnp.ones(num_lcs_lr)
        if detrend_type_multiwave == '2spot_spectroscopic':
            if spot_trend is None or spot_trend2 is None:
                raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
            if len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            if len(spot_trend2) != len(time_lr):
                spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            model_run_args_lr['spot_trend2'] = spot_trend2
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
            init_params_lr['A_spot2'] = jnp.ones(num_lcs_lr)
        elif detrend_type_multiwave == 'spot_spectroscopic':
            if spot_trend is not None and len(spot_trend) != len(time_lr):
                spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['spot_trend'] = spot_trend
            init_params_lr['A_spot'] = jnp.ones(num_lcs_lr)
        if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            if jump_trend is None:
                raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
            if jump_trend is not None and len(jump_trend) != len(time_lr):
                jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_lr))
            model_run_args_lr['jump_trend'] = jump_trend
            init_params_lr['A_jump'] = jnp.ones(num_lcs_lr)

        samples_lr = get_samples(lr_model_for_run, key_mcmc_lr, time_lr, flux_err_lr, flux_lr, init_params_lr, **model_run_args_lr)

        if 'u' in samples_lr:
            ld_u_lr = np.array(samples_lr["u"])
        elif ld_profile == 'power2' and 'c1' in samples_lr:
            c1_med_lr = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med_lr = jnp.nanmedian(samples_lr['c2'], axis=0)
            ld_u_lr = jax.vmap(compute_u_from_c)(c1_med_lr, c2_med_lr)
            ld_u_lr = np.array(ld_u_lr)

        if detrend_type_multiwave != 'none':
            trend_c_lr = np.array(samples_lr["c"])
            if 'v' in samples_lr: trend_v_lr = np.array(samples_lr["v"])
        if 'explinear' in detrend_type_multiwave:
            trend_A_lr = np.array(samples_lr["A"])
            trend_tau_lr = np.array(samples_lr["tau"])

        map_params_lr = {
            "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
            "rors": jnp.nanmedian(samples_lr["rors"], axis=0), 
            "period": PERIOD_FIXED,
        }
        
        if ld_profile == 'power2':
            c1_med = jnp.nanmedian(samples_lr['c1'], axis=0)
            c2_med = jnp.nanmedian(samples_lr['c2'], axis=0)
            map_params_lr['u'] = jax.vmap(compute_u_from_c)(c1_med, c2_med)
        else:
            map_params_lr['u'] = jnp.nanmedian(ld_u_lr, axis=0)

        map_params_lr.update({k: jnp.nanmedian(samples_lr[k], axis=0) for k in TREND_PARAMS if k in samples_lr})

        selected_kernel = COMPUTE_KERNELS[detrend_type_multiwave]
        in_axes_map = {'rors': 0, 'u': 0}
        in_axes_map.update({k: 0 for k in TREND_PARAMS if k in map_params_lr})
        
        final_in_axes = {k: in_axes_map.get(k, None) for k in map_params_lr.keys()}

        if 'gp_spectroscopic' in detrend_type_multiwave:
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None))(map_params_lr, time_lr, gp_trend)
        elif detrend_type_multiwave == '2spot_spectroscopic':
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None, None))(map_params_lr, time_lr, spot_trend, spot_trend2)
        elif detrend_type_multiwave == 'spot_spectroscopic':
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None))(map_params_lr, time_lr, spot_trend)
        elif 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None, None))(map_params_lr, time_lr, jump_trend)
        else:
            model_all = jax.vmap(selected_kernel, in_axes=(final_in_axes, None))(map_params_lr, time_lr)

        residuals = flux_lr - model_all
        plot_noise_binning(residuals, f"{output_dir}/25_{instrument_full_str}_{lr_bin_str}_noisebin.png")

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)
        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas
        time_mask = np.any(point_mask, axis=0)
        valid = ~time_mask
        np.save(lr_mask_path, time_mask)
        gp_trend_lr = gp_trend
        spot_trend_lr = spot_trend
        spot_trend2_lr = spot_trend2
        jump_trend_lr = jump_trend
        time_lr = time_lr[valid]
        flux_lr = flux_lr[:, valid]
        flux_err_lr = flux_err_lr[:, valid]
        if gp_trend_lr is not None: gp_trend_lr = gp_trend_lr[valid]
        if spot_trend_lr is not None: spot_trend_lr = spot_trend_lr[valid]
        if spot_trend2_lr is not None: spot_trend2_lr = spot_trend2_lr[valid]
        if jump_trend_lr is not None: jump_trend_lr = jump_trend_lr[valid]
        
        print("Plotting low-resolution fits and residuals...")
        median_total_error_lr = np.nanmedian(samples_lr['total_error'], axis=0)
        plot_wavelength_offset_summary(time_lr, flux_lr, median_total_error_lr, data.wavelengths_lr,
                                     map_params_lr, {"period": PERIOD_FIXED},
                                     f"{output_dir}/22_{instrument_full_str}_{lr_bin_str}_summary.png",
                                     detrend_type=detrend_type_multiwave, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, spot_trend2=spot_trend2_lr, jump_trend=jump_trend_lr)

        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(data.wavelengths_lr)

        if detrending_type != 'none':
            print("Fitting polynomials to trend coefficients...")
            best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
            if 'v' in samples_lr:
                best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
                plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v, "Wavelength", "v", "Trend Slope v", f"{output_dir}/2opt_v.png")

        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            if ld_profile == 'power2':
                c1_lr = np.array(samples_lr['c1'])
                c2_lr = np.array(samples_lr['c2'])
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, c1_lr, poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, c2_lr, poly_orders)
                plot_poly_fit(wl_lr, c1_lr, best_poly_coeffs_u1, best_order_u1, "Wavelength", "c1", "Limb Darkening c1", f"{output_dir}/2opt_c1.png")
                plot_poly_fit(wl_lr, c2_lr, best_poly_coeffs_u2, best_order_u2, "Wavelength", "c2", "Limb Darkening c2", f"{output_dir}/2opt_c2.png")
            else:
                best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
                best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 0], best_poly_coeffs_u1, best_order_u1, "Wavelength", "u1", "Limb Darkening u1", f"{output_dir}/2opt_u1.png")
                plot_poly_fit(wl_lr, ld_u_lr[:, :, 1], best_poly_coeffs_u2, best_order_u2, "Wavelength", "u2", "Limb Darkening u2", f"{output_dir}/2opt_u2.png")

        poly_save = {}
        if best_poly_coeffs_c is not None:
            poly_save["coeffs_c"] = best_poly_coeffs_c
        if best_poly_coeffs_v is not None:
            poly_save["coeffs_v"] = best_poly_coeffs_v
        if best_poly_coeffs_u1 is not None:
            poly_save["coeffs_u1"] = best_poly_coeffs_u1
        if best_poly_coeffs_u2 is not None:
            poly_save["coeffs_u2"] = best_poly_coeffs_u2
        if poly_save:
            np.savez(poly_coeffs_path, **poly_save)

        plot_transmission_spectrum(wl_lr, samples_lr["rors"], f"{output_dir}/24_{instrument_full_str}_{lr_bin_str}_spectrum")
        save_results(wl_lr, data.wavelengths_err_lr, samples_lr, f"{output_dir}/{instrument_full_str}_{lr_bin_str}.csv")
        save_detailed_fit_results(time_lr, flux_lr, flux_err_lr, data.wavelengths_lr, data.wavelengths_err_lr, samples_lr, map_params_lr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{instrument_full_str}_{lr_bin_str}", median_total_error_lr, gp_trend=gp_trend_lr, spot_trend=spot_trend_lr, jump_trend=jump_trend_lr)

    print(f"\n--- Running High-Resolution Analysis (Binned to {hr_bin_str}) ---")
    time_hr = jnp.array(data.time[spec_good_mask])
    flux_hr = jnp.array(data.flux_hr[:, spec_good_mask])
    flux_err_hr = jnp.array(data.flux_err_hr[:, spec_good_mask])

    if 'gp' in detrending_type:
        gp_df = pd.read_csv(f'{output_dir}/{instrument_full_str}_whitelight_GP_database.csv')
        gp_trend_raw = gp_df['gp_trend'].values
        if len(gp_trend_raw) == len(wl_mad_mask):
            gp_trend_raw = gp_trend_raw[~wl_mad_mask]
        gp_trend = jnp.array(_align_trend_to_time(gp_trend_raw, wl_time_good, np.array(time_hr)))
    else:
        gp_trend = None
    detrend_type_multiwave = _spectro_detrend_type(detrending_type)

    if valid is not None:
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if spot_trend2 is not None and len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        if gp_trend is not None and len(gp_trend) != len(time_hr):
            gp_trend = _align_trend_to_time(gp_trend, wl_time_good, np.array(time_hr))

        time_hr = time_hr[valid]
        flux_hr = flux_hr[:, valid]
        flux_err_hr = flux_err_hr[:, valid]
        if gp_trend is not None: gp_trend = gp_trend[valid]
        if spot_trend is not None: spot_trend = spot_trend[valid]
        if spot_trend2 is not None: spot_trend2 = spot_trend2[valid]
        if jump_trend is not None: jump_trend = jump_trend[valid]

    num_lcs_hr = flux_err_hr.shape[0]
    DEPTHS_BASE_HR = jnp.tile(DEPTH_BASE, (num_lcs_hr, 1))
    
    hr_ld_mode = 'free'
    if flags.get('interpolate_ld', False): hr_ld_mode = 'interpolated'
    elif flags.get('fix_ld', False): hr_ld_mode = 'fixed'
    hr_trend_mode = 'fixed' if flags.get('interpolate_trend', False) else 'free'

    model_run_args_hr = {}
    wl_hr = np.array(data.wavelengths_hr)

    if hr_ld_mode == 'interpolated':
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        
        if ld_profile == 'power2':
            ld_interpolated_hr = jax.vmap(compute_u_from_c)(jnp.array(u1_interp_hr), jnp.array(u2_interp_hr))
        else:
            ld_interpolated_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
            
        model_run_args_hr['ld_interpolated'] = ld_interpolated_hr
    elif hr_ld_mode == 'fixed' or hr_ld_mode == 'free':
        if instrument in ['NIRSPEC/G395H', 'NIRSPEC/G395M', 'NIRSPEC/PRISM', 'NIRSPEC/G140H', 'NIRSPEC/G235H', 'MIRI/LRS']:
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, ld_profile=ld_profile)
        elif instrument == 'NIRISS/SOSS':
            U_mu_hr_init = get_limb_darkening(sld, wl_hr, data.wavelengths_err_hr, instrument, order=order, ld_profile=ld_profile)
        if hr_ld_mode == 'fixed': model_run_args_hr['ld_fixed'] = U_mu_hr_init
        else: model_run_args_hr['mu_u_ld'] = U_mu_hr_init

    if hr_trend_mode == 'fixed':
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = np.column_stack((c_interp_hr, v_interp_hr))
        model_run_args_hr['trend_fixed'] = jnp.array(trend_fixed_hr)

    model_run_args_hr['mu_duration'] = DURATION_BASE
    model_run_args_hr['mu_t0'] = T0_BASE
    model_run_args_hr['mu_b'] = B_BASE
    model_run_args_hr['mu_depths'] = DEPTHS_BASE_HR
    model_run_args_hr['PERIOD'] = PERIOD_FIXED

    init_params_hr = { "depths": DEPTHS_BASE_HR, "u": U_mu_hr_init if hr_ld_mode!='interpolated' else ld_interpolated_hr }
    if hr_trend_mode == 'free':
        if detrend_type_multiwave != 'none':
            if best_poly_coeffs_c is not None and np.ndim(best_poly_coeffs_c) > 0:
                init_params_hr["c"] = np.polyval(best_poly_coeffs_c, wl_hr)
            else:
                init_params_hr["c"] = jnp.full(num_lcs_hr, bestfit_params_wl_df['c'].values[0])
            if 'v' in bestfit_params_wl_df.columns:
                 if best_poly_coeffs_v is not None and np.ndim(best_poly_coeffs_v) > 0:
                     init_params_hr["v"] = np.polyval(best_poly_coeffs_v, wl_hr)
                 else:
                     init_params_hr["v"] = jnp.full(num_lcs_hr, bestfit_params_wl_df['v'].values[0])

    hr_model_for_run = create_vectorized_model(
        detrend_type=detrend_type_multiwave,
        ld_mode=hr_ld_mode,
        trend_mode=hr_trend_mode,
        n_planets=n_planets,
        ld_profile=ld_profile
    )
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        model_run_args_hr['gp_trend'] = gp_trend
        init_params_hr['A_gp'] = jnp.ones(num_lcs_hr)
    if detrend_type_multiwave == '2spot_spectroscopic':
        if spot_trend is None or spot_trend2 is None:
            raise ValueError("2spot_spectroscopic requires WL spot and spot2 trends.")
        if len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        if len(spot_trend2) != len(time_hr):
            spot_trend2 = _align_trend_to_time(spot_trend2, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        model_run_args_hr['spot_trend2'] = spot_trend2
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
        init_params_hr['A_spot2'] = jnp.ones(num_lcs_hr)
    elif detrend_type_multiwave == 'spot_spectroscopic':
        if spot_trend is not None and len(spot_trend) != len(time_hr):
            spot_trend = _align_trend_to_time(spot_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['spot_trend'] = spot_trend
        init_params_hr['A_spot'] = jnp.ones(num_lcs_hr)
    if 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
        if jump_trend is None:
            raise ValueError("linear_discontinuity_spectroscopic requires WL linear_discontinuity to build jump_trend.")
        if jump_trend is not None and len(jump_trend) != len(time_hr):
            jump_trend = _align_trend_to_time(jump_trend, wl_time_good, np.array(time_hr))
        model_run_args_hr['jump_trend'] = jump_trend
        init_params_hr['A_jump'] = jnp.ones(num_lcs_hr)

    if vmap_chunk_size is None:
        samples_hr = get_samples(hr_model_for_run, key_mcmc_hr, time_hr, flux_err_hr, flux_hr, init_params_hr, **model_run_args_hr)
    else:
        checkpoint_prefix = f"{instrument_full_str}_{hr_bin_str}"
        samples_hr = get_samples_chunked(
            hr_model_for_run,
            key_mcmc_hr,
            time_hr,
            flux_err_hr,
            flux_hr,
            init_params_hr,
            vmap_chunk_size,
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            **model_run_args_hr
        )

    map_params_hr = {
        "duration": DURATION_BASE, "t0": T0_BASE, "b": B_BASE,
        "rors": jnp.nanmedian(samples_hr["rors"], axis=0), 
        "period": PERIOD_FIXED
    }
    if "u" in samples_hr and ld_profile != 'power2': 
        map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    elif ld_profile == 'power2':
        if 'c1' in samples_hr:
            c1_med_hr = jnp.nanmedian(samples_hr['c1'], axis=0)
            c2_med_hr = jnp.nanmedian(samples_hr['c2'], axis=0)
            map_params_hr['u'] = jax.vmap(compute_u_from_c)(c1_med_hr, c2_med_hr)
        else:
             if 'u' in samples_hr:
                 map_params_hr["u"] = jnp.nanmedian(np.array(samples_hr["u"]), axis=0)
    map_params_hr.update({k: jnp.nanmedian(samples_hr[k], axis=0) for k in TREND_PARAMS if k in samples_hr})

    in_axes_map_hr = {"rors": 0, "u": 0}
    in_axes_map_hr.update({k: 0 for k in TREND_PARAMS if k in map_params_hr})
    
    final_in_axes_hr = {k: in_axes_map_hr.get(k, None) for k in map_params_hr.keys()}
    selected_kernel_hr = COMPUTE_KERNELS[detrend_type_multiwave]
    
    if 'gp_spectroscopic' in detrend_type_multiwave:
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None))(map_params_hr, time_hr, gp_trend)
    elif detrend_type_multiwave == '2spot_spectroscopic':
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None, None))(map_params_hr, time_hr, spot_trend, spot_trend2)
    elif detrend_type_multiwave == 'spot_spectroscopic':
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None))(map_params_hr, time_hr, spot_trend)
    elif 'linear_discontinuity_spectroscopic' in detrend_type_multiwave:
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None, None))(map_params_hr, time_hr, jump_trend)
    else:
        model_all_hr = jax.vmap(selected_kernel_hr, in_axes=(final_in_axes_hr, None))(map_params_hr, time_hr)

    residuals_hr = np.array(flux_hr - model_all_hr)
    plot_noise_binning(residuals_hr, f"{output_dir}/36_{instrument_full_str}_{hr_bin_str}_noisebin.png")

    median_total_error_hr = np.nanmedian(samples_hr['total_error'], axis=0)
    plot_wavelength_offset_summary(time_hr, flux_hr, median_total_error_hr, data.wavelengths_hr,
                                    map_params_hr, {"period": PERIOD_FIXED},
                                    f"{output_dir}/34_{instrument_full_str}_{hr_bin_str}_summary.png",
                                    detrend_type=detrend_type_multiwave, gp_trend=gp_trend, spot_trend=spot_trend, spot_trend2=spot_trend2, jump_trend=jump_trend)

    plot_transmission_spectrum(wl_hr, samples_hr["rors"], f"{output_dir}/31_{instrument_full_str}_{hr_bin_str}_spectrum")
    save_results(wl_hr, data.wavelengths_err_hr, samples_hr,  f"{output_dir}/{instrument_full_str}_{hr_bin_str}.csv")
    save_detailed_fit_results(time_hr, flux_hr, flux_err_hr, data.wavelengths_hr, data.wavelengths_err_hr, samples_hr, map_params_hr, {"period": PERIOD_FIXED}, detrend_type_multiwave, f"{output_dir}/{instrument_full_str}_{hr_bin_str}", median_total_error_hr, gp_trend=gp_trend, spot_trend=spot_trend, jump_trend=jump_trend)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
