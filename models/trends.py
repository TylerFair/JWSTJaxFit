import jax.numpy as jnp
from .core import compute_transit_model

def _trend_time(t):
    return t - jnp.min(t)

_JUMP_WIDTH_DAYS = 1e-4

def _soft_step(t, t_jump, width=_JUMP_WIDTH_DAYS):
    return 0.5 * (1.0 + jnp.tanh((t - t_jump) / width))

def _poly_trend(params, t_norm, order):
    trend = params["c"] + params["v"] * t_norm
    if order >= 2:
        trend = trend + params["v2"] * t_norm**2
    if order >= 3:
        trend = trend + params["v3"] * t_norm**3
    if order >= 4:
        trend = trend + params["v4"] * t_norm**4
    return trend

def spot_crossing(t, amp, mu, sigma):
    sigma = jnp.maximum(jnp.abs(sigma), 1e-6)
    return amp * jnp.exp(-0.5 * (t - mu) ** 2 / sigma ** 2)

def compute_lc_none(params, t):
    return compute_transit_model(params, t) + 1.0

def compute_lc_linear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    trend = _poly_trend(params, t_norm, order=1)
    return lc_transit + trend

def compute_lc_quadratic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    trend = _poly_trend(params, t_norm, order=2)
    return lc_transit + trend

def compute_lc_cubic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    trend = _poly_trend(params, t_norm, order=3)
    return lc_transit + trend

def compute_lc_quartic(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    trend = _poly_trend(params, t_norm, order=4)
    return lc_transit + trend

def compute_lc_linear_discontinuity(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    jump = params["jump"] * _soft_step(t, params["t_jump"])
    trend = _poly_trend(params, t_norm, order=1) + jump
    return lc_transit + trend

def compute_lc_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    trend = _poly_trend(params, t_norm, order=1) + params["A"] * jnp.exp(-t_norm / params["tau"])
    return lc_transit + trend

def compute_lc_spot(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    trend = _poly_trend(params, t_norm, order=1)
    return lc_transit + trend + spot

def compute_lc_2spot(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    spot_1 = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    spot_2 = spot_crossing(t, params["spot_amp2"], params["spot_mu2"], params["spot_sigma2"])
    trend = _poly_trend(params, t_norm, order=1)
    return lc_transit + trend + spot_1 + spot_2

def compute_lc_spot_linear_discontinuity(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    jump = params["jump"] * _soft_step(t, params["t_jump"])
    trend = _poly_trend(params, t_norm, order=1) + spot + jump
    return lc_transit + trend

def compute_lc_spot_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    spot = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    trend = _poly_trend(params, t_norm, order=1) + spot + explinear
    return lc_transit + trend

def compute_lc_2spot_explinear(params, t):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    spot_1 = spot_crossing(t, params["spot_amp"], params["spot_mu"], params["spot_sigma"])
    spot_2 = spot_crossing(t, params["spot_amp2"], params["spot_mu2"], params["spot_sigma2"])
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    trend = _poly_trend(params, t_norm, order=1) + spot_1 + spot_2 + explinear
    return lc_transit + trend

def compute_lc_spot_spectroscopic(params, t, spot_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_spot"] * spot_trend

def compute_lc_2spot_spectroscopic(params, t, spot_trend, spot_trend2):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_spot"] * spot_trend + params["A_spot2"] * spot_trend2

def compute_lc_linear_discontinuity_spectroscopic(params, t, jump_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_jump"] * jump_trend

def compute_lc_spot_linear_discontinuity_spectroscopic(params, t, spot_trend, jump_trend):
    lc_transit = compute_transit_model(params, t)
    return lc_transit + params["c"] + params["A_spot"] * spot_trend + params["A_jump"] * jump_trend

def compute_lc_spot_explinear_spectroscopic(params, t, spot_trend):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    return lc_transit + params["c"] + params["v"] * t_norm + explinear + params["A_spot"] * spot_trend

def compute_lc_2spot_explinear_spectroscopic(params, t, spot_trend, spot_trend2):
    t_norm = _trend_time(t)
    lc_transit = compute_transit_model(params, t)
    explinear = params["A"] * jnp.exp(-t_norm / params["tau"])
    return (
        lc_transit
        + params["c"]
        + params["v"] * t_norm
        + explinear
        + params["A_spot"] * spot_trend
        + params["A_spot2"] * spot_trend2
    )
