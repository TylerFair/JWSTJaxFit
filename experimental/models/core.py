import jax
import jax.numpy as jnp
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from harmonica.jax import harmonica_transit_power2_ld
import numpy as np

def _to_f64(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)) and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating):
        return jnp.asarray(x, jnp.float64)
    return x

def _tree_to_f64(tree):
    return jax.tree_util.tree_map(_to_f64, tree)


def _broadcast_planet_param(param, n_planets, name):
    arr = jnp.atleast_1d(jnp.asarray(param, dtype=jnp.float64))
    if arr.size == 1 and n_planets > 1:
        return jnp.repeat(arr, n_planets)
    if arr.size != n_planets:
        raise ValueError(
            f"`{name}` must be scalar or length {n_planets}, got shape {arr.shape}."
        )
    return arr


def harmonica_cos_i_from_geometry(b, a_rs, ecc=0.0, omega=0.0):
    fac = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.sin(omega))
    return b / (a_rs * fac)


def harmonica_geometry_is_valid(b, a_rs, ecc=0.0, omega=0.0, tol=1e-10):
    cos_i = harmonica_cos_i_from_geometry(b, a_rs, ecc=ecc, omega=omega)
    return (
        jnp.isfinite(cos_i)
        & jnp.isfinite(a_rs)
        & jnp.isfinite(ecc)
        & jnp.isfinite(omega)
        & (a_rs > 0.0)
        & (ecc >= 0.0)
        & (ecc < 1.0)
        & (cos_i >= -1.0 - tol)
        & (cos_i <= 1.0 + tol)
    )


def harmonica_duration_from_geometry(period, a_rs, b, rors, ecc=0.0, omega=0.0):
    valid = harmonica_geometry_is_valid(b, a_rs, ecc=ecc, omega=omega)
    cos_i = harmonica_cos_i_from_geometry(b, a_rs, ecc=ecc, omega=omega)
    cos_i = jnp.where(valid, jnp.clip(cos_i, -1.0, 1.0), jnp.nan)
    sin_i = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_i**2))
    chord = jnp.sqrt(jnp.maximum(0.0, (1.0 + rors) ** 2 - b ** 2))
    speed_factor = (
        jnp.sqrt(jnp.maximum(0.0, 1.0 - ecc**2))
        / (1.0 + ecc * jnp.sin(omega))
    )
    denom = jnp.maximum(a_rs * sin_i, 1e-12)
    arg = jnp.clip(chord / denom, 0.0, 1.0)
    duration = (period / jnp.pi) * jnp.arcsin(arg) * speed_factor
    return jnp.where(valid, duration, jnp.nan)

def compute_transit_model(params, t):
    """
    Transit Model for one or more planets, using vmap for performance.
    Expects params to contain 'period', 'duration', 't0', 'b', 'rors', 'u'.
    These should be arrays where the 0-th dimension is the planet index,
    except 'u' which is limb darkening parameters.
    """
    periods = jnp.atleast_1d(params["period"])
    durations = jnp.atleast_1d(params["duration"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])

    def get_lc(period, duration, t0, b, rors):
        orbit = TransitOrbit(
            period=period,
            duration=duration,
            time_transit=t0,
            impact_param=b,
            radius_ratio=rors
        )
        return limb_dark_light_curve(orbit, params["u"])(t)

    batched_lcs = jax.vmap(get_lc)(periods, durations, t0s, bs, rorss)
    total_flux = jnp.sum(batched_lcs, axis=0)
    return total_flux


def compute_transit_model_harmonica(params, t):
    """
    Transit model using harmonica with native power-2 limb darkening
    and first-order transmission string r_p(theta) = a_0 + a_1*cos(theta).

    Returns the transit *signal* (0 out of transit, negative during transit)
    matching jaxoplanet convention used by the detrending kernels.

    Expects params to contain:
      'period', 't0', 'b', 'rors' - orbital
      'a_rs'    - semi-major axis in stellar radii
      'ecc'     - eccentricity (0 for circular)
      'omega'   - argument of periastron [radians]
      'a1'      - asymmetry coefficient (0 = symmetric)
      'c_ld'    - power-2 limb darkening c
      'alpha_ld'- power-2 limb darkening alpha

    Derives inclination from (b, a_rs, ecc, omega).
    """
    periods = jnp.atleast_1d(params["period"])
    t0s = jnp.atleast_1d(params["t0"])
    bs = jnp.atleast_1d(params["b"])
    rorss = jnp.atleast_1d(params["rors"])
    n_planets = periods.shape[0]
    a1s = _broadcast_planet_param(
        params.get("a1", jnp.zeros_like(rorss)), n_planets, "a1"
    )
    a_rss = _broadcast_planet_param(params["a_rs"], n_planets, "a_rs")
    eccs = _broadcast_planet_param(params.get("ecc", 0.0), n_planets, "ecc")
    omegas = _broadcast_planet_param(params.get("omega", 0.0), n_planets, "omega")
    c_ld = params["c_ld"]
    alpha_ld = params["alpha_ld"]

    def get_lc(period, t0, b, rors, a1, a_rs, ecc, omega):
        valid = harmonica_geometry_is_valid(b, a_rs, ecc=ecc, omega=omega)
        cos_i = harmonica_cos_i_from_geometry(b, a_rs, ecc=ecc, omega=omega)
        inc = jnp.arccos(jnp.clip(cos_i, -1.0, 1.0))

        def _valid_flux(_):
            r = jnp.array([rors, a1, 0.0])
            # harmonica returns normalised flux (1.0 out of transit).
            # Subtract 1.0 to match jaxoplanet signal convention.
            return harmonica_transit_power2_ld(
                t, t0, period, a_rs, inc, ecc, omega,
                c=c_ld, alpha=alpha_ld, r=r,
            ) - 1.0

        return jax.lax.cond(
            valid,
            _valid_flux,
            lambda _: jnp.full_like(t, jnp.nan, dtype=jnp.float64),
            operand=None,
        )

    batched_lcs = jax.vmap(get_lc)(
        periods, t0s, bs, rorss, a1s, a_rss, eccs, omegas)
    return jnp.sum(batched_lcs, axis=0)


def get_I_power2(c, alpha, u):
    return 1 - c*(1-jnp.power(u,alpha))
