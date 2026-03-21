import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

from .core import (
    get_I_power2,
    compute_transit_model_harmonica,
    harmonica_duration_from_geometry,
    harmonica_geometry_is_valid,
)
from .trends import (
    compute_lc_linear, compute_lc_quadratic, compute_lc_cubic, compute_lc_quartic,
    compute_lc_linear_discontinuity, compute_lc_explinear, compute_lc_spot, compute_lc_2spot,
    compute_lc_none, compute_lc_spot_spectroscopic, compute_lc_2spot_spectroscopic,
    compute_lc_linear_discontinuity_spectroscopic, compute_lc_spot_linear_discontinuity,
    compute_lc_spot_explinear, compute_lc_2spot_explinear,
    compute_lc_spot_linear_discontinuity_spectroscopic, compute_lc_spot_explinear_spectroscopic,
    compute_lc_2spot_explinear_spectroscopic
)
from .gp import (
    compute_lc_gp_mean, compute_lc_linear_gp_mean, compute_lc_quadratic_gp_mean,
    compute_lc_cubic_gp_mean, compute_lc_quartic_gp_mean, compute_lc_explinear_gp_mean,
    compute_lc_gp_spectroscopic, compute_lc_linear_gp_spectroscopic,
    compute_lc_quadratic_gp_spectroscopic, compute_lc_cubic_gp_spectroscopic,
    compute_lc_quartic_gp_spectroscopic, compute_lc_explinear_gp_spectroscopic,
    build_gp, build_gp_linear, build_gp_quadratic, build_gp_cubic, build_gp_quartic,
    build_gp_explinear
)

COMPUTE_KERNELS = {
    'linear': compute_lc_linear,
    'quadratic': compute_lc_quadratic,
    'cubic': compute_lc_cubic,
    'quartic': compute_lc_quartic,
    'linear_discontinuity': compute_lc_linear_discontinuity,
    'explinear': compute_lc_explinear,
    'spot': compute_lc_spot,
    '2spot': compute_lc_2spot,
    'gp': compute_lc_gp_mean,
    'none': compute_lc_none,
    'gp_spectroscopic': compute_lc_gp_spectroscopic,
    'linear+gp_spectroscopic': compute_lc_linear_gp_spectroscopic,
    'quadratic+gp_spectroscopic': compute_lc_quadratic_gp_spectroscopic,
    'cubic+gp_spectroscopic': compute_lc_cubic_gp_spectroscopic,
    'quartic+gp_spectroscopic': compute_lc_quartic_gp_spectroscopic,
    'explinear+gp_spectroscopic': compute_lc_explinear_gp_spectroscopic,
    'spot_spectroscopic': compute_lc_spot_spectroscopic,
    '2spot_spectroscopic': compute_lc_2spot_spectroscopic,
    'linear_discontinuity_spectroscopic': compute_lc_linear_discontinuity_spectroscopic,
    'spot+linear_discontinuity': compute_lc_spot_linear_discontinuity,
    'spot+explinear': compute_lc_spot_explinear,
    '2spot+explinear': compute_lc_2spot_explinear,
    'spot_spectroscopic+linear_discontinuity_spectroscopic': compute_lc_spot_linear_discontinuity_spectroscopic,
    'spot_spectroscopic+explinear': compute_lc_spot_explinear_spectroscopic,
    '2spot_spectroscopic+explinear': compute_lc_2spot_explinear_spectroscopic,
}

COMPOSITE_KERNELS = {
    frozenset({'spot', 'linear_discontinuity'}): compute_lc_spot_linear_discontinuity,
    frozenset({'spot', 'explinear'}): compute_lc_spot_explinear,
    frozenset({'2spot', 'explinear'}): compute_lc_2spot_explinear,
    frozenset({'spot_spectroscopic', 'linear_discontinuity_spectroscopic'}): compute_lc_spot_linear_discontinuity_spectroscopic,
    frozenset({'spot_spectroscopic', 'explinear'}): compute_lc_spot_explinear_spectroscopic,
    frozenset({'2spot_spectroscopic', 'explinear'}): compute_lc_2spot_explinear_spectroscopic,
}

def _split_components(detrend_type):
    return set(detrend_type.split('+'))

def resolve_detrend_kernel(detrend_type):
    if detrend_type in COMPUTE_KERNELS:
        return COMPUTE_KERNELS[detrend_type]
    detrend_components = frozenset(_split_components(detrend_type))
    if detrend_components in COMPOSITE_KERNELS:
        return COMPOSITE_KERNELS[detrend_components]
    raise KeyError(detrend_type)

def _prepare_power2_poly(degree=12, n_mu=300):
    mus = jnp.linspace(0.0, 1.0, n_mu, endpoint=True)
    x = jnp.vander(1.0 - mus, N=degree + 1, increasing=True)[:, 1:]
    p = jnp.asarray(np.linalg.pinv(np.asarray(x)))
    return mus, p
    
def create_whitelight_model(detrend_type='linear', n_planets=1, ld_profile='quadratic',
                            transit_engine='jaxoplanet'):
    print(f"Building whitelight model with: detrend_type='{detrend_type}', "
          f"engine='{transit_engine}' for {n_planets} planets")

    detrend_components = _split_components(detrend_type)
    if ld_profile == "power2" and transit_engine != 'harmonica':
        MUS, P = _prepare_power2_poly()

    def _whitelight_model_static(t, yerr, y=None, prior_params=None):
        durations, t0s, bs, rorss, a_rss = [], [], [], [], []

        def _prior_array(name, default):
            arr = jnp.atleast_1d(jnp.asarray(prior_params.get(name, default),
                                             dtype=jnp.float64))
            if arr.size == 1 and n_planets > 1:
                return jnp.repeat(arr, n_planets)
            if arr.size != n_planets:
                raise ValueError(
                    f"`{name}` must be scalar or length {n_planets}, got shape {arr.shape}."
                )
            return arr

        harmonica_ecc = _prior_array('ecc', 0.0)
        harmonica_omega = _prior_array('omega', 0.0)

        for i in range(n_planets):
            t0s.append(numpyro.sample(f"t0_{i}", dist.Uniform(jnp.min(t), jnp.max(t))))
            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))
            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-6, 0.5))
            rors_i = numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths))
            rorss.append(rors_i)

            if transit_engine == 'harmonica':
                log_a_rs = numpyro.sample(
                    f"log_a_rs_{i}",
                    dist.Uniform(jnp.log(2.0), jnp.log(100.0))
                )
                a_rs_i = numpyro.deterministic(f"a_rs_{i}", jnp.exp(log_a_rs))
                a_rss.append(a_rs_i)
                durations.append(
                    numpyro.deterministic(
                        f"duration_{i}",
                        harmonica_duration_from_geometry(
                            prior_params['period'][i],
                            a_rs_i,
                            bs[-1],
                            rors_i,
                            ecc=harmonica_ecc[i],
                            omega=harmonica_omega[i],
                        ),
                    )
                )
            else:
                logD = numpyro.sample(
                    f"logD_{i}", dist.Uniform(jnp.log(0.0007), jnp.log(1))
                )
                durations.append(
                    numpyro.deterministic(f"duration_{i}", jnp.exp(logD))
                )

        if transit_engine == 'harmonica':
            # Power-2 LD goes directly to harmonica (no polynomial conversion).
            c1 = numpyro.sample('c1', dist.TruncatedNormal(prior_params['u'][0], 0.2, low=0.0, high=1.0))
            c2 = numpyro.sample('c2', dist.TruncatedNormal(prior_params['u'][1], 0.2, low=0.001, high=1.0))
            # Asymmetry: a1*cos(theta) with sin zeroed out.
            a1 = numpyro.sample('a1', dist.Uniform(-0.02, 0.02))
        elif ld_profile == 'quadratic':
            u = numpyro.sample("u", dist.Uniform(0.0, 1.0).expand([2]).to_event(1))
        elif ld_profile == 'power2':
            c1 = numpyro.sample('c1', dist.TruncatedNormal(prior_params['u'][0], 0.2, low=0.0, high=1.0))
            c2 = numpyro.sample('c2', dist.TruncatedNormal(prior_params['u'][1], 0.2, low=0.001, high=1.0))

            prof = get_I_power2(c1, c2, MUS)
            u = P @ (1.0 - prof)
        else:
            raise ValueError(f"Unknown ld_profile: {ld_profile}")

        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-5), jnp.log(1e-2)))
        error = numpyro.deterministic('error', jnp.sqrt(jnp.exp(log_jitter)**2 + yerr**2))

        if transit_engine == 'harmonica':
            params = {
                "period": prior_params['period'],
                "t0": jnp.array(t0s), "b": jnp.array(bs), "rors": jnp.array(rorss),
                "a_rs": jnp.array(a_rss),
                "ecc": harmonica_ecc,
                "omega": harmonica_omega,
                "a1": jnp.array([a1] * n_planets),
                "c_ld": c1, "alpha_ld": c2,
            }
        else:
            params = {
                "period": prior_params['period'], "duration": jnp.array(durations),
                "t0": jnp.array(t0s), "b": jnp.array(bs), "rors": jnp.array(rorss),
                "u": u,
            }

        has_offset_term = not detrend_components.isdisjoint({'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot', 'gp'})
        if has_offset_term:
            params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1))

        has_linear = not detrend_components.isdisjoint({'linear', 'quadratic', 'cubic', 'quartic', 'linear_discontinuity', 'explinear', 'spot', '2spot'})
        if has_linear:
            params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'quadratic', 'cubic', 'quartic'}):
            params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1))

        if not detrend_components.isdisjoint({'cubic', 'quartic'}):
            params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1))

        if 'quartic' in detrend_components:
            params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1))

        if 'linear_discontinuity' in detrend_components:
            t_jump_guess = prior_params.get('t_jump_guess', 0.5 * (jnp.min(t) + jnp.max(t)))
            jump_guess = prior_params.get('jump_guess', 0.0)
            params['t_jump'] = numpyro.sample('t_jump', dist.Normal(t_jump_guess, 1e-2))
            params['jump'] = numpyro.sample('jump', dist.Normal(jump_guess, 0.01))

        if 'explinear' in detrend_components:
            params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1))
            log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)))
            params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))

        if not detrend_components.isdisjoint({'spot', '2spot'}):
            params['spot_amp'] = numpyro.sample('spot_amp', dist.Uniform(0.0, 0.1))
            params['spot_mu'] = numpyro.sample('spot_mu', dist.Normal(prior_params['spot_guess'], 0.01))
            params['spot_sigma'] = numpyro.sample('spot_sigma', dist.Uniform(1e-4, 0.1))
        if '2spot' in detrend_components:
            spot_guess2 = prior_params.get('spot_guess2', prior_params['spot_guess'])
            params['spot_amp2'] = numpyro.sample('spot_amp2', dist.Uniform(0.0, 0.1))
            params['spot_mu2'] = numpyro.sample('spot_mu2', dist.Normal(spot_guess2, 0.01))
            params['spot_sigma2'] = numpyro.sample('spot_sigma2', dist.Uniform(1e-4, 0.1))

        if 'gp' in detrend_components:
            params['GP_log_sigma'] = numpyro.sample('GP_log_sigma', dist.Uniform(jnp.log(1e-5), jnp.log(1e3)))
            params['GP_log_rho'] = numpyro.sample('GP_log_rho', dist.Uniform(jnp.log(0.007), jnp.log(0.3)))

        if transit_engine == 'harmonica':
            valid_geometry = harmonica_geometry_is_valid(
                params["b"], params["a_rs"], ecc=params["ecc"], omega=params["omega"]
            )
            numpyro.factor(
                "harmonica_valid_geometry",
                jnp.where(jnp.all(valid_geometry), 0.0, -jnp.inf),
            )
            # Compute transit + trend inline using harmonica engine.
            transit_signal = compute_transit_model_harmonica(params, t)
            t_norm = t - jnp.min(t)
            if detrend_type == 'none':
                lc_model = transit_signal + 1.0
            else:
                trend = params.get('c', 0.0) + params.get('v', 0.0) * t_norm
                if 'v2' in params:
                    trend = trend + params['v2'] * t_norm ** 2
                if 'v3' in params:
                    trend = trend + params['v3'] * t_norm ** 3
                if 'v4' in params:
                    trend = trend + params['v4'] * t_norm ** 4
                if 'A' in params:
                    trend = trend + params['A'] * jnp.exp(-t_norm / params['tau'])
                if 't_jump' in params:
                    trend = trend + params['jump'] * (
                        0.5 * (1.0 + jnp.tanh((t - params['t_jump']) / 1e-4)))
                lc_model = transit_signal + trend
            numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
        elif 'gp' in detrend_type:
            if detrend_type == 'gp':
                gp_builder = build_gp
            elif detrend_type == 'linear+gp':
                gp_builder = build_gp_linear
            elif detrend_type == 'quadratic+gp':
                gp_builder = build_gp_quadratic
            elif detrend_type == 'cubic+gp':
                gp_builder = build_gp_cubic
            elif detrend_type == 'quartic+gp':
                gp_builder = build_gp_quartic
            elif detrend_type == 'explinear+gp':
                gp_builder = build_gp_explinear
            else:
                raise ValueError(f"Unknown GP detrend_type: {detrend_type}")

            gp = gp_builder(params, t, error)
            numpyro.sample('obs', gp.numpyro_dist(), obs=y)
        else:
            try:
                lc_model = resolve_detrend_kernel(detrend_type)(params, t)
                numpyro.sample('obs', dist.Normal(lc_model, error), obs=y)
            except KeyError:
                raise ValueError(f"Unknown detrend_type: {detrend_type}")

    return _whitelight_model_static

def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free',
                            n_planets=1, ld_profile='quadratic', transit_engine='jaxoplanet'):
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', "
          f"trend='{trend_mode}', engine='{transit_engine}' for {n_planets} planets")

    detrend_components = _split_components(detrend_type)
    if transit_engine != 'harmonica':
        try:
            compute_lc_kernel = resolve_detrend_kernel(detrend_type)
        except KeyError:
            raise ValueError(f"Unsupported detrend_type for vectorized model: {detrend_type}")
        if detrend_components in ({"spot"}, {"2spot"}, {"linear_discontinuity"}):
            raise ValueError(f"Vectorized model does not support '{detrend_type}'. Use '{detrend_type}_spectroscopic' instead.")
    if ld_profile == "power2" and transit_engine != 'harmonica':
        MUS_LD, P_LD = _prepare_power2_poly()

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                               mu_depths=None, PERIOD=None, trend_fixed=None,
                               ld_interpolated=None, ld_fixed=None,
                               mu_spot_amp=None, mu_spot_mu=None, mu_spot_sigma=None,
                               harmonica_a_rs=None, harmonica_ecc=0.0, harmonica_omega=0.0,
                               mu_u_ld=None, gp_trend=None, spot_trend=None, spot_trend2=None, jump_trend=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        durations = mu_duration
        t0s = mu_t0
        bs = mu_b

        # Harmonica orbital params (used only when transit_engine='harmonica').
        _harmonica_a_rs = harmonica_a_rs if harmonica_a_rs is not None else 10.0
        _harmonica_ecc = harmonica_ecc
        _harmonica_omega = harmonica_omega

        depths = numpyro.sample('depths', dist.Uniform(1e-5, 0.5).expand([num_lcs, n_planets]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        yerr_per_lc = jnp.nanmedian(yerr, axis=1)
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter**2 + yerr_per_lc**2))
        error_broadcast = total_error[:, None] * jnp.ones_like(t)

        if transit_engine == 'harmonica':
            # Power-2 LD goes directly to harmonica — no polynomial conversion.
            if ld_mode == 'free':
                c1 = numpyro.sample('c1', dist.TruncatedNormal(mu_u_ld[:, 0], 0.2, low=0.0, high=1.0))
                c2 = numpyro.sample('c2', dist.TruncatedNormal(mu_u_ld[:, 1], 0.2, low=0.001, high=1.0))
            elif ld_mode == 'fixed':
                c1 = numpyro.deterministic('c1', ld_fixed[:, 0])
                c2 = numpyro.deterministic('c2', ld_fixed[:, 1])
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")
            # Per-wavelength asymmetry.
            a1 = numpyro.sample('a1', dist.Uniform(-0.02, 0.02).expand([num_lcs]))
        else:
            if ld_mode == 'free':
                if ld_profile == 'quadratic':
                    u = numpyro.sample('u', dist.TruncatedNormal(loc=mu_u_ld, scale=0.2, low=-1.0, high=1.0).to_event(1))
                elif ld_profile == 'power2':
                    c1 = numpyro.sample('c1', dist.TruncatedNormal(mu_u_ld[:,0], 0.2, low=0.0, high=1.0))
                    c2 = numpyro.sample('c2', dist.TruncatedNormal(mu_u_ld[:,1], 0.2, low=0.001, high=1.0))
                    profs = get_I_power2(c1[:, None], c2[:, None], MUS_LD[None, :])
                    u = (P_LD @ (1.0 - profs).T).T
                else:
                    raise ValueError(f"Unknown ld_profile: {ld_profile}")
            elif ld_mode == 'fixed':
                if ld_profile == 'quadratic':
                    u = numpyro.deterministic('u', ld_fixed)
                elif ld_profile == 'power2':
                    c1_mu = numpyro.deterministic('c1', ld_fixed[:, 0])
                    c2_mu = numpyro.deterministic('c2', ld_fixed[:, 1])
                    profs = get_I_power2(c1_mu[:, None], c2_mu[:, None], MUS_LD[None, :])
                    u = (P_LD @ (1.0 - profs).T).T
            else:
                raise ValueError(f"Unknown ld_mode: {ld_mode}")

        if transit_engine == 'harmonica':
            # Harmonica orbital params come from the model-level closure,
            # not from the jaxoplanet (duration, b) path.
            _harm_orbital = {
                "period": PERIOD, "t0": t0s, "b": bs,
                "a_rs": _harmonica_a_rs, "ecc": _harmonica_ecc,
                "omega": _harmonica_omega,
            }
        else:
            params = {
                "period": PERIOD, "duration": durations, "t0": t0s, "b": bs,
                "rors": rors, "u": u,
            }
            in_axes = {"period": None, "duration": None, "t0": None, "b": None,
                       "rors": 0, "u": 0}

        if transit_engine == 'harmonica':
            valid_geometry = harmonica_geometry_is_valid(
                bs, _harmonica_a_rs, ecc=_harmonica_ecc, omega=_harmonica_omega
            )
            numpyro.factor(
                "harmonica_vectorized_valid_geometry",
                jnp.where(jnp.all(valid_geometry), 0.0, -jnp.inf),
            )
            # ── harmonica vectorized: transit + trend inline ─────────
            c_trend = numpyro.sample('c', dist.Uniform(0.9, 1.1).expand([num_lcs]))
            v_trend = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))

            poly_order = 1
            if 'quartic' in detrend_type:
                poly_order = 4
            elif 'cubic' in detrend_type:
                poly_order = 3
            elif 'quadratic' in detrend_type:
                poly_order = 2

            v2_trend = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 2 else None
            v3_trend = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 3 else None
            v4_trend = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs])) if poly_order >= 4 else None

            has_explinear = 'explinear' in detrend_components
            if has_explinear:
                A_trend = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                tau_trend = numpyro.deterministic('tau', jnp.exp(log_tau))

            def _single_lc_harmonica(t_i, rors_i, a1_i, c1_i, c2_i,
                                     c_t, v_t, v2_t, v3_t, v4_t,
                                     A_t, tau_t):
                params_i = {**_harm_orbital,
                            "rors": rors_i, "a1": jnp.ones_like(rors_i) * a1_i,
                            "c_ld": c1_i, "alpha_ld": c2_i}
                transit_sig = compute_transit_model_harmonica(params_i, t_i)
                t_norm = t_i - jnp.min(t_i)
                trend = c_t + v_t * t_norm
                if poly_order >= 2:
                    trend = trend + v2_t * t_norm ** 2
                if poly_order >= 3:
                    trend = trend + v3_t * t_norm ** 3
                if poly_order >= 4:
                    trend = trend + v4_t * t_norm ** 4
                if has_explinear:
                    trend = trend + A_t * jnp.exp(-t_norm / tau_t)
                return transit_sig + trend

            _z = jnp.zeros(num_lcs)
            y_model = jax.vmap(
                _single_lc_harmonica,
                in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            )(
                t, rors, a1, c1, c2,
                c_trend, v_trend,
                v2_trend if v2_trend is not None else _z,
                v3_trend if v3_trend is not None else _z,
                v4_trend if v4_trend is not None else _z,
                A_trend if has_explinear else _z,
                tau_trend if has_explinear else jnp.ones(num_lcs),
            )
            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

        else:
            # ── jaxoplanet path: existing kernel dispatch ──
            if detrend_type != 'none':
                if trend_mode == 'free':
                    params['c'] = numpyro.sample('c', dist.Uniform(0.9, 1.1).expand([num_lcs]))
                    params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes.update({'c': 0, 'v': 0})

                    poly_order = 1
                    if 'quartic' in detrend_type:
                        poly_order = 4
                    elif 'cubic' in detrend_type:
                        poly_order = 3
                    elif 'quadratic' in detrend_type:
                        poly_order = 2

                    if poly_order >= 2:
                        params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                        in_axes.update({'v2': 0})
                    if poly_order >= 3:
                        params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                        in_axes.update({'v3': 0})
                    if poly_order >= 4:
                        params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                        in_axes.update({'v4': 0})

                    if 'explinear' in detrend_components:
                        params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                        log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                        params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                        in_axes.update({'A': 0, 'tau': 0})

                elif trend_mode == 'fixed':
                    trend_temp = numpyro.deterministic('trend_temp', trend_fixed)
                    params['c'] = numpyro.deterministic('c', trend_temp[:, 0])
                    params['v'] = numpyro.deterministic('v', trend_temp[:, 1])
                    in_axes.update({'c': 0, 'v': 0})

                    poly_order = 1
                    if 'quartic' in detrend_type:
                        poly_order = 4
                    elif 'cubic' in detrend_type:
                        poly_order = 3
                    elif 'quadratic' in detrend_type:
                        poly_order = 2

                    if poly_order >= 2:
                        params['v2'] = numpyro.deterministic('v2', trend_temp[:, 2])
                        in_axes.update({'v2': 0})
                    if poly_order >= 3:
                        params['v3'] = numpyro.deterministic('v3', trend_temp[:, 3])
                        in_axes.update({'v3': 0})
                    if poly_order >= 4:
                        params['v4'] = numpyro.deterministic('v4', trend_temp[:, 4])
                        in_axes.update({'v4': 0})
                    if 'linear_discontinuity' in detrend_components:
                        params['t_jump'] = numpyro.deterministic('t_jump', trend_temp[:, 2])
                        params['jump'] = numpyro.deterministic('jump', trend_temp[:, 3])
                        in_axes.update({'t_jump': 0, 'jump': 0})
                    elif 'explinear' in detrend_components:
                        params['A'] = numpyro.deterministic('A', trend_temp[:, 2])
                        params['tau'] = numpyro.deterministic('tau', trend_temp[:, 3])
                        in_axes.update({'A': 0, 'tau': 0})
                    elif 'spot' in detrend_components and '2spot' not in detrend_components:
                        params['spot_amp'] = numpyro.deterministic('spot_amp', trend_temp[:, 2])
                        params['spot_mu'] = numpyro.deterministic('spot_mu', trend_temp[:, 3])
                        params['spot_sigma'] = numpyro.deterministic('spot_sigma', trend_temp[:, 4])
                        in_axes.update({'spot_amp': 0, 'spot_mu': 0, 'spot_sigma': 0})
                else:
                    raise ValueError(f"Unknown trend_mode: {trend_mode}")

            if 'gp_spectroscopic' in detrend_components:
                params['A_gp'] = numpyro.sample('A_gp', dist.Uniform(0.5, 2).expand([num_lcs]))
                in_axes['A_gp'] = 0

                if 'linear' in detrend_components and 'v' not in params:
                    params['v'] = numpyro.sample('v', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v'] = 0

                poly_order = 1
                if 'quartic' in detrend_components:
                    poly_order = 4
                elif 'cubic' in detrend_components:
                    poly_order = 3
                elif 'quadratic' in detrend_components:
                    poly_order = 2

                if poly_order >= 2 and 'v2' not in params:
                    params['v2'] = numpyro.sample('v2', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v2'] = 0
                if poly_order >= 3 and 'v3' not in params:
                    params['v3'] = numpyro.sample('v3', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v3'] = 0
                if poly_order >= 4 and 'v4' not in params:
                    params['v4'] = numpyro.sample('v4', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    in_axes['v4'] = 0

                if 'explinear' in detrend_components and 'A' not in params:
                    params['A'] = numpyro.sample('A', dist.Uniform(-0.1, 0.1).expand([num_lcs]))
                    log_tau = numpyro.sample('log_tau', dist.Uniform(jnp.log(1e-3), jnp.log(1e-1)).expand([num_lcs]))
                    params['tau'] = numpyro.deterministic('tau', jnp.exp(log_tau))
                    in_axes.update({'A': 0, 'tau': 0})

                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, gp_trend)

            elif '2spot_spectroscopic' in detrend_components:
                params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
                params['A_spot2'] = numpyro.sample('A_spot2', dist.Uniform(0.5, 2).expand([num_lcs]))
                in_axes['A_spot'] = 0
                in_axes['A_spot2'] = 0
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, spot_trend2)
            elif 'spot_spectroscopic' in detrend_components:
                params['A_spot'] = numpyro.sample('A_spot', dist.Uniform(0.5, 2).expand([num_lcs]))
                in_axes['A_spot'] = 0
                if 'linear_discontinuity_spectroscopic' in detrend_components:
                    params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(0.5, 2).expand([num_lcs]))
                    in_axes['A_jump'] = 0
                    y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None, None))(params, t, spot_trend, jump_trend)
                else:
                    y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, spot_trend)
            elif 'linear_discontinuity_spectroscopic' in detrend_components:
                params['A_jump'] = numpyro.sample('A_jump', dist.Uniform(0.5, 2).expand([num_lcs]))
                in_axes['A_jump'] = 0
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None, None))(params, t, jump_trend)
            else:
                y_model = jax.vmap(compute_lc_kernel, in_axes=(in_axes, None))(params, t)

            numpyro.sample('obs', dist.Normal(y_model, error_broadcast), obs=y)

    return _vectorized_model_static
