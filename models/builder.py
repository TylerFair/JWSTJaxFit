import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

from .core import get_I_power2
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
    
def create_whitelight_model(detrend_type='linear', n_planets=1, ld_profile='quadratic'):
    print(f"Building whitelight model with: detrend_type='{detrend_type}' for {n_planets} planets")

    detrend_components = _split_components(detrend_type)
    if ld_profile == "power2":
        MUS, P = _prepare_power2_poly()

    def _whitelight_model_static(t, yerr, y=None, prior_params=None):
        durations, t0s, bs, rorss = [], [], [], []

        for i in range(n_planets):
            logD = numpyro.sample(f"logD_{i}", dist.Uniform(jnp.log(0.0007), jnp.log(1)))
            durations.append(numpyro.deterministic(f"duration_{i}", jnp.exp(logD)))
            t0s.append(numpyro.sample(f"t0_{i}", dist.Uniform(jnp.min(t), jnp.max(t))))
            _b = numpyro.sample(f"_b_{i}", dist.Uniform(-2.0, 2.0))
            bs.append(numpyro.deterministic(f'b_{i}', jnp.abs(_b)))
            depths = numpyro.sample(f'depths_{i}', dist.Uniform(1e-6, 0.5))
            rorss.append(numpyro.deterministic(f"rors_{i}", jnp.sqrt(depths)))

        if ld_profile == 'quadratic':
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

        params = {
            "period": prior_params['period'], "duration": jnp.array(durations), "t0": jnp.array(t0s),
            "b": jnp.array(bs), "rors": jnp.array(rorss), "u": u,
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

        if 'gp' in detrend_type:
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

def create_vectorized_model(detrend_type='linear', ld_mode='free', trend_mode='free', n_planets=1, ld_profile='quadratic'):
    print(f"Building vectorized model with: detrend='{detrend_type}', ld='{ld_mode}', trend='{trend_mode}' for {n_planets} planets")

    detrend_components = _split_components(detrend_type)
    try:
        compute_lc_kernel = resolve_detrend_kernel(detrend_type)
    except KeyError:
        raise ValueError(f"Unsupported detrend_type for vectorized model: {detrend_type}")
    if detrend_components in ({"spot"}, {"2spot"}, {"linear_discontinuity"}):
        raise ValueError(f"Vectorized model does not support '{detrend_type}'. Use '{detrend_type}_spectroscopic' instead.")
    if ld_profile == "power2":
        MUS_LD, P_LD = _prepare_power2_poly()

    def _vectorized_model_static(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_b=None,
                               mu_depths=None, PERIOD=None, trend_fixed=None,
                               ld_interpolated=None, ld_fixed=None,
                               mu_spot_amp=None, mu_spot_mu=None, mu_spot_sigma=None,
                               mu_u_ld=None, gp_trend=None, spot_trend=None, spot_trend2=None, jump_trend=None):

        num_lcs = jnp.atleast_2d(yerr).shape[0]
        durations = mu_duration
        t0s = mu_t0
        bs = mu_b

        depths = numpyro.sample('depths', dist.Uniform(1e-5, 0.5).expand([num_lcs, n_planets]))
        rors = numpyro.deterministic("rors", jnp.sqrt(depths))

        yerr_per_lc = jnp.nanmedian(yerr, axis=1)
        log_jitter = numpyro.sample('log_jitter', dist.Uniform(jnp.log(1e-6), jnp.log(1)).expand([num_lcs]))
        jitter = jnp.exp(log_jitter)
        total_error = numpyro.deterministic('total_error', jnp.sqrt(jitter**2 + yerr_per_lc**2))
        error_broadcast = total_error[:, None] * jnp.ones_like(t)

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

        params = {
            "period": PERIOD, "duration": durations, "t0": t0s, "b": bs, "rors": rors, "u": u,
        }
        in_axes = {"period": None, "duration": None, "t0": None, "b": None, "rors": 0, "u": 0}

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
