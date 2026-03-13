import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
import jax.numpy as jnp

def plot_map_fits(t, indiv_y, jitter, wavelengths, map_params, transit_params, filename, ncols=3, detrend_type='linear'):
    """
    Plot the MAP fits for each wavelength. The transit parameters (period, duration,
    impact parameter, and transit time) are provided via transit_params, and detrend_offset
    is the value used to detrend the light curve.
    """
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))

    
    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        # Extract MAP parameters for this wavelength
        rors_i = map_params['rors'][i]
        u_i = map_params['u'][i]
        if detrend_type != 'none':
            c_i = map_params['c'][i]
            v_i = map_params['v'][i]

        # Multi-planet model calculation
        total_model_flux = np.zeros_like(t)
        periods = np.atleast_1d(transit_params["period"])
        durations = np.atleast_1d(map_params["duration"])
        bs = np.atleast_1d(map_params["b"])
        t0s = np.atleast_1d(map_params["t0"])
        rors_i_all_planets = np.atleast_1d(rors_i)

        num_planets = len(periods)
        for p_idx in range(num_planets):
            orbit = TransitOrbit(
                period=periods[p_idx],
                duration=durations[p_idx],
                impact_param=bs[p_idx],
                time_transit=t0s[p_idx],
                radius_ratio=rors_i_all_planets[p_idx],
            )
            planet_model = limb_dark_light_curve(orbit, u_i)(t)
            total_model_flux += planet_model

        model = total_model_flux

        if detrend_type == 'linear':
            trend = c_i + v_i * (t - jnp.min(t))
        elif detrend_type == 'explinear':
            A_i = map_params['A'][i]
            tau_i = map_params['tau'][i]
            trend = c_i + v_i * (t - jnp.min(t)) + A_i * jnp.exp(-(t - jnp.min(t))/tau_i)
        elif detrend_type == 'spot':
            spot_amp = map_params['spot_amp'][i]
            spot_mu = map_params['spot_mu'][i]
            spot_sigma = map_params['spot_sigma'][i]
            trend = c_i + v_i * (t - jnp.min(t)) + (spot_amp * jnp.exp(-0.5 * (t - spot_mu)**2 / spot_sigma**2))
        elif detrend_type == '2spot':
            spot_amp = map_params['spot_amp'][i]
            spot_mu = map_params['spot_mu'][i]
            spot_sigma = map_params['spot_sigma'][i]
            spot_amp2 = map_params['spot_amp2'][i]
            spot_mu2 = map_params['spot_mu2'][i]
            spot_sigma2 = map_params['spot_sigma2'][i]
            spot_term = spot_amp * jnp.exp(-0.5 * (t - spot_mu)**2 / spot_sigma**2)
            spot_term2 = spot_amp2 * jnp.exp(-0.5 * (t - spot_mu2)**2 / spot_sigma2**2)
            trend = c_i + v_i * (t - jnp.min(t)) + spot_term + spot_term2
        elif detrend_type == 'none':
            trend = 1.0
        else:
            # Fallback for complex detrending types in plots (simplification)
            if detrend_type != 'none':
                 try:
                    trend = c_i + v_i * (t - jnp.min(t))
                 except:
                    trend = 1.0
            else:
                 trend = 1.0
            
        model = model + trend
        ax.errorbar(t, indiv_y[i], yerr=jitter[i], fmt='.', alpha=0.3,
                    color=colors[i], label='Data', ms=1, zorder=2)
        ax.plot(t, model, c='k', alpha=1, lw=2.8,
                label='MAP Model', zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.2f} μm',
                transform=ax.transAxes, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig


def plot_map_residuals(t, indiv_y, jitter, wavelengths, map_params, transit_params, filename, ncols=3, detrend_type='linear'):
    """
    Plot the residuals for each wavelength using the transit parameters provided via transit_params.
    """
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))

    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        rors_i = map_params['rors'][i]
        u_i = map_params['u'][i]
        if detrend_type != 'none':
            c_i = map_params['c'][i]
            v_i = map_params['v'][i]
        
        # Multi-planet model calculation
        total_model_flux = np.zeros_like(t)
        periods = np.atleast_1d(transit_params["period"])
        durations = np.atleast_1d(map_params["duration"])
        bs = np.atleast_1d(map_params["b"])
        t0s = np.atleast_1d(map_params["t0"])
        rors_i_all_planets = np.atleast_1d(rors_i)

        num_planets = len(periods)
        for p_idx in range(num_planets):
            orbit = TransitOrbit(
                period=periods[p_idx],
                duration=durations[p_idx],
                impact_param=bs[p_idx],
                time_transit=t0s[p_idx],
                radius_ratio=rors_i_all_planets[p_idx],
            )
            planet_model = limb_dark_light_curve(orbit, u_i)(t)
            total_model_flux += planet_model

        model = total_model_flux
        if detrend_type == 'linear':
            trend = c_i + v_i * (t - jnp.min(t))
        elif detrend_type == 'explinear':
            A_i = map_params['A'][i]
            tau_i = map_params['tau'][i]
            trend = c_i + v_i * (t - jnp.min(t)) + A_i * jnp.exp(-(t - jnp.min(t))/tau_i)
        elif detrend_type == 'spot':
            spot_amp = map_params['spot_amp'][i]
            spot_mu = map_params['spot_mu'][i]
            spot_sigma = map_params['spot_sigma'][i]
            trend = c_i + v_i * (t - jnp.min(t)) + (spot_amp * jnp.exp(-0.5 * (t - spot_mu)**2 / spot_sigma**2))
        elif detrend_type == '2spot':
            spot_amp = map_params['spot_amp'][i]
            spot_mu = map_params['spot_mu'][i]
            spot_sigma = map_params['spot_sigma'][i]
            spot_amp2 = map_params['spot_amp2'][i]
            spot_mu2 = map_params['spot_mu2'][i]
            spot_sigma2 = map_params['spot_sigma2'][i]
            spot_term = spot_amp * jnp.exp(-0.5 * (t - spot_mu)**2 / spot_sigma**2)
            spot_term2 = spot_amp2 * jnp.exp(-0.5 * (t - spot_mu2)**2 / spot_sigma2**2)
            trend = c_i + v_i * (t - jnp.min(t)) + spot_term + spot_term2
        elif detrend_type == 'none':
            trend = 1.0
        else:
             # Fallback
            if detrend_type != 'none':
                 try:
                    trend = c_i + v_i * (t - jnp.min(t))
                 except:
                    trend = 1.0
            else:
                 trend = 1.0

        model = model + trend
        residuals = indiv_y[i] - model
        
        ax.errorbar(t, residuals, yerr=jitter[i], fmt='.', alpha=0.3,
                    ms=1, color=colors[i])
        ax.axhline(y=0, color='k', alpha=1, lw=2.8, zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.3f} μm',
                transform=ax.transAxes, fontsize=10)
        rms = np.nanmedian(np.abs(np.diff(residuals)))*1e6
        ax.text(0.05, 0.85, f'Noise = {round(rms)}',
                transform=ax.transAxes, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig


def plot_transmission_spectrum(wavelengths, rors_posterior, filename):
    """
    Plot the transmission spectrum from MCMC results. Handles multi-planet systems.
    
    Parameters:
    -----------
    wavelengths : array-like
        The wavelengths at which the spectrum is measured.
    rors_posterior : array-like
        The posterior samples for rors (radius ratio). Shape can be (n_samples, n_lcs) for
        a single planet or (n_samples, n_lcs, n_planets) for multiple planets.
    """
    depth_chain = rors_posterior**2

    if depth_chain.ndim == 2:  # single planet case for backward compatibility
        depth_chain = depth_chain[:, :, np.newaxis]

    n_planets = depth_chain.shape[2]

    fig = plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_planets))

    for i in range(n_planets):
        plt.figure()
        planet_depth_chain = depth_chain[:, :, i]
        depth_median = np.percentile(planet_depth_chain, 50, axis=0)
        depth_16 = np.percentile(planet_depth_chain, 16, axis=0)
        depth_84 = np.percentile(planet_depth_chain, 84, axis=0)

        y_err = [depth_median - depth_16, depth_84 - depth_median]

        plt.errorbar(wavelengths, depth_median * 1e6,
                     yerr=np.array(y_err) * 1e6,
                     fmt='o', mfc='k', mec='k', ecolor='k', label=f'Planet {i + 1}')

        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Depth (ppm)")
        plt.savefig(filename+f'_0{i}', dpi=200)
        plt.close()
    return fig

def plot_wavelength_offset_summary(
    t, indiv_y, jitter, wavelengths, map_params, transit_params,
    filename, detrend_type='linear', use_hours=True, residual_scale=2.0,
    gp_trend=None, spot_trend=None, spot_trend2=None, jump_trend=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    num_lcs = indiv_y.shape[0]

    # --- choose up to 10 curves across wavelength, sort by wavelength ---
    if num_lcs > 10:
        min_wl, max_wl = wavelengths.min(), wavelengths.max()
        target_wls = np.linspace(min_wl, max_wl, 10)
        indices = [np.argmin(np.abs(wavelengths - wl)) for wl in target_wls]
        indices = np.unique(indices)
    else:
        indices = np.arange(num_lcs)
    indices = indices[np.argsort(wavelengths[indices])]
    selected_lcs = len(indices)

    # --- time relative to t0 (hours like the paper) ---
    t0_ref = float(np.atleast_1d(map_params["t0"])[0]) # Center on first planet
    t_centered = (t - t0_ref) * (24.0 if use_hours else 1.0)
    t_unit = "hours" if use_hours else "days"

    # --- colors ---
    norm = plt.Normalize(wavelengths[indices].min(), wavelengths[indices].max())
    cmap = plt.cm.turbo
    colors = cmap(norm(wavelengths[indices]))

    # --- depths & vertical spacing ---
    all_rors = np.asarray(map_params['rors'])[indices]
    # Use first planet's depths for spacing if multi-planet
    rors_for_spacing = all_rors if all_rors.ndim == 1 else all_rors[:, 0]
    depths = rors_for_spacing**2
    depth_med = float(np.nanmedian(depths))
    depth_max = float(np.nanmax(depths))
    step = 0.5 * depth_med
    offsets = np.arange(selected_lcs) * step

    # --- figure: 2/3 model, 1/3 residuals ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(12, 0.9 + 0.6 * selected_lcs),
        gridspec_kw={'width_ratios': [2, 1]},
        sharey=True
    )

    for ax in (ax1, ax2):
       # ax.spines['top'].set_visible(False)
       # ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)


    # --- per-curve plotting ---
    for i, idx in enumerate(indices):
        yoff = offsets[i]
        rors_i_all_planets = np.atleast_1d(map_params['rors'][idx])
        u_i = np.asarray(map_params['u'][idx])

        # Multi-planet model calculation
        total_model_flux = np.zeros_like(t)
        periods = np.atleast_1d(transit_params["period"])
        durations = np.atleast_1d(map_params["duration"])
        bs = np.atleast_1d(map_params["b"])
        t0s = np.atleast_1d(map_params["t0"])
        num_planets = len(periods)

        for p_idx in range(num_planets):
            orbit = TransitOrbit(
                period=periods[p_idx],
                duration=durations[p_idx],
                impact_param=bs[p_idx],
                time_transit=t0s[p_idx],
                radius_ratio=rors_i_all_planets[p_idx],
            )
            planet_model = limb_dark_light_curve(orbit, u_i)(t)
            total_model_flux += planet_model

        model_transit = total_model_flux

        # Detrend model
        trend = np.ones_like(t)
        if detrend_type != 'none':
            c_i = map_params['c'][idx]
            
            if 'gp_spectroscopic' in detrend_type:
                trend_parametric = c_i
                if 'linear' in detrend_type and 'v' in map_params:
                    trend_parametric += map_params['v'][idx] * (t - np.min(t))
                if 'quadratic' in detrend_type and 'v2' in map_params:
                     trend_parametric += map_params['v2'][idx] * (t - np.min(t))**2
                
                trend = trend_parametric + map_params['A_gp'][idx] * gp_trend
            
            elif detrend_type == '2spot_spectroscopic':
                trend = c_i + map_params['A_spot'][idx] * spot_trend + map_params['A_spot2'][idx] * spot_trend2
            elif detrend_type == 'spot_spectroscopic':
                trend = c_i + map_params['A_spot'][idx] * spot_trend
            
            elif detrend_type == 'linear_discontinuity_spectroscopic':
                v_i = map_params.get('v', [0.0]*num_lcs)[idx]
                t_shift = t - np.min(t)
                trend = c_i + v_i * t_shift + map_params['A_jump'][idx] * jump_trend

            else:
                v_i = map_params.get('v', [0.0]*num_lcs)[idx]
                t_shift = t - np.min(t)
                if detrend_type == 'linear':
                    trend = c_i + v_i * t_shift
                elif detrend_type == 'linear_discontinuity':
                    t_jump_i = map_params['t_jump'][idx]
                    jump_i = map_params['jump'][idx]
                    trend = c_i + v_i * t_shift + np.where(t > t_jump_i, jump_i, 0.0)
                elif detrend_type == 'explinear':
                    A_i = map_params['A'][idx]; tau_i = map_params['tau'][idx]
                    trend = c_i + v_i * t_shift + A_i * np.exp(-t_shift / tau_i)
                elif detrend_type == 'spot':
                    spot_amp = map_params['spot_amp'][idx]
                    spot_mu  = map_params['spot_mu'][idx]
                    spot_sig = map_params['spot_sigma'][idx]
                    trend = c_i + v_i * t_shift + spot_amp * np.exp(-0.5 * (t - spot_mu)**2 / spot_sig**2)
                elif detrend_type == '2spot':
                    spot_amp = map_params['spot_amp'][idx]
                    spot_mu  = map_params['spot_mu'][idx]
                    spot_sig = map_params['spot_sigma'][idx]
                    spot_amp2 = map_params['spot_amp2'][idx]
                    spot_mu2  = map_params['spot_mu2'][idx]
                    spot_sig2 = map_params['spot_sigma2'][idx]
                    spot_term = spot_amp * np.exp(-0.5 * (t - spot_mu)**2 / spot_sig**2)
                    spot_term2 = spot_amp2 * np.exp(-0.5 * (t - spot_mu2)**2 / spot_sig2**2)
                    trend = c_i + v_i * t_shift + spot_term + spot_term2
                elif detrend_type == 'quadratic':
                    v2_i = map_params['v2'][idx]
                    trend = c_i + v_i * t_shift + v2_i * t_shift**2
                # Fallback
                else:
                    trend = c_i + v_i * t_shift

        full_model = model_transit + trend
        resid = indiv_y[idx] - full_model

        # --- Left: data + model (same color) ---
        ax1.scatter(t_centered, indiv_y[idx] - yoff, s=3, alpha=0.45,
                    color=colors[i], rasterized=True)
        ax1.plot(t_centered, full_model - yoff, '-', lw=1, color=colors[i])

        ax1.text(
            t_centered.min(), 1.0 - yoff + 0.001,
            f"{wavelengths[idx]:.2f} μm",
            ha='left', va='bottom', color=colors[i], fontsize=12, fontweight='bold'
        )

        # --- Right: residuals ONLY ---
        baseline = 1.0 - yoff
        y_res_plot = baseline + residual_scale * resid   # <- no model added
        ax2.scatter(t_centered, y_res_plot, s=3, alpha=0.45,
                    color=colors[i], rasterized=True)
        ax2.axhline(baseline, color=colors[i], linestyle='--', lw=0.8, alpha=0.8)
        ax2.text(
            t_centered.min(), baseline + 0.15 * step,
            f"Error: {jitter[idx]*1e6:.0f} ppm",
            ha='right', va='bottom', color=colors[i], fontsize=12, fontweight='bold'
        )


    # --- dynamic y-lims (LEFT): one depth above, one below lowest min ---
    y_top_left = 1.0 + depth_max * 0.15
    lowest_min_left = np.min(1.0 - offsets - depths)
    y_bot_left = lowest_min_left - depth_max * 0.1
    ax1.set_ylim(y_bot_left, y_top_left)
    

    # labels & x-lims
    for ax in (ax1, ax2):
        ax.set_xlim(t_centered.min(), t_centered.max())
        ax.set_xlabel(f"time from transit center [{t_unit}]", fontsize=10)
    ax1.set_ylabel("relative flux (+ offset)", fontsize=10)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(labelleft=False)  # keep y only on the left
    plt.subplots_adjust(wspace=0.02)  # almost touching
    
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=False, prune=None))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    return fig
