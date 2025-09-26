import numpy as np
import warnings
from functools import partial
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import ODEintWarning
from scipy.optimize import differential_evolution

# ===== RSS for a single dataset =====
def rss_one(data_A, interp_func, penalty_value=1e6):
    try:
        times = data_A[:, 0]
        obs   = data_A[:, 1]

        # Find unique times and how many replicates there are per time
        unique_times, counts = np.unique(times, return_counts=True)

        # Interpolate model at unique time points
        model_vals = interp_func(unique_times)

        # Repeat to match number of replicates
        model_rep  = np.repeat(model_vals, counts)

        diff = obs - model_rep
        if not np.all(np.isfinite(diff)):
            return penalty_value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rss = np.sum(diff**2)
        return rss if np.isfinite(rss) else penalty_value
        
    except Exception:
        return penalty_value

# ===== Generalized cost function =====
def cost_function(
    params,
    model_func,
    num_variables,        # Total number of model variables
    data_list,            # list of arrays: each [:,0]=time, [:,1]=obs
    qt_vars,             # list of ints: variable index of the model for each data array
    windows_list=None,    # list of arrays (Tmin,Tmax,Vmin,Vmax) OR None per item
    win_vars=None,        # list of ints: variable index of the model for each windows array
    use_windows=True,
    penalty_value=1e6,
):
    try:
        if not isinstance(data_list, (list, tuple)):
            data_list = [data_list]
        if not isinstance(qt_vars, (list, tuple)):
            qt_vars = [qt_vars]

        if windows_list is None:
            windows_list = []
        if win_vars is None:
            win_vars = []

        # split parameters and initial conditions
        p_vec = params[:-num_variables]
        X0    = params[-num_variables:]

        # time horizon: cover all data times and window Tmax
        t_max = 0.0
        for d in data_list:
            if d is not None and len(d) > 0:
                t_max = max(t_max, np.max(d[:, 0]))
        if use_windows and windows_list:
            for W in windows_list:
                if W is not None and len(W) > 0:
                    t_max = max(t_max, np.max(W[:, 1]))
        if t_max <= 0:
            t_max = 1.0

        # High resolution to evaluate window violations
        t_eval = np.linspace(0.0, t_max, 1000)

        # integrate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ODEintWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            sol = odeint(model_func, X0, t_eval, args=(p_vec,))

        # The stochasticity is an artifact to prevent the DE from stopping due to a homogeneous population.
        if not np.all(np.isfinite(sol)):
            return penalty_value * np.random.uniform(1, 2)

        # window constraints across all provided sets
        if use_windows and windows_list:
            for W, v_idx in zip(windows_list, win_vars):
                if W is None or len(W) == 0:
                    continue
                traj = sol[:, v_idx]
                for row in W:
                    Tmin, Tmax, Vmin, Vmax = row
                    mask = (t_eval >= Tmin) & (t_eval <= Tmax)
                    if not np.any((traj[mask] >= Vmin) & (traj[mask] <= Vmax)):
                        return penalty_value * np.random.uniform(1, 2)

        # total RSS across all quantitative datasets
        total_rss = 0.0
        for d, v_idx in zip(data_list, qt_vars):
            if d is None or len(d) == 0:
                continue
            interp_f = interp1d(
                t_eval, sol[:, v_idx],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )
            total_rss += rss_one(d, interp_f, penalty_value=penalty_value)

        if not np.isfinite(total_rss):
            return penalty_value

        return total_rss

    except Exception:
        return penalty_value * np.random.uniform(1, 2)

# ===== DE wrapper =====
def run_DE(
    bounds,
    model_func,
    num_variables,
    data_list,
    qt_vars,
    windows_list=None,
    win_vars=None,
    use_windows=True,
    penalty_value=1e6,
    maxiter=100,
    popsize=15,
    disp=True,
):
    objective = partial(
        cost_function,
        model_func=model_func,
        num_variables=num_variables,
        data_list=data_list,
        qt_vars=qt_vars,
        windows_list=windows_list,
        win_vars=win_vars,
        use_windows=use_windows,
        penalty_value=penalty_value,
    )

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy='rand1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        workers=-1,
        updating='deferred',
        disp=disp,
    )
    return result

# ===== profile likelihood  =====
def run_profile_likelihood(
    param_index, param_range, steps,
    bounds,
    model_func,
    num_variables,
    data_list, qt_vars,
    windows_list=None, win_vars=None,
    use_windows=True,
    penalty_value=1e6,
    maxiter=100, popsize=20, seed=42,
):
    fixed_vals = np.linspace(param_range[0], param_range[1], steps)
    costs = []
    for i, val in enumerate(fixed_vals):
        mod_bounds = list(bounds)
        mod_bounds[param_index] = (val, val)
        res = run_DE(
            bounds=mod_bounds,
            model_func=model_func,
            num_variables=num_variables,
            data_list=data_list,
            qt_vars=qt_vars,
            windows_list=windows_list,
            win_vars=win_vars,
            use_windows=use_windows,
            penalty_value=penalty_value,
            maxiter=maxiter,
            popsize=popsize,
            disp=False,
        )
        costs.append(res.fun)
        
        print(f"[{i+1}/{steps}] Fixed param {param_index+1} = {val:.4f} → Cost {res.fun:.4f}")
        
    return fixed_vals, np.array(costs)

# ===== bootstrap (generalized over the first quantitative dataset by default) =====
def run_bootstrap(
    n_bootstrap,
    bounds,
    model_func,
    num_variables,
    data_list, qt_vars,
    windows_list=None, win_vars=None,
    use_windows=False,
    penalty_value=1e6,
    maxiter=100, popsize=20, seed=0,
    resample_index=0,   # which dataset in data_list to resample
):
    rng = np.random.default_rng(seed)
    
    param_mat = []
    for b in range(n_bootstrap):
        dl = list(data_list)
        d  = dl[resample_index]
        idx = rng.integers(0, d.shape[0], size=d.shape[0])
        dl[resample_index] = d[idx, :]  # resampled copy

        res = run_DE(
            bounds=bounds,
            model_func=model_func,
            num_variables=num_variables,
            data_list=dl,
            qt_vars=qt_vars,
            windows_list=windows_list,
            win_vars=win_vars,
            use_windows=use_windows,
            penalty_value=penalty_value,
            maxiter=maxiter,
            popsize=popsize,
            disp=False,
        )
        
        print(f"[Bootstrap {b+1}/{n_bootstrap}] cost={res.fun:.4f}")

        # Save only parameter estimates (exclude X0)
        param_mat.append(res.x[:-num_variables])
        
    return np.array(param_mat)
