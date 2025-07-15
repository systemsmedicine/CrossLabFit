# run_optimization.py

import numpy as np
import warnings
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.integrate import ODEintWarning
from scipy.optimize import differential_evolution
from functools import partial

def rss(data_A, interp_func):
    try:
        times = data_A[:, 0]
        obs = data_A[:, 1]

        # Find unique times and how many replicates there are per time
        unique_times, counts = np.unique(times, return_counts=True)

        # Interpolate model at unique time points
        model_values = interp_func(unique_times)

        # Repeat to match number of replicates
        model_repeated = np.repeat(model_values, counts)

        diff = obs - model_repeated
        if not np.all(np.isfinite(diff)):
            return 1e6

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rss = np.sum(diff ** 2)

        if not np.isfinite(rss):
            return 1e6
        
        return rss

    except Exception:
        return 1e6

def costFunction(params, model, data_A, windows=None, use_windows=True,
                  num_variables=3, rss_variable=0, window_variable=2, penalty_value=1e6):
    try:
        # Separate parameters and initial conditions
        param_vec = params[:-num_variables]
        X0 = params[-num_variables:]

        # Time range
        t_max_A = np.max(data_A[:, 0])
        if use_windows and windows is not None:
            t_max = max(t_max_A, np.max(windows[:, 1]))
        else:
            t_max = t_max_A

        t_eval = np.linspace(0, t_max, 1000)

        # Solve ODE model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ODEintWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            sol = odeint(model, X0, t_eval, args=(param_vec,))

        if not np.all(np.isfinite(sol)):
            return penalty_value * np.random.uniform(1, 2)

        # Window constraint check
        if use_windows and windows is not None:
            var_w = sol[:, window_variable]
            for row in windows:
                tmin, tmax, vmin, vmax = row
                mask = (t_eval >= tmin) & (t_eval <= tmax)
                if not np.any((var_w[mask] >= vmin) & (var_w[mask] <= vmax)):
                    return penalty_value * np.random.uniform(1, 2)

        # RSS computation
        var_rss = sol[:, rss_variable]
        interp_func = interp1d(t_eval, var_rss, kind='linear', bounds_error=False, fill_value='extrapolate')
        return rss(data_A, interp_func)

    except Exception:
        return penalty_value * np.random.uniform(1, 2)

def run_DE(bounds, model_func, data_A, windows, use_windows, num_variables,
           rss_variable, window_variable, penalty_value=1e6, maxiter=100, popsize=15, disp=True):

    objective = partial(
    costFunction,
    model=model_func,
    data_A=data_A,
    windows=windows,
    use_windows=use_windows,
    num_variables=num_variables,
    rss_variable=rss_variable,
    window_variable=window_variable,
    penalty_value=penalty_value
    )

    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy='rand1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=0.8,
        recombination=0.8,
        polish=True,
        workers=-1,
        updating='deferred',
        disp=disp
    )

    return result

def run_profile_likelihood(param_index, param_range, steps, bounds, model_func, data_A, windows,
                               use_windows, num_variables, rss_variable, window_variable,
                               penalty_value=1e6, maxiter=100, popsize=20, seed=42):

    fixed_values = np.linspace(param_range[0], param_range[1], steps)
    profile_costs = []

    for i, val in enumerate(fixed_values):
        # Create a copy of bounds
        mod_bounds = bounds.copy()
        
        # Fix the selected parameter by setting min = max = val
        mod_bounds[param_index] = (val, val)

        # Run optimization with modified bounds
        result = run_DE(
            bounds=mod_bounds,
            model_func=model_func,
            data_A=data_A,
            windows=windows,
            use_windows=use_windows,
            num_variables=num_variables,
            rss_variable=rss_variable,
            window_variable=window_variable,
            penalty_value=penalty_value,
            maxiter=maxiter,
            disp=False,
        )

        profile_costs.append(result.fun)

        print(f"[{i+1}/{steps}] Fixed param {param_index+1} at {val:.4f} → Cost: {result.fun:.4f}")

    return fixed_values, profile_costs

def run_bootstrap(n_bootstrap, bounds, model_func, data_A, windows=None, use_windows=False,
              num_variables=3, rss_variable=0, window_variable=2, penalty_value=1e6, 
              maxiter=100, popsize=20):

    # Store all bootstrap estimates
    param_matrix = []

    # Extract times for proper resampling grouping
    unique_times = np.unique(data_A[:, 0])

    rng = np.random.default_rng()

    for b in range(n_bootstrap):
        # RESAMPLING WITH REPLACEMENT
        resample_idx = rng.integers(low=0, high=data_A.shape[0], size=data_A.shape[0])
        bootstrap_sample = data_A[resample_idx, :]

        # RUN OPTIMIZATION
        result = run_DE(
            bounds=bounds,
            model_func=model_func,
            data_A=bootstrap_sample,
            windows=windows,
            use_windows=use_windows,
            num_variables=num_variables,
            rss_variable=rss_variable,
            window_variable=window_variable,
            penalty_value=penalty_value,
            maxiter=maxiter,
            popsize=popsize,
            disp=False
        )

        print(f"[Bootstrap {b+1}/{n_bootstrap}] Final cost: {result.fun:.4f}")

        # Save only parameter estimates (exclude X0)
        best_params = result.x[:-num_variables]
        param_matrix.append(best_params)

    return np.array(param_matrix)