"""
linear_theory.py
Richtmyer impulsive model for RMI growth, including magnetic suppression.
"""

import numpy as np

def richtmyer_linear_theory(t_arr, gamma, mach, rho1, rho_heavy, a0, k, By=0.0):
    """Compute linear RMI growth from Richtmyer's impulsive model."""
    M2 = mach * mach

    if abs(By) < 1e-14:
        cs1 = np.sqrt(gamma * 1.0 / rho1)
        vs = mach * cs1
    else:
        cs1_sq = gamma * 1.0 / rho1
        va1_sq = By**2 / rho1
        cf1 = np.sqrt(cs1_sq + va1_sq)
        vs = mach * cf1

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    a0_post = a0 / r
    rho2_light = rho1 * r
    rho2_heavy = rho_heavy * r

    A_post = (rho2_heavy - rho2_light) / (rho2_heavy + rho2_light)
    delta_v = vs * (1.0 - 1.0 / r)

    x_shock_init = 1.2
    x_interface = 1.5
    t_shock = (x_interface - x_shock_init) / vs

    da_dt = A_post * k * a0_post * delta_v

    a_linear = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        if t < t_shock:
            a_linear[i] = a0
        else:
            dt = t - t_shock
            if abs(By) < 1e-14:
                a_linear[i] = a0_post + da_dt * dt
            else:
                rho_avg = 0.5 * (rho2_light + rho2_heavy)
                vA = By / np.sqrt(rho_avg)
                omega_A = k * vA
                if omega_A > 1e-14:
                    a_linear[i] = a0_post * np.cos(omega_A * dt) +                                   (da_dt / omega_A) * np.sin(omega_A * dt)
                else:
                    a_linear[i] = a0_post + da_dt * dt

    info = {
        'A_post': A_post,
        'da_dt': da_dt,
        'delta_v': delta_v,
        'vs': vs,
        't_shock': t_shock,
        'a0_post': a0_post,
        'r': r,
    }
    return np.abs(a_linear), t_shock, info
