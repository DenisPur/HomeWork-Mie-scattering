import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from numpy import interp
from scipy.optimize import curve_fit
from scipy.integrate import simpson, trapezoid
# import PyMieScatt as ps
import miepython


# def vectorized_RayleighMieQ(wl, m, d):
#     assert len(wl) == len(m)
#     result = np.zeros_like(wl)
#     for index in range(len(wl)):
#         _, _, _, _, q_pr, _, _ = ps.RayleighMieQ(m[index], wl[index]*10e9, d)
#         result[index] = q_pr
#     return result


def vectorized_RayleighMieQ(wl, m, d):
    assert len(wl) == len(m)
    result = np.zeros_like(wl)
    for index in range(len(wl)):
        qext, qsca, qback, g = miepython.mie(m[index], np.pi * d / (wl[index]*10e9))
        result[index] = qext - qsca
    return result


def vectorized_Qpr(dust_nk: pd.DataFrame,
                   wl_in_meters: np.ndarray,
                   diameter_in_nm: float) -> np.ndarray:
    waves = dust_nk['wavelenght'].to_numpy()
    n_array = dust_nk['n'].to_numpy()
    k_array = dust_nk['k'].to_numpy()

    interp_n = interp(wl_in_meters, waves, n_array)
    interp_k = interp(wl_in_meters, waves, k_array)

    m_array = interp_n - interp_k * 1j
    q_pr = vectorized_RayleighMieQ(wl_in_meters, m_array, diameter_in_nm)
    return q_pr


def planck_shorten(wl, t):
    h = 6.62607015 * 10**(-34)
    c = 299_792_458
    k = 1.380649 * 10**(-23)  
    pwr = h * c / (wl * k * t)
    intensity = 1 / ( (wl**5) * (np.exp(pwr) - 1.0) )
    return intensity / max(intensity)


def Qpr_tilda(dust_nk: pd.DataFrame,
              wl_in_meters: np.ndarray,
              diameter_in_nm: float,
              temperature: float) -> float:
    q_pr = vectorized_Qpr(dust_nk, wl_in_meters, diameter_in_nm)
    plank = planck_shorten(wl_in_meters, temperature)

    return trapezoid(q_pr * plank, wl_in_meters) / trapezoid(plank, wl_in_meters)


def vectorized_Qpr_tilda(dust_nk: pd.DataFrame,
                         diameter_in_nm: float | np.ndarray,
                         temperature: float | np.ndarray) -> np.ndarray | float:
    wl_metr = 10.0 ** np.arange(start=-8, stop=-5, step=0.003)
    
    if isinstance(diameter_in_nm, np.ndarray) and isinstance(temperature, np.ndarray):
        raise TypeError
    if isinstance(diameter_in_nm, np.ndarray):
        result = np.zeros_like(diameter_in_nm, dtype=np.float64)
        for index in range(len(diameter_in_nm)):
            result[index] = Qpr_tilda(dust_nk, wl_metr, diameter_in_nm[index], temperature)
    elif isinstance(temperature, np.ndarray):
        result = np.zeros_like(temperature, dtype=np.float64)
        for index in range(len(temperature)):
            result[index] = Qpr_tilda(dust_nk, wl_metr, diameter_in_nm, temperature[index])
    else:
        result = Qpr_tilda(dust_nk, wl_metr, diameter_in_nm, temperature)
    return result


def stars_TRM(sequence: str, type: str) -> tuple[float, float, float]:
        # Temperatures
    T_array = np.array([15500, 8500, 6580, 5520, 4130])
        # Main sequence
    R_ms = 10**np.array([0.58, 0.24, 0.08, -0.03, -0.13])
    M_ms = 10**np.array([0.81, 0.32, 0.11, -0.03, -0.11])
        # Giants
    R_g = 10**np.array([1.0, 0.7, 0.6, 1.0, 1.4])
    M_g = 10**np.array([1.4, 1.1, 1.0, 1.1, 1.2])
        # Supergiants
    R_sup = 10**np.array([1.5, 1.7, 1.9, 2.1, 2.6])
    M_sup = 10**np.array([0.81, 0.32, 0.11, 0.5, 0.7])

    type_dict = {'B' : 0, 'A' : 1, 'F' : 2, 'G' : 3, 'K' : 4}
    temp_i = type_dict[type]

    if sequence.lower().startswith('m'):
        return T_array[temp_i], R_ms[temp_i], M_ms[temp_i]
    if sequence.lower().startswith('g'):
        return T_array[temp_i], R_g[temp_i], M_g[temp_i]
    if sequence.lower().startswith('s'):
        return T_array[temp_i], R_sup[temp_i], M_sup[temp_i]


def beta(dust_nk, star_prop, a, rho) -> float:
    t, r, m = star_prop
    q = vectorized_Qpr_tilda(dust_nk, 2*a, t)
    return 2.12e-8 * (r*7e8)**2 * t**4 / (m*2e30) * q / (a*1e-9 * rho)
