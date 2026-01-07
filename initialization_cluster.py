import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tools import natural_units as nu
from numpy.polynomial.laguerre import laggauss
from multiprocessing import Pool
from scipy.interpolate import interp1d

# Cluster/batch plumbing (outer serial scan; inner Pool per scan point)
import argparse
import itertools

# ----------------------
# Precision
mp.mp.dps = 25

# ----------------------
# Inner parallelism per scan point
# (Used only inside precompute_brem_table / precompute_anni_table)
N_proc = 6

# ----------------------
# Output root
base_path = "./test"

# ----------------------
# Scan grids (EDIT HERE; no need to type in terminal)
# All grids are in *physical units* (GeV / keV) or dimensionless.
SCAN_GRIDS = {
    "mchi_GeV": list(np.logspace(1, 5, 5)),     # e.g. 1e3 ... 1e5 GeV
    "mV_keV":   list(np.logspace(-3, 3, 7)),    # e.g. 0.1 ... 100 keV
    "alpha":    [0.01, 0.1, 1.0],
}

# ----------------------
# Default single-point tag (used when --mode single)
my_tag = "brem_test_1"

# ----------------------
# Halo fiducials (dimensionful, in nu)
rho_s     = 1.28e7 * nu.mSun / nu.kpc**3
r_s       = 6.5 * nu.kpc
sigma_fid = 1 / rho_s / r_s
M_fid     = 4.0 * mp.pi * rho_s * r_s**3
v_fid     = mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s) * r_s
lumi_fid  = mp.power(4 * mp.pi * rho_s * r_s**2, 5/2) * mp.power(nu.G_Newton, 3/2)
t_fid     = 1 / mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s)
C_fid     = mp.power(4 * mp.pi * nu.G_Newton, 3/2) * mp.power(rho_s, 5/2) * r_s**2

# ----------------------
# Conductivity parameters
a = mp.mpf('2.257')
c = mp.mpf('0.60')

# ----------------------
# Default baryon Plummer parameters (dimensionless)
my_mass_norm = mp.mpf('0.0')
my_scale_norm = mp.mpf('0.1')

# ----------------------
# Default particle physics parameters (dimensionful in nu)
m_chi = 1e5 * nu.GeV
m_V = 1 * nu.keV
alpha_chi = 0.1

omega = m_V / m_chi
g_chi = mp.sqrt(4 * mp.pi * alpha_chi)
my_omega = omega / v_fid

sigma_0 = g_chi**4 / 4 / mp.pi / m_chi**2 / omega**4 / m_chi
my_sigma_0 = sigma_0 / sigma_fid

my_cs_type = "ruth"
if_brem = True
if_anni = False

brem_prefactor = g_chi**6 / m_chi**3 / 96 / mp.power(mp.pi, 7/2) / sigma_fid / v_fid**2
anni_prefactor = mp.sqrt(mp.pi) * alpha_chi**2  / 4 / m_chi**3 * rho_s * t_fid
# rho**2 * anni_prefactor * anni_int = dM / dV / dt.  rho, M, V(=r^2 dr) and t are all normalized by fid values.

def configure_point(*,
                    mchi_GeV: float,
                    mV_keV: float,
                    alpha: float,
                    mass_norm: float,
                    scale_norm: float,
                    do_brem: bool = None,
                    do_anni: bool = None,
                    nproc_inner: int = None,
                    cs_type: str = None):
    """Update globals for a single scan point.

    Inputs:
      - mchi_GeV:   m_chi in GeV
      - mV_keV:     m_V in keV
      - alpha:      alpha_chi
      - mass_norm:  Plummer mass_norm (dimensionless)
      - scale_norm: Plummer scale_norm (dimensionless)
    """
    global m_chi, m_V, alpha_chi
    global omega, g_chi, my_omega
    global sigma_0, my_sigma_0
    global my_mass_norm, my_scale_norm
    global if_brem, if_anni
    global my_cs_type
    global brem_prefactor, anni_prefactor
    global N_proc

    if nproc_inner is not None:
        N_proc = int(nproc_inner)

    if cs_type is not None:
        my_cs_type = str(cs_type)

    if do_brem is not None:
        if_brem = bool(do_brem)
    if do_anni is not None:
        if_anni = bool(do_anni)

    my_mass_norm  = mp.mpf(mass_norm)
    my_scale_norm = mp.mpf(scale_norm)

    m_chi = mp.mpf(mchi_GeV) * nu.GeV
    m_V   = mp.mpf(mV_keV) * nu.keV
    alpha_chi = mp.mpf(alpha)

    omega = m_V / m_chi
    g_chi = mp.sqrt(4 * mp.pi * alpha_chi)

    my_omega = omega / v_fid

    sigma_0 = g_chi**4 / 4 / mp.pi / m_chi**2 / omega**4 / m_chi
    my_sigma_0 = sigma_0 / sigma_fid

    brem_prefactor = g_chi**6 / m_chi**3 / 96 / mp.power(mp.pi, 7/2) / sigma_fid / v_fid**2
    anni_prefactor = mp.sqrt(mp.pi) * alpha_chi**2  / 4 / m_chi**3 * rho_s * t_fid


# ----------------------
# 1D Lagrangian zone parameters
# ----------------------
r_min = mp.mpf('0.005')
r_max = mp.mpf('1000.0')
layer = 160
extra_layer = 10

# simulation parameters
epsilon = 0.001
epsilonRho = 0.001
default_age_of_universe_in_gyr = 20
my_default_age_of_universe = default_age_of_universe_in_gyr * 1e9 * nu.year / t_fid


# ----------------------
# Dimensionless density and mass functions
# ----------------------
def density_dm(r):
    return 1/(r * (1+r)**2)

def mass_dm(r):
    return -r/(1+r) + mp.log(1+r)

def density_baryon(r, mass_norm, ars):
    return (3*mass_norm)/(ars**3) * (1+r**2/ars**2)**(-5/2)

def mass_baryon(r, mass_norm, ars):
    return mass_norm * (1 + ars**2 * r**(-2))**(-1.5)

def density_total(r, mass_norm, ars):
    return density_dm(r) + density_baryon(r, mass_norm, ars)

def mass_total(r, mass_norm, ars):
    return mass_dm(r) + mass_baryon(r, mass_norm, ars)

def vd_dm(r, mass_norm, ars):
    r, mass_norm, ars = mp.mpf(r), mp.mpf(mass_norm), mp.mpf(ars)

    term = -(1/2) * r * (
        (2 * mass_norm * (1+r) * (
            mp.sqrt(1+ars**2) * (
                -ars**4 - (1+r) * (-r + mp.sqrt(ars**2 + r**2)) +
                ars**2 * (2+r-2*r**2 + 2*mp.sqrt(ars**2 + r**2) + 2*r*mp.sqrt(ars**2 + r**2))
            ) -
            6 * ars**2 * (1+r) * mp.sqrt(ars**2 + r**2) * (
                mp.acoth(mp.sqrt(1+ars**2)) +
                mp.atanh((-1-r+mp.sqrt(ars**2 + r**2))/mp.sqrt(1+ars**2))
            )
        )) / (ars**2 * (1+ars**2)**(5/2) * mp.sqrt(ars**2 + r**2))
        +
        (2 + 9*r + 6*r**2 - 6*r * (1+r)**2 * mp.log(1+1/r)) / r
        +
        (1/r**2) * (1+r) * (
            -r * (1 + r*(-1 + mp.pi**2 * (1+r)) + 5*r*(1+r)*mp.log(r)) +
            (-1 + r*(3 + r*(11 + 5*r))) * mp.log(1+r) -
            3 * r**2 * (1+r) * mp.log(1+r)**2 -
            6 * r**2 * (1+r) * mp.polylog(2, -r)
        )
    )

    if mp.im(term) != 0 and abs(mp.im(term)) < 1e-10:
        term = mp.re(term)

    return mp.sqrt(term)

def big_dev(r, mass_norm, ars):
    r, mass_norm, ars = mp.mpf(r), mp.mpf(mass_norm), mp.mpf(ars)

    result = 1/2 * (
        -4 - (4*mass_norm)/(1+ars**2)**2 + (2*mass_norm)/(ars+ars**3)**2 + mp.pi**2 -
        4*r - (16*mass_norm*r)/(1+ars**2)**2 + (8*mass_norm*r)/(ars+ars**3)**2 + 4*mp.pi**2*r +
        5*r**2 - (12*mass_norm*r**2)/(1+ars**2)**2 + (6*mass_norm*r**2)/(ars+ars**3)**2 + 3*mp.pi**2*r**2 -
        8/(1+r) - (26*r)/(1+r) - (22*r**2)/(1+r) - (5*r**3)/(1+r) +

        (4*mass_norm*r**2)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) -
        (2*ars**2*mass_norm*r**2)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) +
        (6*mass_norm*r**3)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) -
        (2*ars**2*mass_norm*r**3)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) +
        (2*mass_norm*r**3)/((ars+ars**3)**2 * (ars**2+r**2)**(3/2)) -
        (2*mass_norm*r**4)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) +
        (4*mass_norm*r**4)/((ars+ars**3)**2 * (ars**2+r**2)**(3/2)) -
        (4*mass_norm*r**5)/((1+ars**2)**2 * (ars**2+r**2)**(3/2)) +
        (2*mass_norm*r**5)/((ars+ars**3)**2 * (ars**2+r**2)**(3/2)) -

        (4*mass_norm)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) +
        (2*ars**2*mass_norm)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) -
        (12*mass_norm*r)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) +
        (4*ars**2*mass_norm*r)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) -
        (4*mass_norm*r)/((ars+ars**3)**2 * mp.sqrt(ars**2+r**2)) +
        (6*mass_norm*r**2)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) -
        (12*mass_norm*r**2)/((ars+ars**3)**2 * mp.sqrt(ars**2+r**2)) +
        (16*mass_norm*r**3)/((1+ars**2)**2 * mp.sqrt(ars**2+r**2)) -
        (8*mass_norm*r**3)/((ars+ars**3)**2 * mp.sqrt(ars**2+r**2)) +

        1/(r+r**2) -
        (6*mass_norm*r)/((1+ars**2)**2 * (1+r) * (-r+mp.sqrt(ars**2+r**2))) -
        (12*mass_norm*r**2)/((1+ars**2)**2 * (1+r) * (-r+mp.sqrt(ars**2+r**2))) -
        (6*mass_norm*r**3)/((1+ars**2)**2 * (1+r) * (-r+mp.sqrt(ars**2+r**2))) +

        (6*mass_norm*r**2)/((1+ars**2)**2 * (1+r) * mp.sqrt(ars**2+r**2) * (-r+mp.sqrt(ars**2+r**2))) +
        (12*mass_norm*r**3)/((1+ars**2)**2 * (1+r) * mp.sqrt(ars**2+r**2) * (-r+mp.sqrt(ars**2+r**2))) +
        (6*mass_norm*r**4)/((1+ars**2)**2 * (1+r) * mp.sqrt(ars**2+r**2) * (-r+mp.sqrt(ars**2+r**2))) +

        (12*mass_norm * (1+4*r+3*r**2) * mp.acoth(mp.sqrt(1+ars**2)))/(1+ars**2)**(5/2) +
        (12*mass_norm * mp.atanh((-1-r+mp.sqrt(ars**2+r**2))/mp.sqrt(1+ars**2)))/(1+ars**2)**(5/2) +
        (48*mass_norm*r * mp.atanh((-1-r+mp.sqrt(ars**2+r**2))/mp.sqrt(1+ars**2)))/(1+ars**2)**(5/2) +
        (36*mass_norm*r**2 * mp.atanh((-1-r+mp.sqrt(ars**2+r**2))/mp.sqrt(1+ars**2)))/(1+ars**2)**(5/2) +

        6*mp.log(1+1/r) + 24*r*mp.log(1+1/r) + 18*r**2*mp.log(1+1/r) +
        5*mp.log(r) + 20*r*mp.log(r) + 15*r**2*mp.log(r) -
        20*mp.log(1+r) - mp.log(1+r)/r**2 - 38*r*mp.log(1+r) - 15*r**2*mp.log(1+r) +
        3*mp.log(1+r)**2 + 12*r*mp.log(1+r)**2 + 9*r**2*mp.log(1+r)**2 +
        6 * (1+4*r+3*r**2) * mp.polylog(2, -r)
    )

    return result

def log_space(start, stop, num):
    start_log = mp.log10(start)
    stop_log = mp.log10(stop)
    step = (stop_log - start_log) / (num - 1)
    return [mp.power(10, start_log + i * step) for i in range(num)]

def diff_cs_ruth(v, w, x):
    y = v**2 / w**2
    return 1 / 2 / (1 + y * (1 - x) / 2)**2

def diff_cs_moll(v, w, x):
    y = v**2 / w**2
    top  = (3 * x**2 + 1) * y**2 + 4 * y + 4
    down = ( (1 - x**2) * y**2 + 4 * y + 4 )**2
    return top / down

def tot_cs_ruth(v, w):
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    return 1 / (1+y)

def tot_cs_moll(v, w):
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    return 1 / (1 + y) - 1 / (y**2 + 2 * y) * mp.log(1 + y)

def I_ruth(v, w):
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    if abs(y) < mp.mpf(1e-4):
        return (mp.mpf(2/3)
                - mp.mpf(2/3) * y
                + mp.mpf(3/5) * y**2)
    return 4 * ((2 + y) * mp.log(1 + y) - 2 * y) / y**3

def I_moll(v, w):
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    if abs(y) < mp.mpf(1e-4):
        return (mp.mpf(1/3)
                - mp.mpf(1/3) * y
                + mp.mpf(1/3) * y**2
                - mp.mpf(1/3) * y**3)
    top = 2 * (2 * (y**2 + 5 * y + 5) * mp.log(1 + y) - 5 * (y**2 + 2 * y))
    down = y**3 * (2 + y)
    return top / down

def big_int(vd, w , cs_type):
    vd_mp = mp.mpf(vd)
    w_mp = mp.mpf(w)

    def integrand(x):
        x_mp = mp.mpf(x)
        v_rel = 2 * vd_mp * mp.sqrt(x_mp)
        if cs_type == "ruth":
            Isig = I_ruth(v_rel, w_mp)
        elif cs_type == "moll":
            Isig = I_moll(v_rel, w_mp)
        else:
            raise ValueError(f"Unknown cs_type '{cs_type}'. Use 'ruth' or 'moll'.")
        return x_mp**3 * Isig * mp.e**(-x_mp)

    integral_val = mp.quad(integrand, [0, mp.inf])
    return 128 * integral_val

def luminosity_dm(r, a, c, my_sigma_0, w, mass_norm, ars, cs_type):
    r_val = mp.mpf(r)

    density = density_dm(r_val)
    vd = vd_dm(r_val, mass_norm, ars)
    bd = big_dev(r_val, mass_norm, ars)
    bi = big_int(vd, w, cs_type=cs_type)
    smfp = 600 * mp.sqrt(mp.pi) * vd / my_sigma_0 / bi
    lmfp = 3 / 2 * a * c * density * vd**3 * my_sigma_0 * bi / 512
    return (-1) * r_val**2 * smfp * lmfp / (smfp + lmfp) * bd

def brem_int(vd):
    v_min_local = mp.sqrt(4 * m_V / m_chi) / v_fid
    zeta = v_min_local / vd
    def inner_int(t):
        aa = zeta / t
        x_max = mp.mpf('1.0')
        x_min = aa**2
        def f(x):
            return (1 + 0.5 * aa**4 / x**2) * mp.sqrt(1 - aa**4 / x**2) * 2 * mp.atanh(mp.sqrt(1 - x))
        return t**3 * mp.e**(-t**2 / 4) * mp.quad(f, [x_min, x_max])
    return mp.re(mp.quad(inner_int, [zeta, mp.inf]))

def brem_int_for_pool(vd):
    return float(brem_int(vd))

def anni_int(vd):
    def S(x):
        y = 2 * mp.pi * x
        return y / (1 - mp.e**(-y))
    def integrand(t):
        return t**2 * mp.e**(-t**2 / 4) * S(alpha_chi / t / vd / v_fid)
    return mp.re(mp.quad(integrand, [0, mp.inf]))

def anni_int_for_pool(vd):
    return float(anni_int(vd))

# Precompute integrals on a vd grid
_vd_min = mp.mpf('1e-2')
_vd_max = mp.mpf('1e2')
n_v_big = 200
vd_sample = log_space(_vd_min, _vd_max, n_v_big)

def precompute_brem_table(n_proc=N_proc):
    vd_sample_brem = [float(v) for v in vd_sample]
    if int(n_proc) <= 1:
        return [brem_int_for_pool(v) for v in vd_sample_brem]
    with Pool(processes=int(n_proc)) as pool:
        return pool.map(brem_int_for_pool, vd_sample_brem)

def precompute_anni_table(n_proc=N_proc):
    vd_sample_anni = [float(v) for v in vd_sample]
    if int(n_proc) <= 1:
        return [anni_int_for_pool(v) for v in vd_sample_anni]
    with Pool(processes=int(n_proc)) as pool:
        return pool.map(anni_int_for_pool, vd_sample_anni)

# Create radius lists
r_list1 = log_space(r_min, r_max, layer + extra_layer)
r_list2 = [r_list1[0] / 2] + [(r_list1[i-1] + r_list1[i]) / 2 for i in range(1, len(r_list1))]

def calculate_lists():
    m_list = [mass_dm(r) for r in r_list1]
    rho_list = [density_dm(r) for r in r_list2]
    vd_list = [vd_dm(r, my_mass_norm, my_scale_norm) for r in r_list2]
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    l_list = [luminosity_dm(r, a, c, my_sigma_0, my_omega, my_mass_norm, my_scale_norm, my_cs_type) for r in r_list1]

    # Truncate to extra layers
    r_list1_trunc = r_list1[:layer]
    r_list2_trunc = r_list2[:layer]
    m_list_trunc = m_list[:layer]
    rho_list_trunc = rho_list[:layer]
    u_list_trunc = u_list[:layer]
    vd_list_trunc = vd_list[:layer]
    l_list_trunc = l_list[:layer]

    kn_list_trunc = [(1/(tot_cs_ruth(vd_list[i], my_omega) * my_sigma_0 * rho_list[i])) /
                     mp.sqrt((2 * u_list[i]) / (3 * rho_list[i]))
                     for i in range(layer)]

    big_int_sample  = [big_int(vd, my_omega, cs_type=my_cs_type) for vd in vd_sample]

    def build_piecewise_interpolator(x_grid, y_grid):
        x = np.asarray(x_grid, dtype=float)
        y = np.asarray(y_grid, dtype=float)

        keep = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
        x = x[keep]
        y = y[keep]

        if x.size < 2:
            def _zero(_x):
                return 0.0
            return _zero

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        def y_of_x(xval):
            xv = float(xval)
            if (not np.isfinite(xv)) or (xv <= 0.0):
                return 0.0

            if xv <= x[0]:
                return float(y[0])
            if xv >= x[-1]:
                return float(y[-1])

            i1 = int(np.searchsorted(x, xv, side="right"))
            i0 = i1 - 1

            x0, x1 = x[i0], x[i1]
            y0, y1 = y[i0], y[i1]

            t_lin = (xv - x0) / (x1 - x0)

            if (y0 > 0.0) and (y1 > 0.0):
                lx  = np.log(xv)
                lx0 = np.log(x0)
                lx1 = np.log(x1)
                t   = (lx - lx0) / (lx1 - lx0)

                ly0 = np.log(y0)
                ly1 = np.log(y1)
                return float(np.exp(ly0 + t * (ly1 - ly0)))

            return float(y0 + t_lin * (y1 - y0))

        return y_of_x

    if if_brem:
        brem_int_sample = precompute_brem_table(N_proc)
        brem_from_table = build_piecewise_interpolator(vd_sample, brem_int_sample)

        def cooling_brem_from_table(r_val, mass_norm, ars):
            vd = vd_dm(r_val, mass_norm, ars)
            rho = density_dm(r_val)
            return brem_prefactor * rho**2 * vd * brem_from_table(vd)

        c_list = [cooling_brem_from_table(r, my_mass_norm, my_scale_norm) for r in r_list2]
    else:
        brem_int_sample = [0.0 for _ in vd_sample]
        c_list = [0.0 for _ in r_list2]

    if if_anni:
        anni_int_sample = precompute_anni_table(N_proc)
        anni_from_table = build_piecewise_interpolator(vd_sample, anni_int_sample)

        def annihilation_rate_from_table(r_val, mass_norm, ars):
            vd = vd_dm(r_val, mass_norm, ars)
            rho = density_dm(r_val)
            return rho**2 * anni_prefactor * anni_from_table(vd)

        anni_list = [annihilation_rate_from_table(r, my_mass_norm, my_scale_norm) for r in r_list2]
    else:
        anni_int_sample = [0.0 for _ in vd_sample]
        anni_list = [0.0 for _ in r_list2]

    return {
        'r_list1_trunc': r_list1_trunc,
        'r_list2_trunc': r_list2_trunc,
        'm_list_trunc': m_list_trunc,
        'rho_list_trunc': rho_list_trunc,
        'u_list_trunc': u_list_trunc,
        'vd_list_trunc': vd_list_trunc,
        'l_list_trunc': l_list_trunc,
        'kn_list_trunc': kn_list_trunc,
        'c_list_trunc': c_list[:layer],
        'anni_list_trunc': anni_list[:layer],
        'vd_sample': vd_sample,
        'big_int_sample': big_int_sample,
        'brem_int_sample': brem_int_sample,
        'anni_int_sample': anni_int_sample
    }

def plot_results(results):
    r1 = np.array([float(mp.re(r)) for r in results['r_list1_trunc']])
    r2 = np.array([float(mp.re(r)) for r in results['r_list2_trunc']])
    m = np.array([float(mp.re(val)) for val in results['m_list_trunc']])
    rho = np.array([float(mp.re(val)) for val in results['rho_list_trunc']])
    rho_b = np.array([float(mp.re(density_baryon(r, my_mass_norm, my_scale_norm))) for r in r2])
    vd = np.array([float(mp.re(val)) for val in results['vd_list_trunc']])
    lum = np.array([float(mp.re(val)) for val in results['l_list_trunc']])
    kn = np.array([float(mp.re(val)) for val in results['kn_list_trunc']])
    col = np.array([float(mp.re(val)) for val in results['c_list_trunc']])

    plt.figure(figsize=(10, 10))
    plt.loglog(r1, m, label=r'$M_{\chi}$')
    plt.loglog(r2, rho, label=r'$\rho_{\chi}$')
    plt.loglog(r2, rho_b, label=r'$\rho_{b}$')
    plt.loglog(r2, vd, label=r'$\nu_{\chi}$')
    plt.loglog(r1, lum, label=r'$L_{\chi}$')
    plt.loglog(r1, -lum, label=r'$-L_{\chi}$')
    plt.loglog(r2, kn, label=r'$Kn_{\chi}$')
    plt.loglog(r2, col, label=r'$C_{\chi}$')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.title('Red Dot Halo Initial Condition')
    plt.xlabel('r')
    plt.tight_layout()
    return plt

def export_data(results, tag=None):
    if tag is None:
        tag = datetime.now().strftime("%Y%m%d") + "A"

    os.makedirs(base_path, exist_ok=True)
    output_dir = os.path.join(base_path, tag)
    os.makedirs(output_dir, exist_ok=True)

    basic_info = [
        f"name = {tag}",
        "t = 0",
        "## Dimensionless parameters are for simulations ##",
        f"a = {a}",
        f"c = {c}",
        f"sigma_0 = {my_sigma_0}",
        f"omega = {my_omega}",
        "Initial dark matter profile = NFW",
        "Initial baryon profile = Plummer",
        f"rmin = {float(r_min)}",
        f"rmax = {float(r_max)}",
        f"Shell Number = {len(r_list1)}",
        f"Extra shell = {extra_layer}",
        f"baryon_Plummer_mass_norm = {float(my_mass_norm)}",
        f"baryon_Plummer_ars = {float(my_scale_norm)}",
        f"brem_prefactor = {float(brem_prefactor)}",
        f"anni_prefactor = {float(anni_prefactor)}",
        f"if_brem = {int(bool(if_brem))}",
        f"if_anni = {int(bool(if_anni))}",
        "## Age of Universe in fidutical time. Epsilon for |u| / u <= epsilon ##",
        f"default_age_of_universe = {float(my_default_age_of_universe)}",
        f"epsilon = {float(epsilon)}",
        f"epsilonRho = {float(epsilonRho)}",
        "## Dimensional parameters are for readout and presentations ##",
        f"r_s_in_nu = {float(r_s)}",
        f"rho_s_in_nu = {float(rho_s)}",
        f"m_chi_in_GeV = {float(m_chi / nu.GeV)}",
        f"m_V_in_MeV = {float(m_V / nu.MeV)}",
        f"alpha_chi = {float(alpha_chi)}"
    ]

    with open(os.path.join(output_dir, f"Basic-{tag}.txt"), 'w') as f:
        f.write('\n'.join(basic_info))

    _dbl_min_normal = np.finfo(float).tiny
    def _clamp_subnormal_to_zero(v):
        vv = float(mp.re(v))
        if 0.0 < abs(vv) < _dbl_min_normal:
            return 0.0
        return vv

    r_list_str = [f"{float(mp.re(r)):.10g}" for r in results['r_list1_trunc']] + ['']
    m_list_str = [f"{float(mp.re(m)):.10g}" for m in results['m_list_trunc']] + ['']
    rho_list_str = [f"{float(mp.re(rho)):.10g}" for rho in results['rho_list_trunc']] + ['']
    u_list_str = [f"{float(mp.re(u)):.10g}" for u in results['u_list_trunc']] + ['']
    l_list_str = [f"{float(mp.re(l)):.10g}" for l in results['l_list_trunc']] + ['']

    vd_sample_list_str = [f"{float(mp.re(vdi)):.10g}" for vdi in results['vd_sample']] + ['']
    big_int_list_str = [f"{float(mp.re(l)):.10g}" for l in results['big_int_sample']] + ['']
    brem_int_list_str = [f"{_clamp_subnormal_to_zero(l):.10g}" for l in results['brem_int_sample']] + ['']
    anni_int_list_str = [f"{_clamp_subnormal_to_zero(l):.10g}" for l in results['anni_int_sample']] + ['']

    c_list_str = [f"{_clamp_subnormal_to_zero(c):.10g}" for c in results['c_list_trunc']] + ['']
    anni_list_str = [f"{_clamp_subnormal_to_zero(a):.10g}" for a in results['anni_list_trunc']] + ['']

    def _write(name, lines):
        with open(os.path.join(output_dir, name), 'w') as f:
            f.write('\n'.join(lines))

    _write(f"RList-{tag}.txt", r_list_str)
    _write(f"MList-{tag}.txt", m_list_str)
    _write(f"RhoList-{tag}.txt", rho_list_str)
    _write(f"uList-{tag}.txt", u_list_str)
    _write(f"LList-{tag}.txt", l_list_str)
    _write(f"vdiList-{tag}.txt", vd_sample_list_str)
    _write(f"biList-{tag}.txt", big_int_list_str)
    _write(f"bmList-{tag}.txt", brem_int_list_str)
    _write(f"aiList-{tag}.txt", anni_int_list_str)
    _write(f"CList-{tag}.txt", c_list_str)
    _write(f"AnniList-{tag}.txt", anni_list_str)

    return output_dir

def _tag_from_point(i: int, *, mchi_GeV: float, mV_keV: float, alpha: float,
                    mass_norm: float, scale_norm: float, prefix: str = "scan"):
    # Intentionally exclude mass_norm / scale_norm from tag to keep directory names short.
    # Uniqueness is guaranteed by the running index i.
    return (f"{prefix}-{i:06d}-"
            f"mchi{mchi_GeV:.3g}GeV-"
            f"mV{mV_keV:.3g}keV-"
            f"a{alpha:.3g}")

def main():
    global base_path, my_tag

    parser = argparse.ArgumentParser(
        description="Generate gravothermal initial conditions (outer serial scan; inner Pool per scan point)."
    )
    parser.add_argument("--mode", choices=["single", "batch"], default="batch",
                        help="Default: batch (uses SCAN_GRIDS inside this file).")
    parser.add_argument("--base-path", default=base_path)
    parser.add_argument("--nproc", type=int, default=N_proc)

    # single-point overrides (optional)
    parser.add_argument("--tag", default=my_tag)
    parser.add_argument("--mchi", type=float, default=float(m_chi / nu.GeV))
    parser.add_argument("--mV",   type=float, default=float(m_V / nu.keV))
    parser.add_argument("--alpha", type=float, default=float(alpha_chi))

    parser.add_argument("--if-brem", type=int, default=int(bool(if_brem)))
    parser.add_argument("--if-anni", type=int, default=int(bool(if_anni)))
    parser.add_argument("--cs-type", choices=["ruth", "moll"], default=my_cs_type)

    args = parser.parse_args()

    base_path = args.base_path
    os.makedirs(base_path, exist_ok=True)

    do_brem = bool(int(args.if_brem))
    do_anni = bool(int(args.if_anni))

    if args.mode == "single":
        configure_point(mchi_GeV=args.mchi,
                        mV_keV=args.mV,
                        alpha=args.alpha,
                        mass_norm=float(my_mass_norm),
                        scale_norm=float(my_scale_norm),
                        do_brem=do_brem,
                        do_anni=do_anni,
                        nproc_inner=args.nproc,
                        cs_type=args.cs_type)

        tag = args.tag
        print(f"[single] generating tag={tag} (inner N_proc={N_proc})")
        results = calculate_lists()
        outdir = export_data(results, tag)

        # ALWAYS save profile.pdf
        plt_fig = plot_results(results)
        plt_fig.savefig(os.path.join(outdir, "profile.pdf"))

        print(f"[single] done -> {outdir}")
        return

    # batch mode: use in-file SCAN_GRIDS (no terminal typing for grids)
    mchi_list = SCAN_GRIDS["mchi_GeV"]
    mV_list   = SCAN_GRIDS["mV_keV"]
    a_list    = SCAN_GRIDS["alpha"]

    points = list(itertools.product(mchi_list, mV_list, a_list))
    n_total = len(points)
    print(f"[batch] total points = {n_total} (outer serial; inner N_proc={args.nproc})")

    for idx, (mchi_GeV, mV_keV, alpha) in enumerate(points, start=1):
        tag = _tag_from_point(idx, mchi_GeV=mchi_GeV, mV_keV=mV_keV, alpha=alpha,
                              mass_norm=float(my_mass_norm), scale_norm=float(my_scale_norm))

        print(f"[batch] ({idx}/{n_total}) generating {tag}")

        configure_point(mchi_GeV=mchi_GeV,
                        mV_keV=mV_keV,
                        alpha=alpha,
                        mass_norm=float(my_mass_norm),
                        scale_norm=float(my_scale_norm),
                        do_brem=do_brem,
                        do_anni=do_anni,
                        nproc_inner=args.nproc,
                        cs_type=args.cs_type)

        results = calculate_lists()
        outdir = export_data(results, tag)

        # ALWAYS save profile.pdf per point
        plt_fig = plot_results(results)
        plt_fig.savefig(os.path.join(outdir, "profile.pdf"))

    print("[batch] done")

if __name__ == "__main__":
    main()
