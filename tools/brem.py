import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import natural_units as nu
from numpy.polynomial.laguerre import laggauss
from multiprocessing import Pool
from scipy.interpolate import interp1d
# Set precision 
mp.mp.dps = 25
# Number of cores
N_proc = 6

# Physical values with dimension
# '_fid' parameters are in natural units, 'my_' parameters are remormalized by fids.
rho_s     = 2.74e8 * nu.mSun / nu.kpc**3
r_s       = 0.141 * nu.kpc
sigma_fid = 1 / rho_s / r_s
v_fid     = mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s) * r_s
lumi_fid  = mp.power(4 * mp.pi * rho_s * r_s**2, 5/2) * mp.power(nu.G_Newton, 3/2)
t_fid     = 1 / mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s)
C_fid     = mp.power(4 * mp.pi * nu.G_Newton, 3/2) * mp.power(rho_s, 5/2) * r_s**2

# The following are all the velocity-dependent parameters.
# MODEL PARAMETERS FOR THE INPUT!
m_chi = 1e3 * nu.GeV     # DM mass
m_V = 1 * nu.keV    # mediator mass
alpha_chi = 1e-3
# Induced Equations
omega = m_V / m_chi    # mass ratio
g_chi = mp.sqrt(4 * mp.pi * alpha_chi)          # coupling constant
# omega as a velocity also needs to be converted
my_omega = omega / v_fid
# sigma_0 takes a 1/m to be in the form of sigma/m like SIDM strength
sigma_0 = g_chi**4 / 4 / mp.pi / m_chi**2 / omega**4 / m_chi
my_sigma_0 = sigma_0 / sigma_fid
# sigma_1 is g_chi^4/m_chi^3 - which is in similar form of sigma_0.
sigma_1 = g_chi**4 / m_chi**3
my_sigma_1 = sigma_1 / sigma_fid
my_cs_type = "ruth"
if_brem = True
if_anni = False

brem_prefactor = g_chi**6 / m_chi**3 / 96 / mp.power(mp.pi, 7/2) / sigma_fid / v_fid**2    # Need 1/v_fid**2 to balance the fiducial values.

# Create logarithmically spaced radius points
def log_space(start, stop, num):
    """Create logarithmically spaced points similar to Mathematica's Subdivide"""
    start_log = mp.log10(start)
    stop_log = mp.log10(stop)
    step = (stop_log - start_log) / (num - 1)
    return [mp.power(10, start_log + i * step) for i in range(num)]

# We are in place to define particle physics functions.
# differential cross section only takes the dimensionless velocity and angular terms, without the sigma at front.
def diff_cs_ruth(v, w, x): # v for velocity (renormalized), x for cos\theta
    y = v**2 / w**2
    return 1 / 2 / (1 + y * (1 - x) / 2)**2

def diff_cs_moll(v, w, x):
    y = v**2 / w**2
    top  = (3 * x**2 + 1) * y**2 + 4 * y + 4
    down = ( (1 - x**2) * y**2 + 4 * y + 4 )**2
    return top / down

def tot_cs_ruth(v, w): # total cross section but without sigma at front
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    return 1 / (1+y)

def tot_cs_moll(v, w):
    v_mp = mp.mpf(v)
    w_mp = mp.mpf(w)
    y = v_mp**2 / w_mp**2
    return 1 / (1 + y) - 1 / (y**2 + 2 * y) * mp.log(1 + y)

def I_ruth(v, w): # angular integral of cross section with weight of sin^2(theta)
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
    
    # 用 mp.quad 在 [0, ∞) 上积分
    integral_val = mp.quad(integrand, [0, mp.inf])
    return 128 * integral_val

def brem_int(x):
    zeta = mp.sqrt(4 * x)
    def inner_int(t):
        a = zeta / t
        x_max = mp.mpf('1.0')
        x_min = a**2
        def f(x):
            return (1 + 0.5 * a**4 / x**2) * mp.sqrt(1 - a**4 / x**2) * 2 * mp.atanh(mp.sqrt(1 - x))
        return t**3 * mp.e**(-t**2 / 4) * mp.quad(f, [x_min, x_max])
    return mp.re(mp.quad(inner_int, [zeta, mp.inf]))

def brem_int_for_pool(x):
    # 单独封装一层, 方便序列化
    return float(brem_int(x))

# Precompute big_int and brem_int on a velocity grid:
# vd in units of v_fid, ranging from 1e-2 * v_fid to 1e2 * v_fid (dimensionless 1e-2 to 1e2)
v_min = mp.mpf('1e-2')
v_max = mp.mpf('1e2')
n_v_big = 200  # number of sample points for big_int(vd)

# vdi = vd for big integral
vd_sample = log_space(v_min, v_max, n_v_big)

def precompute_brem_table(n_proc = N_proc):
    # 转成普通 float，避免 pickling mpmath 对象太重
    vd_sample_brem = [float(v) for v in vd_sample]

    with Pool(processes=n_proc) as pool:
        results = pool.map(brem_int_for_pool, vd_sample_brem)
    return results

# -----------------------------
# Fig.7-style ratio plot: brem(x)/brem(0)
# -----------------------------

def _logspace_float(xmin, xmax, n):
    """log-spaced floats from xmin to xmax (both > 0)."""
    return np.logspace(np.log10(xmin), np.log10(xmax), n)


def compute_brem_ratio_table(
    x_min=1e-2,
    x_max=3.0,
    n_x=120,
    x0_for_brem0=1e-12,
    n_proc=N_proc,
    cache_path=None,
):
    """Compute R(x)=brem(x)/brem(0) on a log grid.

    Notes
    -----
    - x = m_V/T. x=0 cannot be used on a log axis, and brem_int(x) is numerically delicate at x=0.
      We approximate brem(0) by evaluating at a very small x0_for_brem0.
    - If cache_path is provided, saves/loads npz with x_grid and ratio.
    """

    # Load cache if present
    if cache_path is not None and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["x_grid"], data["ratio"], float(data["brem0"])

    x_grid = _logspace_float(x_min, x_max, n_x)

    # brem(0) proxy
    brem0 = float(brem_int(x0_for_brem0))
    if not np.isfinite(brem0) or brem0 <= 0.0:
        raise RuntimeError(f"brem0 evaluation failed: brem_int({x0_for_brem0}) = {brem0}")

    # Parallel evaluation on x_grid
    with Pool(processes=n_proc) as pool:
        bvals = np.array(pool.map(brem_int_for_pool, x_grid.astype(float)), dtype=float)

    ratio = bvals / brem0

    if cache_path is not None:
        np.savez(cache_path, x_grid=x_grid, ratio=ratio, brem0=brem0)

    return x_grid, ratio, brem0


def plot_brem_ratio_fig7_style(
    x_grid,
    ratio,
    out_path="brem_ratio_fig7_style.png",
    title=None,
):
    """Make a Fig.7 (upper-left) style plot: log-log, x in [1e-2, ~3], y in [1e-2, 1]."""

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e-2, 3.0)
    ax.set_ylim(1e-2, 1.05)

    # Orange shaded kinematic suppression region (Fig.7 uses a shaded band at high m/T)
    ax.axvspan(1.0, 3.0, alpha=0.18)

    ax.plot(x_grid, ratio, lw=2.0, label=r"$R_{\rm brem}=\mathrm{brem}(x)/\mathrm{brem}(0)$")

    ax.set_xlabel(r"$m_{\phi,V}/T$")
    ax.set_ylabel(r"$R$")

    if title is not None:
        ax.set_title(title)

    ax.legend(frameon=False, loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return fig, ax


if __name__ == "__main__":
    # Default: reproduce the *shape/range/style* of Fig.7 upper-left panel for the ratio R(x).
    # Uses a small-x proxy for brem(0).
    cache = "brem_ratio_cache.npz"
    xg, Rg, b0 = compute_brem_ratio_table(
        x_min=1e-2,
        x_max=3.0,
        n_x=120,
        x0_for_brem0=1e-12,
        n_proc=N_proc,
        cache_path=cache,
    )
    print(f"brem0(proxy) = brem_int(1e-12) = {b0:.6e}")
    plot_brem_ratio_fig7_style(xg, Rg, out_path="brem_ratio_fig7_style.png")
    plot_brem_ratio_fig7_style(xg, Rg, out_path="brem_ratio_fig7_style.pdf")