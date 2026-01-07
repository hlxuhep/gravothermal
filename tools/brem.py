import csv
import os
import time
from multiprocessing import Pool, cpu_count

import mpmath as mp
import numpy as np

import natural_units as nu

# ============================================================================
# Numerics
# ============================================================================
mp.mp.dps = 35

# ============================================================================
# Fiducial halo scales (same as your original file)
# ============================================================================
rho_s = 1.28e7 * nu.mSun / nu.kpc**3
r_s = 6.5 * nu.kpc

sigma_fid = 1 / rho_s / r_s
v_fid = mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s) * r_s

# ============================================================================
# Fixed evaluation point (user request)
# ============================================================================
R_EVAL = mp.mpf('0.5')
MASS_NORM = mp.mpf('0')
ARS = mp.mpf('1.0')

THRESHOLD = 1e-20

# Additional scan filter (user request): require sigma_eff in cm^2/g within this band.
SIGMA_EFF_MIN_CM2_PER_G = 0.1
SIGMA_EFF_MAX_CM2_PER_G = 100.0

# Default number of processes for multiprocessing scans.
# Override by passing n_proc=... to scan_parameter_space().
N_PROC_DEFAULT = 6


def _print_progress(prefix: str, i: int, n: int, t0: float, extra: str = ""):
    """Print a single-line progress update with ETA."""
    dt = max(1e-9, time.time() - t0)
    frac = i / max(1, n)
    rate = i / dt
    eta = (n - i) / rate if rate > 0 else float('inf')
    msg = f"{prefix} {i}/{n} ({frac:.1%}) elapsed {dt:.1f}s ETA {eta:.1f}s"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


# ============================================================================
# Halo functions (restore original analytic vd_dm)
#   r = r/r_s (dimensionless)
#   mass_norm = M_b/(4*pi*rho_s*r_s^3)
#   ars = a/r_s
# ============================================================================
def density_dm(r):
    """NFW density profile in units of rho_s."""
    r = mp.mpf(r)
    return 1 / (r * (1 + r) ** 2)


def vd_dm(r, mass_norm, ars):
    """Dark matter 1D velocity dispersion in units of v_fid.

    This is your original closed-form expression (kept essentially unchanged).
    """
    r, mass_norm, ars = mp.mpf(r), mp.mpf(mass_norm), mp.mpf(ars)

    term = -(mp.mpf('0.5')) * r * (
        (2 * mass_norm * (1 + r) * (
            mp.sqrt(1 + ars**2) * (
                -ars**4
                - (1 + r) * (-r + mp.sqrt(ars**2 + r**2))
                + ars**2 * (
                    2 + r - 2 * r**2
                    + 2 * mp.sqrt(ars**2 + r**2)
                    + 2 * r * mp.sqrt(ars**2 + r**2)
                )
            )
            - 6 * ars**2 * (1 + r) * mp.sqrt(ars**2 + r**2) * (
                mp.acoth(mp.sqrt(1 + ars**2))
                + mp.atanh(
                    (-1 - r + mp.sqrt(ars**2 + r**2)) / mp.sqrt(1 + ars**2)
                )
            )
        )) / (ars**2 * (1 + ars**2)**(mp.mpf('2.5')) * mp.sqrt(ars**2 + r**2))
        +
        (2 + 9 * r + 6 * r**2 - 6 * r * (1 + r)**2 * mp.log(1 + 1 / r)) / r
        +
        (1 / r**2) * (1 + r) * (
            -r * (1 + r * (-1 + mp.pi**2 * (1 + r)) + 5 * r * (1 + r) * mp.log(r))
            + (-1 + r * (3 + r * (11 + 5 * r))) * mp.log(1 + r)
            - 3 * r**2 * (1 + r) * mp.log(1 + r)**2
            - 6 * r**2 * (1 + r) * mp.polylog(2, -r)
        )
    )

    # Guard against tiny imaginary parts from numerical noise
    if mp.im(term) != 0 and abs(mp.im(term)) < mp.mpf('1e-10'):
        term = mp.re(term)

    return mp.sqrt(term)


# Cache halo quantities at the requested evaluation point.
RHO_EVAL = density_dm(R_EVAL)
VD_EVAL = vd_dm(R_EVAL, MASS_NORM, ARS)

# Units from your natural_units.py
CM = nu.cm
GRAM = nu.gram

# Characteristic relative speed for evaluating sigma_eff.
# We take v_rel,typ = sqrt(2) * v_1D, where v_1D = VD_EVAL * v_fid.
VREL_EVAL = mp.sqrt(2) * VD_EVAL * v_fid


def sigma_eff_cm2_per_g(m_chi_GeV: float, m_V_keV: float, alpha_chi: float) -> float:

    m_chi = mp.mpf(m_chi_GeV) * nu.GeV
    m_V = mp.mpf(m_V_keV) * nu.keV
    alpha = mp.mpf(alpha_chi)

    v = VREL_EVAL
    if v <= 0:
        return 0.0

    # Dimensionless w = m_V / m_chi
    w = m_V / m_chi
    if w <= 0:
        return 0.0

    # σ0 = 4π α^2 m_chi^2 / m_V^4
    sigma0 = 4 * mp.pi * alpha**2 * (m_chi**2) / (m_V**4)

    x = (v**2) / (w**2)  # = (m_chi v / m_V)^2

    bracket = (2 + x) * mp.log(1 + x) - 2 * x

    sigma_V = (6 * sigma0 * (w**6) / (v**6)) * bracket

    sigma_per_mass = sigma_V / m_chi

    # Convert to cm^2/g
    cm2_per_g_unit = (CM**2) / GRAM
    return float(sigma_per_mass / cm2_per_g_unit)

# ============================================================================
# Brem integral
#   In your original code: brem_int(vd) depends on v_min/vd where
#     v_min = sqrt(4 m_V/m_chi)/v_fid
#   So for a fixed halo point, it is a function of zeta = v_min / v_d only.
# ============================================================================

def brem_int_from_zeta(zeta: mp.mpf) -> mp.mpf:
    """Nested integral in your original brem_int, rewritten as I(zeta)."""
    zeta = mp.mpf(zeta)

    if zeta <= 0:
        return mp.mpf('0')

    def inner_int(t):
        a = zeta / t
        x_max = mp.mpf('1.0')
        x_min = a**2

        # If x_min >= 1, the phase space closes
        if x_min >= x_max:
            return mp.mpf('0')

        a4 = a**4

        def f(x):
            x = mp.mpf(x)
            inside = 1 - a4 / (x**2)
            if inside <= 0:
                return mp.mpf('0')
            return (1 + mp.mpf('0.5') * a4 / (x**2)) * mp.sqrt(inside) * 2 * mp.atanh(mp.sqrt(1 - x))

        return t**3 * mp.e**(-t**2 / 4) * mp.quad(f, [x_min, x_max])

    return mp.re(mp.quad(inner_int, [zeta, mp.inf]))


# --------------------------------------------------------------------------
# Tabulate I(zeta) on a log-grid.
# We store (log_zetas, logI) and do linear interpolation in log-log space.
# This design is multiprocessing-friendly (plain numpy arrays are picklable).
# --------------------------------------------------------------------------
def _logI_at_zeta_worker(zeta_float: float) -> float:
    """Worker: compute log I(zeta)."""
    mp.mp.dps = 35
    z = mp.mpf(zeta_float)
    I = brem_int_from_zeta(z)
    if I <= 0:
        return -1e300
    return float(mp.log(I))


def build_brem_table(zeta_min: float, zeta_max: float, n: int = 80, n_proc: int = 1):
    """Build a log-log table for I(zeta) over [zeta_min, zeta_max].

    Returns
    -------
    log_zetas : np.ndarray
        log(zeta) grid
    logI : np.ndarray
        log(I(zeta)) values on that grid
    """
    zeta_min = max(float(zeta_min), 1e-12)
    zeta_max = max(float(zeta_max), zeta_min * 10)

    zetas = np.logspace(np.log10(zeta_min), np.log10(zeta_max), int(n))
    log_zetas = np.log(zetas)

    n_total = len(zetas)
    step = max(1, n_total // 20)  # ~5% updates
    t0 = time.time()

    if n_proc is None or n_proc <= 1:
        logI_list = []
        for i, z in enumerate(zetas, start=1):
            logI_list.append(_logI_at_zeta_worker(float(z)))
            if i == 1 or i % step == 0 or i == n_total:
                _print_progress("[brem-table]", i, n_total, t0)
        logI = np.array(logI_list, dtype=float)
    else:
        logI_list = []
        with Pool(processes=int(n_proc)) as pool:
            for i, val in enumerate(pool.imap(_logI_at_zeta_worker, map(float, zetas), chunksize=4), start=1):
                logI_list.append(val)
                if i == 1 or i % step == 0 or i == n_total:
                    _print_progress("[brem-table]", i, n_total, t0, extra=f"n_proc={int(n_proc)}")
        logI = np.array(logI_list, dtype=float)

    return log_zetas, logI


def I_of_zeta_from_table(zeta: float, log_zetas: np.ndarray, logI: np.ndarray) -> float:
    """Evaluate I(zeta) using linear interpolation in log-log space.

    We extrapolate linearly using the first/last two points if outside the table.
    """
    if zeta <= 0:
        return 0.0

    x = float(np.log(float(zeta)))

    # Left extrapolation
    if x <= log_zetas[0]:
        x0, x1 = log_zetas[0], log_zetas[1]
        y0, y1 = logI[0], logI[1]
        y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return float(np.exp(y))

    # Right extrapolation
    if x >= log_zetas[-1]:
        x0, x1 = log_zetas[-2], log_zetas[-1]
        y0, y1 = logI[-2], logI[-1]
        y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return float(np.exp(y))

    # In-range interpolation
    y = float(np.interp(x, log_zetas, logI))
    return float(np.exp(y))


def cooling_brem_fixed(
    m_chi_GeV: float,
    m_V_keV: float,
    alpha_chi: float,
    log_zetas: np.ndarray,
    logI: np.ndarray,
) -> float:
    """Compute cooling_brem(0.1, 0, 1.0) for given (m_chi, m_V, alpha_chi).

    Units:
      m_chi_GeV : GeV
      m_V_keV   : keV
      alpha_chi : dimensionless

    Output:
      Same dimensionless normalization as your original cooling_brem.
    """
    m_chi = mp.mpf(m_chi_GeV) * nu.GeV
    m_V = mp.mpf(m_V_keV) * nu.keV
    alpha = mp.mpf(alpha_chi)

    g_chi = mp.sqrt(4 * mp.pi * alpha)

    # Same prefactor as in your original script:
    # brem_prefactor = g_chi^6 / m_chi^3 / (96 * pi^(7/2)) / sigma_fid / v_fid^2
    brem_prefactor = g_chi**6 / m_chi**3 / 96 / mp.power(mp.pi, mp.mpf('3.5')) / sigma_fid / v_fid**2

    v_min = mp.sqrt(4 * m_V / m_chi) / v_fid
    zeta = float(v_min / VD_EVAL)

    I = I_of_zeta_from_table(zeta, log_zetas, logI)
    return float(brem_prefactor * (RHO_EVAL**2) * VD_EVAL * mp.mpf(I))


# Globals used inside multiprocessing workers (set by _init_scan_worker).
_G_LOG_ZETAS = None
_G_LOGI = None


def _init_scan_worker(log_zetas: np.ndarray, logI: np.ndarray):
    """Initializer for multiprocessing scan workers."""
    global _G_LOG_ZETAS, _G_LOGI
    mp.mp.dps = 35
    _G_LOG_ZETAS = log_zetas
    _G_LOGI = logI


def _cooling_worker(args):
    """Worker for scan: args = (mchi, mV, alpha).

    Returns
    -------
    (mchi, mV, alpha, cooling, sigma_eff_cm2_per_g)
    """
    mchi, mV, a = args

    sig = sigma_eff_cm2_per_g(float(mchi), float(mV), float(a))
    val = cooling_brem_fixed(float(mchi), float(mV), float(a), _G_LOG_ZETAS, _G_LOGI)

    return float(mchi), float(mV), float(a), float(val), float(sig)


# ============================================================================
# Scan driver
# ============================================================================
def scan_parameter_space(
    mchi_range=(1, 1e6), n_mchi=25,
    mV_range=(1e-6, 1e3), n_mV=37,
    alpha_range=(1e-4, 1.0), n_alpha=21,
    threshold=THRESHOLD,
    out_csv="scan_brem_pass.csv",
    n_interp=80,
    n_proc=N_PROC_DEFAULT,
):
    """Scan (m_chi, m_V, alpha) and record all points with cooling_brem_fixed > threshold."""

    mchi_min, mchi_max = mchi_range
    mV_min, mV_max = mV_range

    # zeta = (sqrt(4 mV/mchi)/v_fid) / VD_EVAL
    # so zeta ranges like sqrt(mV/mchi)
    zeta_min = float(mp.sqrt(4 * (mp.mpf(mV_min) * nu.keV) / (mp.mpf(mchi_max) * nu.GeV)) / v_fid / VD_EVAL)
    zeta_max = float(mp.sqrt(4 * (mp.mpf(mV_max) * nu.keV) / (mp.mpf(mchi_min) * nu.GeV)) / v_fid / VD_EVAL)

    log_zetas, logI = build_brem_table(zeta_min, zeta_max, n=n_interp, n_proc=min(int(n_proc), int(n_interp)))

    mchi_grid = np.logspace(np.log10(mchi_min), np.log10(mchi_max), n_mchi)
    mV_grid = np.logspace(np.log10(mV_min), np.log10(mV_max), n_mV)
    alpha_grid = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alpha)

    points = [(mchi, mV, a) for mchi in mchi_grid for mV in mV_grid for a in alpha_grid]

    n_total = len(points)
    step = max(1, n_total // 50)  # ~2% updates
    t0_scan = time.time()
    print(f"[scan] total points = {n_total} | n_proc = {int(n_proc) if n_proc is not None else 1}", flush=True)

    passed = []

    if n_proc is None or int(n_proc) <= 1:
        for i, p in enumerate(points, start=1):
            mchi, mV, a, val, sig = _cooling_worker(p)
            if (SIGMA_EFF_MIN_CM2_PER_G <= sig <= SIGMA_EFF_MAX_CM2_PER_G) and (val > threshold):
                passed.append((mchi, mV, a, val, sig))
            if i == 1 or i % step == 0 or i == n_total:
                _print_progress("[scan]", i, n_total, t0_scan, extra=f"passed={len(passed)}")
    else:
        n_proc_eff = max(1, int(n_proc))
        chunksize = max(50, n_total // (20 * n_proc_eff))
        done = 0
        with Pool(
            processes=n_proc_eff,
            initializer=_init_scan_worker,
            initargs=(log_zetas, logI),
        ) as pool:
            for mchi, mV, a, val, sig in pool.imap_unordered(_cooling_worker, points, chunksize=chunksize):
                done += 1
                if (SIGMA_EFF_MIN_CM2_PER_G <= sig <= SIGMA_EFF_MAX_CM2_PER_G) and (val > threshold):
                    passed.append((mchi, mV, a, val, sig))
                if done == 1 or done % step == 0 or done == n_total:
                    _print_progress("[scan]", done, n_total, t0_scan, extra=f"passed={len(passed)} | n_proc={n_proc_eff}")

    # Save all passing points
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m_chi_GeV", "m_V_keV", "alpha_chi", "cooling_brem_r0p1", "sigma_eff_cm2_per_g"])
        w.writerows(passed)

    # Summaries (min/max over PASSING points)
    summary = {
        "N_pass": len(passed),
        "threshold": float(threshold),
        "mchi_range_scanned": (float(mchi_min), float(mchi_max)),
        "mV_range_scanned": (float(mV_min), float(mV_max)),
        "alpha_range_scanned": (float(alpha_range[0]), float(alpha_range[1])),
    }

    if passed:
        arr = np.array(passed)
        summary.update({
            "mchi_pass_minmax": (float(arr[:, 0].min()), float(arr[:, 0].max())),
            "mV_pass_minmax": (float(arr[:, 1].min()), float(arr[:, 1].max())),
            "alpha_pass_minmax": (float(arr[:, 2].min()), float(arr[:, 2].max())),
            "cooling_pass_minmax": (float(arr[:, 3].min()), float(arr[:, 3].max())),
            "sigma_eff_pass_minmax": (float(arr[:, 4].min()), float(arr[:, 4].max())),
            # mV/mchi (convert keV->GeV with 1 keV = 1e-6 GeV)
            "ratio_mV_over_mchi_pass_minmax": (
                float((arr[:, 1] * 1e-6 / arr[:, 0]).min()),
                float((arr[:, 1] * 1e-6 / arr[:, 0]).max()),
            ),
        })

    return summary


if __name__ == "__main__":
    print("[info] Fixed point: cooling_brem(r=0.1, mass_norm=0, ars=1.0)")
    print(f"[info] RHO_EVAL = {RHO_EVAL}")
    print(f"[info] VD_EVAL  = {VD_EVAL}")
    print(f"[info] N_PROC_DEFAULT = {N_PROC_DEFAULT}")

    summary = scan_parameter_space(n_proc=N_PROC_DEFAULT)

    print("\n[scan summary]")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary.get("N_pass", 0) == 0:
        print("\n[warning] No points passed the threshold in the current scan box.")
        print("          Enlarge ranges or lower THRESHOLD.")
    else:
        print("\n[output] Saved passing points to: scan_brem_pass.csv")