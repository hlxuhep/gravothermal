import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tools import natural_units as nu
from numpy.polynomial.laguerre import laggauss
from multiprocessing import Pool
from scipy.interpolate import interp1d
# Set precision 
mp.mp.dps = 25
# Number of cores
N_proc = 6

# ----------------------
# USER CONFIGURATION
# ----------------------
# Absolute path where output directories should be created
# Change this to your desired output location
base_path = "./test"

# Output name
my_tag = "2205.03392.fig5bremtest"

# Physical values with dimension
# '_fid' parameters are in natural units, 'my_' parameters are remormalized by fids.
rho_s     = 2.74e8 * nu.mSun / nu.kpc**3
r_s       = 0.141 * nu.kpc
sigma_fid = 1 / rho_s / r_s
v_fid     = mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s) * r_s
lumi_fid  = mp.power(4 * mp.pi * rho_s * r_s**2, 5/2) * mp.power(nu.G_Newton, 3/2)
t_fid     = 1 / mp.sqrt(4 * mp.pi * nu.G_Newton * rho_s)
C_fid     = mp.power(4 * mp.pi * nu.G_Newton, 3/2) * mp.power(rho_s, 5/2) * r_s**2

# Model parameters
# a,b,c are the parameters for the SIDM conductivity terms
a = mp.mpf('2.257')
# b = mp.mpf('1.385')
c = mp.mpf('0.75')
# my_mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)
my_mass_norm = mp.mpf('0.0')
# my_scale_norm is the normalized baryon scale radius, a/r_s
my_scale_norm = mp.mpf('0.1')

# The following are all the velocity-dependent parameters.
# MODEL PARAMETERS FOR THE INPUT!
m_chi = 9.7 * nu.GeV     # DM mass
m_V = 32.0 * nu.keV    # mediator mass
alpha_chi = 1e-6
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

# 1D Lagragian zone parameters
r_min = mp.mpf('0.005')  # default 10^-4
r_max = mp.mpf('1000.0')  # default 10^2
layer = 160
# extra_layers are added to the end of the list to ensure a smooth 1D velocity dispersion profile
extra_layer = 10

# simulation parameters
epsilon = 0.001   # ε = max(|delta u / u|)
default_age_of_universe_in_gyr = 20   # simulation time limit in gyr
my_default_age_of_universe = default_age_of_universe_in_gyr * 1e9 * nu.year / t_fid  # renormalized


#----------------------
# Define all the dimensionless density and mass functions
# ----------------------
def density_dm(r):
    """Dark matter density function, assuming a NFW profile"""
    return 1/(r * (1+r)**2)

def mass_dm(r):
    """Dark matter mass function, assuming a NFW profile"""
    return -r/(1+r) + mp.log(1+r)

def density_baryon(r, mass_norm, ars):
    """Baryon density function, assuming a Plummer profile"""
    """mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)"""
    """ars is the normalized scale radius, a/r_s"""
    return (3*mass_norm)/(ars**3) * (1+r**2/ars**2)**(-5/2)

def mass_baryon(r, mass_norm, ars):
    """Baryon mass function, assuming a Plummer profile"""
    return mass_norm * (1 + ars**2 * r**(-2))**(-1.5)

def density_total(r, mass_norm, ars):
    """Total density function"""
    return density_dm(r) + density_baryon(r, mass_norm, ars)

def mass_total(r, mass_norm, ars):
    """Total enclosed mass function"""
    return mass_dm(r) + mass_baryon(r, mass_norm, ars)

def vd_dm(r, mass_norm, ars):
    """Dark matter 1D velocity dispersion"""
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
    
    # Ensure we get a real result (in case of small numerical errors)
    if mp.im(term) != 0 and abs(mp.im(term)) < 1e-10:
        term = mp.re(term)
    
    result = mp.sqrt(term)
    return result

def big_dev(r, mass_norm, ars):
    """deviation function of partial vd_dm^2/partial r"""
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

# Velocity-Dependent conductivity and lumonosity
def luminosity_dm(r, a, c, my_sigma_0, w, mass_norm, ars, cs_type):
    """Dark matter luminosity function"""
    r_val = mp.mpf(r)
    
    density = density_dm(r_val)
    vd = vd_dm(r_val, mass_norm, ars)
    bd = big_dev(r_val, mass_norm, ars)
    bi = big_int(vd, w, cs_type=cs_type)
    smfp = 600 * mp.sqrt(mp.pi) * vd / my_sigma_0 / bi
    lmfp = 3 / 2 * a * c * density * vd**3 * my_sigma_0 * bi / 512
    return (-1) * r_val**2 * smfp * lmfp / (smfp + lmfp) * bd     # DON'T FORGET THE MINUS SIGN AND THE R SQUARE!!!

def brem_int(vd):
    v_min = mp.sqrt(4 * m_V / m_chi) / v_fid
    zeta = v_min / vd
    def inner_int(t):
        a = zeta / t
        x_max = mp.mpf('1.0')
        x_min = a**2
        def f(x):
            return (1 + 0.5 * a**4 / x**2) * mp.sqrt(1 - a**4 / x**2) * 2 * mp.atanh(mp.sqrt(1 - x))
        return t**3 * mp.e**(-t**2 / 4) * mp.quad(f, [x_min, x_max])
    return mp.re(mp.quad(inner_int, [zeta, mp.inf]))

def cooling_brem(r_val, mass_norm, ars):   # all in fidutical values
    vd = vd_dm(r_val, mass_norm, ars)
    rho = density_dm(r_val)
    return brem_prefactor * rho**2 * vd * brem_int(vd)

def brem_int_for_pool(vd):
    # 单独封装一层, 方便序列化
    return float(brem_int(vd))

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

# Create radius lists
# r_list1 is based on the radius range and number of layers
r_list1 = log_space(r_min, r_max, layer + extra_layer)
# r_list2 is based on the average of adjacent points in r_list1
r_list2 = [r_list1[0] / 2] + [(r_list1[i-1] + r_list1[i]) / 2 for i in range(1, len(r_list1))]

# Calculate lists
def calculate_lists():
    # Calculate the dark matter enclosed mass
    m_list = [mass_dm(r) for r in r_list1]
    # Calculate the dark matter density
    rho_list = [density_dm(r) for r in r_list2]
    # Calculate the dark matter 1D velocity dispersion
    vd_list = [vd_dm(r, my_mass_norm, my_scale_norm) for r in r_list2]
    # Calculate specific kinetic energy 
    u_list = [mp.mpf('1.5') * mp.re(v)**2 for v in vd_list]
    # Calculate the dark matter luminosity
    l_list = [luminosity_dm(r, a, c, my_sigma_0, my_omega, my_mass_norm, my_scale_norm, my_cs_type) for r in r_list1]
    
    # Truncate to extra layers
    r_list1_trunc = r_list1[:layer]
    r_list2_trunc = r_list2[:layer]
    m_list_trunc = m_list[:layer]
    rho_list_trunc = rho_list[:layer]
    u_list_trunc = u_list[:layer]
    vd_list_trunc = vd_list[:layer]
    l_list_trunc = l_list[:layer]
    
    # Calculate Knudsen number
    kn_list_trunc = [(1/(tot_cs_ruth(vd_list[i], my_omega) * my_sigma_0 * rho_list[i])) / 
                     mp.sqrt((2 * u_list[i]) / (3 * rho_list[i])) 
                     for i in range(layer)]

    # Precompute big_int and brem_int on a velocity grid:

    big_int_sample  = [big_int(vd, my_omega, cs_type=my_cs_type) for vd in vd_sample]

    if if_brem:
        # tabulate brems 2D 积分（得到的是 python float 列表）
        brem_int_sample = precompute_brem_table()

        def build_piecewise_interpolator(x_grid, y_grid, *, name="table"):
            x = np.asarray(x_grid, dtype=float)
            y = np.asarray(y_grid, dtype=float)

            # Keep only finite points with x>0; do not filter on y sign.
            keep = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
            x = x[keep]
            y = y[keep]

            if x.size < 2:
                def _zero(_x):
                    return 0.0
                return _zero

            # Ensure ascending x
            order = np.argsort(x)
            x = x[order]
            y = y[order]

            def y_of_x(xval):
                xv = float(xval)
                if (not np.isfinite(xv)) or (xv <= 0.0):
                    return 0.0

                # Clamp to endpoints
                if xv <= x[0]:
                    return float(y[0])
                if xv >= x[-1]:
                    return float(y[-1])

                # Bracket index: x[i0] <= xv < x[i1]
                i1 = int(np.searchsorted(x, xv, side="right"))
                i0 = i1 - 1

                x0, x1 = x[i0], x[i1]
                y0, y1 = y[i0], y[i1]

                # Linear fallback (always defined if x0!=x1)
                t_lin = (xv - x0) / (x1 - x0)

                # Log–log only if both endpoints positive
                if (y0 > 0.0) and (y1 > 0.0) and (x0 > 0.0) and (x1 > 0.0):
                    lx  = np.log(xv)
                    lx0 = np.log(x0)
                    lx1 = np.log(x1)
                    t   = (lx - lx0) / (lx1 - lx0)

                    ly0 = np.log(y0)
                    ly1 = np.log(y1)
                    return float(np.exp(ly0 + t * (ly1 - ly0)))

                return float(y0 + t_lin * (y1 - y0))

            return y_of_x
    
        brem_from_table = build_piecewise_interpolator(vd_sample, brem_int_sample, name="brem_int")
        def cooling_brem_from_table(r_val, mass_norm, ars):
            vd = vd_dm(r_val, mass_norm, ars)
            rho = density_dm(r_val)
            return brem_prefactor * rho**2 * vd * brem_from_table(vd)

        c_list = [cooling_brem_from_table(r, my_mass_norm, my_scale_norm) for r in r_list2]
        
    else:
        # no bremsstrahlung cooling: use zero tables with the same shape
        brem_int_sample = [0.0 for _ in vd_sample]
        c_list = [0.0 for r in r_list2]

    c_list_trunc = c_list[:layer]
    return {
        'r_list1_trunc': r_list1_trunc,
        'r_list2_trunc': r_list2_trunc,
        'm_list_trunc': m_list_trunc,
        'rho_list_trunc': rho_list_trunc,
        'u_list_trunc': u_list_trunc,
        'vd_list_trunc': vd_list_trunc,
        'l_list_trunc': l_list_trunc,
        'kn_list_trunc': kn_list_trunc,
        'c_list_trunc': c_list_trunc,
        'vd_sample': vd_sample,
        'big_int_sample': big_int_sample,
        'brem_int_sample': brem_int_sample
    }

def plot_results(results):
    """Plot the results similar to the Mathematica plot"""
    # Convert mpmath objects to numpy floats for plotting (handling complex numbers)
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

def export_data(results, my_tag=None):
    """Export data to files similar to the Mathematica export"""
    if my_tag is None:
        my_tag = datetime.now().strftime("%Y%m%d") + "A"
    
    # Use the base_path defined at the top of the script
    # First check if "initial" folder exists, create it if not
    # initial_dir = os.path.join(base_path, "initial")
    initial_dir = base_path
    os.makedirs(initial_dir, exist_ok=True)
    
    # Create the date directory inside the initial folder
    output_dir = os.path.join(initial_dir, my_tag)
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic parameters with default formatting
    basic_info = [
        f"name = {my_tag}",
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
        f"if_brem = {int(bool(if_brem))}",
        f"if_anni = {int(bool(if_anni))}",
        "## Age of Universe in fidutical time. Epsilon for |u| / u <= epsilon ##",
        f"default_age_of_universe = {float(my_default_age_of_universe)}",
        f"epsilon = {float(epsilon)}",
        "## Dimensional parameters are for readout and presentations ##",
        f"r_s_in_nu = {float(r_s)}",  # in natural units
        f"rho_s_in_nu = {float(rho_s)}", # in natural units
        f"m_chi_in_GeV = {float(m_chi / nu.GeV)}",
        f"m_V_in_MeV = {float(m_V / nu.MeV)}",
        f"alpha_chi = {float(alpha_chi)}"
    ]
    
    # Write basic info
    basic_file = os.path.join(output_dir, f"Basic-{my_tag}.txt")
    with open(basic_file, 'w') as f:
        f.write('\n'.join(basic_info))
    
    # NOTE: values smaller than the smallest *normal* double (DBL_MIN ~ 2.225e-308)
    # can cause `std::istringstream >> double` to fail on some macOS/libc++ setups.
    # We clamp such subnormal nonzero values to 0.0 for exported lookup tables.
    _dbl_min_normal = np.finfo(float).tiny  # ~= 2.2250738585072014e-308
    def _clamp_subnormal_to_zero(v):
        vv = float(mp.re(v))
        if 0.0 < abs(vv) < _dbl_min_normal:
            return 0.0
        return vv
    # Convert mpmath values to strings for export with 10 effective digits
    # Use mp.re to extract real part if values are complex
    r_list_str = [f"{float(mp.re(r)):.10g}" for r in results['r_list1_trunc']] + ['']
    m_list_str = [f"{float(mp.re(m)):.10g}" for m in results['m_list_trunc']] + ['']
    rho_list_str = [f"{float(mp.re(rho)):.10g}" for rho in results['rho_list_trunc']] + ['']
    u_list_str = [f"{float(mp.re(u)):.10g}" for u in results['u_list_trunc']] + ['']
    l_list_str = [f"{float(mp.re(l)):.10g}" for l in results['l_list_trunc']] + ['']

    # big integral (big_int) and brem integral (brem_int) calculated at vd points:
    vd_sample_list_str = [f"{float(mp.re(vdi)):.10g}" for vdi in results['vd_sample']] + ['']
    big_int_list_str = [f"{float(mp.re(l)):.10g}" for l in results['big_int_sample']] + ['']
    brem_int_list_str = [f"{_clamp_subnormal_to_zero(l):.10g}" for l in results['brem_int_sample']] + ['']

    c_list_str = [f"{_clamp_subnormal_to_zero(c):.10g}" for c in results['c_list_trunc']] + ['']
    
    # Write data files with full paths
    r_file = os.path.join(output_dir, f"RList-{my_tag}.txt")
    with open(r_file, 'w') as f:
        f.write('\n'.join(r_list_str))
    
    m_file = os.path.join(output_dir, f"MList-{my_tag}.txt")
    with open(m_file, 'w') as f:
        f.write('\n'.join(m_list_str))
    
    rho_file = os.path.join(output_dir, f"RhoList-{my_tag}.txt")
    with open(rho_file, 'w') as f:
        f.write('\n'.join(rho_list_str))
    
    u_file = os.path.join(output_dir, f"uList-{my_tag}.txt")
    with open(u_file, 'w') as f:
        f.write('\n'.join(u_list_str))
    
    l_file = os.path.join(output_dir, f"LList-{my_tag}.txt")
    with open(l_file, 'w') as f:
        f.write('\n'.join(l_list_str))

    vdi_file = os.path.join(output_dir, f"vdiList-{my_tag}.txt")
    with open(vdi_file, 'w') as f:
        f.write('\n'.join(vd_sample_list_str))

    big_int_file = os.path.join(output_dir, f"biList-{my_tag}.txt")
    with open(big_int_file, 'w') as f:
        f.write('\n'.join(big_int_list_str))

    brem_int_file = os.path.join(output_dir, f"bmList-{my_tag}.txt")
    with open(brem_int_file, 'w') as f:
        f.write('\n'.join(brem_int_list_str))

    c_file = os.path.join(output_dir, f"CList-{my_tag}.txt")
    with open(c_file, 'w') as f:
        f.write('\n'.join(c_list_str))
    
    return output_dir

# Main execution
if __name__ == "__main__":
    # Calculate all lists
    results = calculate_lists()
    
    # Export data to files and get the output directory path
    output_dir = export_data(results, my_tag)
    
    # Plot and save figure as PDF
    plt_fig = plot_results(results)
    # Save PDF to the same directory as other files
    pdf_path = os.path.join(output_dir, "profile.pdf")
    plt_fig.savefig(pdf_path)

    
    print(f"Calculation and export completed successfully.")
