# Author: John A. Marohn
#
# Date: 2025-06-25
# Summary: Rewrite functions in dissipation9b.py for incorporation into fast, compiled matrix-generating functions
# See the exploratory code in dissipation-theory--Study-53.ipynb and dissipation-theory--Study-54.ipynb,
#
# Date: 2025-07-01
# Summary: Most of code in this file will be superceded by the code in dissipation9c.py, developed in 
# dissipation-theory--Study-58.ipynb

import numpy as np
from numba import jit, njit, prange
from dissipationtheory.dissipation9b import Kmetal_jit
import scipy

def disentangle(temp):
    """Convert a complex 3-vector into a real 6-vector"""

    return np.array([temp[0].real, temp[0].imag,
                 temp[1].real, temp[1].imag,
                 temp[2].real, temp[2].imag])

def compare(a,b):
    """Compare two numpy vectors a and b."""
    epsilon_rel = (a-b) / a
    epsilon_pct = 100 * (a-b) / a
    print("  reference  value      relative percent ")
    print("  ========== ========== ======== ========")
    for k, (a_, b_, epsilon_rel_, epsilon_pct_) in enumerate(zip(a, b, epsilon_rel, epsilon_pct)):
        print("{:d} {:+1.7f} {:+1.7f} {:+1.1e} {:+1.3f}".format(k, a_, b_, epsilon_rel_, epsilon_pct_))

@jit(nopython=True)
def Cmatrix_jit(Rj, Rk):
    """The untiless Coulomb potential Green's function matrix."""

    result = np.zeros((len(Rk),len(Rj)))
    for k, rk in enumerate(Rk):
        for j, rj in enumerate(Rj):
            result[k,j] = 1 / np.linalg.norm(rj - rk)
    return result

@jit(nopython=True)
def KmatrixIV_jit(sample, rj, rk):
    """The Green's function matrices for an image charge."""

    K0mat = complex(1,0) * np.zeros((len(rk),len(rj)))
    K1mat = complex(1,0) * np.zeros((len(rk),len(rj)))
    K2mat = complex(1,0) * np.zeros((len(rk),len(rj)))
    
    for k, rke in enumerate(rk):
        for j, rje in enumerate(rj):
            K0mat[k,j], K1mat[k,j], K2mat[k,j] = Kmetal_jit(sample, rje, rke)
            
    return K0mat, K1mat, K2mat

def inputs3(sample3, y, omega, rj, rk):
    """A function that makes a dictionary of inputs for integrand3jitfast."""
    pars1 = {'y': y, 'es': sample3.epsilon_s, 'zr': sample3.z_r, 'omega0': sample3.omega0}
    pars2 = {'kD': sample3.kD, 'omega': omega, 'location1': rj, 'location2': rk}
    return {**pars1, **pars2}   

@jit(nopython=True)
def integrand3jitfast(y, es, zr, omega0, kD, omega, location1, location2):
    """Integrand for a Sample III object, a semi-infinite dielectric.
    
    In the code below, `y` is the unitless integration variable.
    """

    Omega = omega/omega0
    k_over_eta = y / np.sqrt(y**2 + (zr * kD)**2 * (1/es + complex(0,1) * Omega))

    p0 = 1 + complex(0,1) * es * Omega
    p1 = k_over_eta / (es * p0)
    p6 = complex(0,1) * Omega / p0

    theta_norm = p6 + p1
    rp = (1 - theta_norm) / (1 + theta_norm)

    rhoX = (location1[0] - location2[0])/ zr
    rhoY = (location1[1] - location2[1])/ zr
    argument = y * np.sqrt(rhoX**2 + rhoY**2)
    exponent = y * (location1[2] + location2[2])/ zr

    integrand = np.array([       np.real(rp),        np.imag(rp), 
                          y    * np.real(rp), y    * np.imag(rp),
                          y**2 * np.real(rp), y**2 * np.imag(rp)]) * scipy.special.j0(argument) * np.exp(-1 * exponent)

    return integrand

def inputsIII(sample3, omega, rj, rk, N=30):
    """A function that makes a dictionary of inputs for KmatrixIII_jit."""
    pars1 = {'es': sample3.epsilon_s, 'zr': sample3.z_r, 'omega0': sample3.omega0, 'kD': sample3.kD}
    an, _ = scipy.integrate.newton_cotes(N, 1)
    pars2 = {'omega': omega, 'rj': rj, 'rk': rk, 'N': N, 'an': an}
    return {**pars1, **pars2}

@njit(parallel=True)
def KmatrixIII_jit(es, zr, omega0, kD, omega, rj, rk, N, an):
    """The Green's function matrices for a Type III sample."""

    ymax = 30.    
    y = np.linspace(0., ymax, N+1)
    dy = ymax / N

    K0mat = complex(1,0) * np.zeros((len(rk),len(rj),len(y)))
    K1mat = complex(1,0) * np.zeros((len(rk),len(rj),len(y)))
    K2mat = complex(1,0) * np.zeros((len(rk),len(rj),len(y)))

    K0matANS = complex(1,0) * np.zeros((len(rk),len(rj)))
    K1matANS = complex(1,0) * np.zeros((len(rk),len(rj)))
    K2matANS = complex(1,0) * np.zeros((len(rk),len(rj)))
    
    for k in prange(len(rk)):
        rke = rk[k]
        for j in prange(len(rj)):
            rje = rj[j]
            for m, y_ in enumerate(y):
                
                result = integrand3jitfast(y_, es, zr, omega0, kD, omega, rje, rke)

                K0mat[k,j,m] = complex(result[0], result[1])
                K1mat[k,j,m] = complex(result[2], result[3])
                K2mat[k,j,m] = complex(result[4], result[5])

    K0matANS = dy * (an * K0mat).sum(axis=2)
    K1matANS = dy * (an * K1mat).sum(axis=2)
    K2matANS = dy * (an * K2mat).sum(axis=2)
            
    return K0matANS, K1matANS, K2matANS