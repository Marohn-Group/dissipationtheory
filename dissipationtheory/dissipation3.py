# Author: John A. Marohn
# Date: 2024-10-12
# Summary: Merge dissipation and dissipation2, rewriting the code in dissipation in terms of Loring's unitless K integral.

import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from dissipationtheory.capacitance import Csphere
from dissipationtheory.capacitance import CsphereOverSemi
from scipy import integrate
from numba import jit
from numba import float64, complex128, boolean
from numba.experimental import jitclass
from numba import deferred_type

class CantileverModel(object):
    """Cantilever object.  
    SI units for each parameter are indicated below.
    A parameter can be given in any equivalent unit using the units package ``pint``.
    For example, :: 
    
        from dissipationtheory.constants import ureg
        R = ureg.Quantity(50., 'nm')
    
    :param ureg.Quantity f_c: cantilever frequency [Hz] 
    :param ureg.Quantity k_c: cantilever spring constant [N/m]
    :param ureg.Quantity V_ts: tip-sample voltage [V]
    :param ureg.Quantity R: tip radius [m]
    :param ureg.Quantity d: tip-sample separation [m]

    """
    def __init__(self, f_c, k_c, V_ts, R, d):

        self.f_c = f_c
        self.k_c = k_c
        self.V_ts = V_ts
        self.R = R
        self.d = d

    @property
    def omega_c(self):
        """Cantilever resonance frequency in units of radians/second."""
        return 2 * np.pi * self.f_c

    def __repr__(self):

        str = 'cantilever\n\n'
        str = str + '      resonance freq = {:.3f} kHz\n'.format(self.f_c.to('kHz').magnitude)
        str = str + '                     = {:.3e} rad/s\n'.format(self.omega_c.to('Hz').magnitude)
        str = str + '     spring constant = {:.3f} N/m\n'.format(self.k_c.to('N/m').magnitude)
        str = str + '  tip-sample voltage = {:.3f} V\n'.format(self.V_ts.to('V').magnitude)
        str = str + '              radius = {:.3f} nm\n'.format(self.R.to('nm').magnitude)
        str = str + '              height = {:.3f} nm\n'.format(self.d.to('nm').magnitude)
        
        return str
    
class SampleModel1(object):
    """Model I sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | semiconductor | dielectric
    
    The dielectric substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_s: semiconductor layer's thickness [m]
    :param ureg.Quantity sigma: semiconductor layer's conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity z_r: reference height [m] (used in computations)
    """

    def __init__(self, cantilever, h_s, epsilon_s, sigma, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.sigma = sigma
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

    @property
    def omega0(self):
        """Omega0, the roll-off frequency."""
        return (self.sigma / epsilon0).to('Hz')

    @property
    def mu(self):
        """Mobility."""
        return (self.sigma / (qe * self.rho)).to('m^2/(V s)')

    @property
    def D(self):
        """Diffusion constant."""
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        """Diffusion length."""
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        """Debye length."""
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        """Effective relative complex dielectric constant."""
        return (self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2).to_base_units()

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\nsemiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = {:.1f} nm\n'.format(self.h_s.to('nm').magnitude)
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.3e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.3e} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '         roll-off frequency = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)
        str = str + '                   mobility = {:0.3e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.3e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)                
        str = str + '               Debye length = {:0.3e} nm\n'.format(self.LD.to('nm').magnitude)
        str = str + '           diffusion length = {:0.3e} nm\n'.format(self.Ld.to('nm').magnitude)
        str = str + '   effective epsilon (real) = {:0.3f}\n'.format(self.epsilon_eff.to('').real.magnitude)
        str = str + '   effective epsilon (imag) = {:0.3f}\n\n'.format(self.epsilon_eff.to('').imag.magnitude)
        str = str + 'dielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = infinite'
        
        return str

class SampleModel2(object):
    """Model II sample object defined in Lekkala2013nov, Fig. 1(b)::
    
        cantilever | vacuum gap | dielectric | semiconductor 
    
    The semiconductor substrate is semi-infinite.
    
    :param CantileverModel cantilever: an object storing the cantilever properties, including the tip-sample separation, i.e. the vacuum-gap thickness
    :param ureg.Quantity epsilon_d: dielectric layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity h_d: dielectric layer's thickness [m]
    :param ureg.Quantity epsilon_s: semiconductor layer's complex relative dielectric constant [unitless]
    :param ureg.Quantity mu: semiconductor layer's charge conductivity [S/m]
    :param ureg.Quantity rho: semiconductor layer's charge density [1/m^3]
    :param ureg.Quantity z_r: reference height [m] (used in computations)
    """

    def __init__(self, cantilever, epsilon_d, h_d, epsilon_s, sigma, rho, z_r):

        self.cantilever = cantilever
        self.epsilon_d = epsilon_d
        self.h_d = h_d
        self.epsilon_s = epsilon_s
        self.sigma = sigma
        self.rho = rho
        self.z_r = z_r

    @property
    def omega0(self):
        """Omega0, the roll-off frequency."""
        return (self.sigma / epsilon0).to('Hz')

    @property
    def mu(self):
        """Mobility."""
        return (self.sigma / (qe * self.rho)).to('m^2/(V s)')

    @property
    def D(self):
        """Diffusion constant."""
        return ((kb * ureg.Quantity(300., 'K') * self.mu) / qe).to('m^2/s')

    @property
    def Ld(self):
        """Diffusion length."""
        return (np.sqrt(self.D / self.cantilever.omega_c)).to('nm')
    
    @property
    def LD(self):
        """Debye length."""
        return (np.sqrt((epsilon0 * kb * ureg.Quantity(300., 'K'))/(self.rho * qe * qe))).to('nm')

    @property
    def epsilon_eff(self):
        """Effective relative complex dielectric constant."""
        return (self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2).to_base_units()

    def __repr__(self):

        str = self.cantilever.__repr__()
        str = str + '\ndielectric\n\n'
        str = str + '  epsilon (real) = {:0.3f}\n'.format(self.epsilon_d.to('').real.magnitude)
        str = str + '  epsilon (imag) = {:0.3f}\n'.format(self.epsilon_d.to('').imag.magnitude)
        str = str + '       thickness = {:.1f} nm\n\n'.format(self.h_d.to('nm').magnitude)
        str = str + 'semiconductor\n\n'
        str = str + '             epsilon (real) = {:0.3f}\n'.format(self.epsilon_s.to('').real.magnitude)
        str = str + '             epsilon (imag) = {:0.3f}\n'.format(self.epsilon_s.to('').imag.magnitude)
        str = str + '                  thickness = infinite\n'
        str = str + '               conductivity = {:0.3e} S/m\n'.format(self.sigma.to('S/m').magnitude)
        str = str + '             charge density = {:0.2e} m^{{-3}}\n'.format(self.rho.to('1/m^3').magnitude)
        str = str + '           reference height = {:0.1f} nm\n'.format(self.z_r.to('nm').magnitude)
        str = str + '\n'
        str = str + '         roll-off frequency = {:0.3e} Hz\n'.format(self.omega0.to('Hz').magnitude)        
        str = str + '                   mobility = {:0.2e} m^2/(V s)\n'.format(self.mu.to('m^2/(V s)').magnitude)
        str = str + '         diffusion constant = {:0.2e} m^2/s\n'.format(self.D.to('m^2/s').magnitude)
        str = str + '               Debye length = {:0.1f} nm\n'.format(self.LD.to('nm').magnitude)
        str = str + '           diffusion length = {:0.1f} nm\n'.format(self.Ld.to('nm').magnitude)
        str = str + '   effective epsilon (real) = {:0.3f}\n'.format(self.epsilon_eff.to('').real.magnitude)
        str = str + '   effective epsilon (imag) = {:0.3f}\n\n'.format(self.epsilon_eff.to('').imag.magnitude)

        return str
    
def mysech(x):
    """Define my own ``sech()`` function to avoid overflow problems."""

    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.cosh(x[mask])
    
    return values

def mycsch(x):
    """Define my own ``csch()`` function to avoid overflow problems."""

    x = np.array(x)
    mask = abs(x.real) < 710.4
    values = np.zeros_like(x, dtype=complex)
    values[mask] = 1/np.sinh(x[mask])
    
    return values

def theta1norm(psi, sample, omega, power, part=np.imag):
    """Theta function for Sample geometry I as defined by Loring, rewritten in unitless variables by Marohn.
    In the code below, `psi` is the unitless variable of integration."""

    # Recompute Ld and epsilon_eff appropriate for the new frequency omega

    Ld = np.sqrt(sample.D / omega).to('nm')
    epsilon_eff = (sample.epsilon_s - complex(0,1) * Ld**2 / sample.LD**2).to_base_units()

    r1 = Ld**2 / (sample.epsilon_s * sample.LD**2)
    r2 = sample.z_r**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.z_r**2 / Ld**2
    lambduh = (complex(0,1) * r1 * psi / np.sqrt(psi**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude
     
    khs = (psi * sample.h_s / sample.z_r).to('dimensionless').magnitude

    r1 = sample.h_s**2 / sample.z_r**2
    r2 = sample.h_s**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.h_s**2 / Ld**2
    etahs = (np.sqrt(r1 * psi**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude

    alpha = (epsilon_eff/sample.epsilon_d).to('dimensionless').magnitude
    
    r1 = 1/epsilon_eff
    t1 = - lambduh / np.tanh(etahs)
    t2 = np.tanh(khs) * np.tanh(etahs) \
         + alpha * np.tanh(etahs) \
         - lambduh \
         + 2 * lambduh * mysech(khs) * mysech(etahs) \
         - lambduh**2 * np.tanh(khs) * mysech(etahs) * mycsch(etahs)
    t3 = np.tanh(etahs) \
         + np.tanh(khs) * (-1 * lambduh + alpha * np.tanh(etahs))

    theta = (r1 * (t1 + t2 / t3)).to('dimensionless').magnitude

    rp = (1 - theta) / (1 + theta)
    exponent = (2 * sample.cantilever.d / sample.z_r).to('dimensionless').magnitude

    integrand = psi**power * np.exp(-1 * psi * exponent) * part(rp)

    return integrand

def theta2norm(psi, sample, omega, power, part=np.imag):
    """Theta function for Sample II geometry as defined by Loring, rewritten in unitless variables by Marohn.
    In the code below, `psi` is the unitless variable of integration."""

    # Recompute Ld and epsilon_eff appropriate for the new frequency omega

    Ld = np.sqrt(sample.D / omega).to('nm')
    epsilon_eff = (sample.epsilon_s - complex(0,1) * Ld**2 / sample.LD**2).to_base_units()

    r1 = Ld**2 / (sample.epsilon_s * sample.LD**2)
    r2 = sample.z_r**2 / (sample.epsilon_s * sample.LD**2)
    r3 = sample.z_r**2 / Ld**2
    lambduh = (complex(0,1) * r1 * psi / np.sqrt(psi**2 + r2 + complex(0,1) * r3)).to('dimensionless').magnitude
     
    khd = (psi * sample.h_d / sample.z_r).to('dimensionless').magnitude

    r1 = 1/sample.epsilon_d
    t1 = epsilon_eff * np.tanh(khd) + (1 - lambduh) * sample.epsilon_d
    t2 = epsilon_eff + (1 - lambduh) * sample.epsilon_d * np.tanh(khd)

    theta = ((r1 * t1) / t2).magnitude

    rp = (1 - theta) / (1 + theta)
    exponent = (2 * sample.cantilever.d / sample.z_r).to('dimensionless').magnitude

    integrand = psi**power * np.exp(-1 * psi * exponent) * part(rp) 

    return integrand

def K(power, theta, sample, omega, part=np.imag):
    """Compute the integral :math:`K`."""
    
    prefactor = 1/np.power(sample.z_r, power+1)    
    integral = integrate.quad(theta, 0., np.inf, args=(sample, omega, power, part))[0]
    
    return (prefactor * integral).to_base_units()


def gamma_perpendicular(theta, sample):
    """Compute the friction coefficient for a cantilever oscillating in the perpendicular geometry."""

    prefactor = -sample.cantilever.V_ts**2 / (4 * np.pi * epsilon0 * sample.cantilever.omega_c)
    omega_c = sample.cantilever.omega_c

    # Use sample.epsilon_s.real, not sample.epsilon_d.real, to compute capacitance

    c0 = CsphereOverSemi(0, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)        
    c1 = CsphereOverSemi(1, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)
    
    # Return an array, with units, consisting of the three terms contributing to the total.
    # Remove the units, create a list, convert the list to an array, and add back in the units.

    result =  [( prefactor *     c0 * c0 * K(2, theta, sample, omega_c, np.imag)).to('pN s/m').magnitude,
               (-prefactor * 2 * c0 * c1 * K(1, theta, sample, omega_c, np.imag)).to('pN s/m').magnitude,
               ( prefactor *     c1 * c1 * K(0, theta, sample, omega_c, np.imag)).to('pN s/m').magnitude]
    
    return ureg.Quantity(np.ravel(np.array(result)), 'pN s/m')

def blds_perpendicular(theta, sample, omega_m):
    """Compute the BLDS frequency-shift signal for a cantilever oscillating in the 
    perpendicular geometry.  You are always going to want to compute a BLDS 
    frequency-shift *spectrum*, so assume omega_m is a `numpy` array, with units, 
    and loop over omega_m values. For example,
    
        omega_m = ureg.Quantity(2 * np.pi * np.logspace(1, 3, 3), 'Hz').
    
    """

    prefactor = -(sample.cantilever.f_c * sample.cantilever.V_ts**2)/(16 * np.pi * epsilon0 * sample.cantilever.k_c)

    # Use sample.epsilon_s.real, not sample.epsilon_d.real, to compute capacitance

    c0 = CsphereOverSemi(0, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)        
    c1 = CsphereOverSemi(1, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)
    c2 = CsphereOverSemi(2, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)
     
    result = ureg.Quantity(np.zeros((len(omega_m), 3)), 'Hz')
    for index, omega in enumerate(omega_m):

        # Return an array, with units, consisting of the three terms contributing to the total.
        # Remove the units, create a list, convert the list to an array, and add back in the units.

        result_list = [( prefactor     * c0 * c0 * K(2, theta, sample, omega, np.real)).to('Hz').magnitude,
                       (-prefactor * 2 * c0 * c1 * K(1, theta, sample, omega, np.real)).to('Hz').magnitude,
                       ( prefactor *     c0 * c2 * K(0, theta, sample, omega, np.real)).to('Hz').magnitude]
    
        result[index,:] = ureg.Quantity(np.ravel(np.array(result_list)), 'Hz')
        
    return result

def BLDSpre(sample):
    """The prefactor in Loring's approximation expressions for the BLDS spectrum."""

    # Use sample.epsilon_s.real, not sample.epsilon_d.real, to compute capacitance

    c0 = CsphereOverSemi(0, sample.cantilever.d * np.ones(1), sample.cantilever.R, sample.epsilon_s.real.magnitude)   

    a = sample.cantilever.f_c * c0**2 * sample.cantilever.V_ts**2
    b = 16 * np.pi * epsilon0 * sample.cantilever.k_c * sample.cantilever.d**3

    return (a/b).to('Hz')

def BLDSlimits(sample, x):
    """Loring's expansions for the BLDS frequency shift are carried out in terms of a unitless parameter 
    :math:`x \equiv \left( \frac{h}{\lambda_{\mathrm{D}}} \right)^2` with with :math:`h` the tip-sample 
    separation and :math:`\lambda_{\mathrm{D}}` the Debye length.  Given an array of :math:`x`-values,
    return an array of values for the low-density (i.e., small :math:`x`) and high-density (i.e., 
    large :math:`x`) limiting values of the absolute value of the BLDS frequency shift."""

    pre = BLDSpre(sample)
    er = sample.epsilon_s.real.magnitude
    a = 0.25 * (er - 1)/(er + 1)
    ones = np.ones(len(x))
    
    return [pre * a * ones, pre * 0.25 * ones]

def BLDSapprox(sample, x):
    """Make a function to return Loring's low-density approximation for the BLDS frequency shift.  
    The dependent variable here is :math:`x = (h/\lambda_{\mathrm{D}})^2`.  The approximation is of 
    the form :math:`c_1 + c_2 x` with :math:`c_1` and :mathr:`c_2` constants.  The approximation
    increases without bound at large :math:`x`.  Only return values where the approximate BLDS
    frequency shift is less than the high-frequency limit.  Return a copy of the $x$ array where
    this is true; this truncated $x$ array will be handy for plotting."""

    pre = BLDSpre(sample)
    _, BLDS_high = BLDSlimits(sample, x)
    
    er = sample.epsilon_s.real.magnitude
    a = 0.25 * (er - 1)/(er + 1)
    b = 0.50 * er / (er + 1)**2

    BLDS_approx = pre * (a + b * x)
    mask = BLDS_approx.to('Hz').magnitude <= BLDS_high.to('Hz').magnitude
    
    return [x[mask], BLDS_approx[mask]]