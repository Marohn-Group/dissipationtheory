       # Author: John A. Marohn
# Date: 2024-10-12
# Summary: Rwrite the code in dissipation3 so that the tip charge is located at the center of the tip sphere.

import numpy as np
import cmath
from dissipationtheory.constants import ureg, epsilon0, qe, kb
from scipy import integrate
from numba import jit
from numba import float64, complex128, boolean
from numba.experimental import jitclass
from numba import deferred_type
import pandas as pd

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
    :param ureg.Quantity R: tip radius [m], used to compute the tip charge
    :param ureg.Quantity angle: tip cone half angle [degrees], used in numerical model to compute tip charge
    :param ureg.Quantity d: tip-sample separation [m], used to compute the tip charge
    :param ureg.Quantity z_c: tip-sample separation [m], location of the tip charge, typically set to zc = d + R

    """
    def __init__(self, f_c, k_c, V_ts, R, angle, d, z_c):

        self.f_c = f_c
        self.k_c = k_c
        self.V_ts = V_ts
        self.R = R
        self.angle = angle
        self.d = d
        self.z_c = z_c

    @property
    def omega_c(self):
        """Cantilever resonance frequency in units of radians/second."""
        return 2 * np.pi * self.f_c

    def __repr__(self):

        str = 'cantilever\n\n'
        str = str + '         resonance freq = {:.3f} kHz\n'.format(self.f_c.to('kHz').magnitude)
        str = str + '                        = {:.3e} rad/s\n'.format(self.omega_c.to('Hz').magnitude)
        str = str + '        spring constant = {:.3f} N/m\n'.format(self.k_c.to('N/m').magnitude)
        str = str + '     tip-sample voltage = {:.3f} V\n'.format(self.V_ts.to('V').magnitude)
        str = str + '                 radius = {:.3f} nm\n'.format(self.R.to('nm').magnitude)
        str = str + '        cone half angle = {:.3f} degree\n'.format(self.angle.to('degree').magnitude)
        str = str + '                 height = {:.3f} nm\n'.format(self.d.to('nm').magnitude)
        str = str + '  tip charge z location = {:.3f} nm\n'.format(self.z_c.to('nm').magnitude)
        
        return str
    
    def args(self):
        """A dictionary with the cantilever parameters, useful for initializing 
        a jit version of the cantilever object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'), 
                d = ureg.Quantity(38, 'nm'),
                z_c = ureg.Quantity(73, 'nm')
            )
            cantilever_jit = CantileverModelJit(**cantilever.args())

        """

        args = {
            'f_c': self.f_c.to('Hz').magnitude, 
            'k_c': self.k_c.to('N/m').magnitude, 
            'V_ts': self.V_ts.to('V').magnitude, 
            'R': self.R.to('m').magnitude,
            'angle': self.angle.to('degree').magnitude,
            'd': self.d.to('m').magnitude,
            'z_c': self.z_c.to('m').magnitude
        }

        return args

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
        str = str + '                  thickness = {:0.1f} nm\n'.format(self.h_s.to('nm').magnitude)
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

    def args(self):
        """A dictionary with the sample parameters, useful for initializing 
        a jit version of the sample object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'), 
                d = ureg.Quantity(38, 'nm'),
                z_c = ureg.Quantity(73, 'nm')
            )

            sample = SampleModel1(
                cantilever = cantilever,
                h_s = ureg.Quantity(500, 'nm'),
                epsilon_s = ureg.Quantity(complex(20, -0.2), ''),
                sigma = ureg.Quantity(1E-5, 'S/m'),
                rho = ureg.Quantity(1e21, '1/m^3'),
                epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
                z_r = ureg.Quantity(300, 'nm')
            )

            sample_jit = SampleModel1Jit(**sample.args())

        """

        cantilever_jit = CantileverModelJit(**self.cantilever.args())

        args = {
            'cantilever': cantilever_jit,
            'h_s': self.h_s.to('m').magnitude,
            'epsilon_s': self.epsilon_s.to('').magnitude,
            'sigma': self.sigma.to('S/m').magnitude,
            'rho': self.rho.to('1/m^3').magnitude,
            'epsilon_d': self.epsilon_d.to('').magnitude,
            'z_r': self.z_r.to('m').magnitude
        }

        return args

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
        str = str + '       thickness = {:0.1f} nm\n\n'.format(self.h_d.to('nm').magnitude)
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

    def args(self):
        """A dictionary with the sample parameters, useful for initializing 
        a jit version of the sample object:
        
            cantilever = CantileverModel(
                f_c = ureg.Quantity(75, 'kHz'),
                k_c = ureg.Quantity(2.8, 'N/m'), 
                V_ts = ureg.Quantity(1, 'V'), 
                R = ureg.Quantity(35, 'nm'),
                angle = ureg.Quantity(20, 'degree'), 
                d = ureg.Quantity(38, 'nm'),
                z_c = ureg.Quantity(73, 'nm')
            )

            sample2 = SampleModel2(
                cantilever = cantilever,
                epsilon_d = ureg.Quantity(complex(20, -0.2), ''),
                h_d = ureg.Quantity(1, 'nm'),
                epsilon_s = ureg.Quantity(complex(20, -0.2), ''),
                sigma = ureg.Quantity(1E-5, 'S/m'),
                rho = ureg.Quantity(1e21, '1/m^3'),
                z_r = ureg.Quantity(300, 'nm')
            )            

            sample_jit = SampleModel2Jit(**sample.args())

        """

        cantilever_jit = CantileverModelJit(**self.cantilever.args())

        args = {
            'cantilever': cantilever_jit,
            'epsilon_d': self.epsilon_d.to('').magnitude,
            'h_d': self.h_d.to('m').magnitude,
            'epsilon_s': self.epsilon_s.to('').magnitude,
            'sigma': self.sigma.to('S/m').magnitude,
            'rho': self.rho.to('1/m^3').magnitude,
            'z_r': self.z_r.to('m').magnitude
        }

        return args

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
    In the code below, `psi` is the unitless variable of integration.  The tip charge is located at 
    sample.cantilever.z_c, not sample.cantilever.d."""

    # Recompute Ld and epsilon_eff appropriate for the new frequency omega

    Ld = np.sqrt(sample.D / omega).to('nm')
    epsilon_eff = (sample.epsilon_s - complex(0,1) * Ld**2 / sample.LD**2).to_base_units()

    # Now the main calculation

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
    exponent = (2 * sample.cantilever.z_c / sample.z_r).to('dimensionless').magnitude

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
    exponent = (2 * sample.cantilever.z_c / sample.z_r).to('dimensionless').magnitude

    integrand = psi**power * np.exp(-1 * psi * exponent) * part(rp) 

    return integrand

def K(power, theta, sample, omega, part=np.imag):
    """Compute the integral :math:`K`."""
    
    prefactor = 1/np.power(sample.z_r, power+1)    
    integral = integrate.quad(theta, 0., np.inf, args=(sample, omega, power, part))[0]
    
    return (prefactor * integral).to_base_units()


def gamma_perpendicular(theta, sample):
    """Compute the friction coefficient for a cantilever oscillating in the perpendicular 
    geometry. :math:`\\gamma_{\\perp}`."""

    prefactor = -sample.cantilever.V_ts**2/(8 * np.pi * epsilon0 * sample.cantilever.omega_c)

    omega_c = sample.cantilever.omega_c

    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * sample.cantilever.R
    
    # Return a friction number with units

    return (prefactor * c0 * c0 * K(2, theta, sample, omega_c, np.imag)).to('pN s/m')
 
def blds_perpendicular(theta, sample, omega_m):
    """Compute the BLDS frequency-shift signal for a cantilever oscillating in the 
    perpendicular geometry.  You are always going to want to compute a BLDS 
    frequency-shift *spectrum*, so assume omega_m is a `numpy` array, with units, 
    and loop over omega_m values. For example,
    
        omega_m = ureg.Quantity(2 * np.pi * np.logspace(1, 3, 3), 'Hz').
    
    """

    prefactor = -(sample.cantilever.f_c * sample.cantilever.V_ts**2)/\
                 (8 * np.pi * epsilon0 * sample.cantilever.k_c)

    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * sample.cantilever.R

    # Return a frequency shift with units

    result = ureg.Quantity(np.zeros(len(omega_m)), 'Hz')
    for index, omega in enumerate(omega_m):

        result[index] = (prefactor * c0 * c0 * K(2, theta, sample, omega, np.real)).to('Hz')
        
    return result

def gamma_perpendicular_approx(sample, x):
    """Low-density expansion for :math:`\\gamma_{\\perp}`.  The expansion is written
    in terms of the unitless variable 
    :math:`x = \\omega_0/(\\epsilon_{\\mathrm{s}}^{\\prime} \omega_{\mathrm{c}})`"""
    
    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * sample.cantilever.R
    
    # Compute the critical density with units
    # 
    # rhox = ((sample.cantilever.omega_c * epsilon0)/(qe * sample.mu)).to('1/m^3') 
    # r2  = ((sample.epsilon_s.real**2 + sample.epsilon_s.imag**2)/sample.epsilon_s.real).to('dimensionless').magnitude
    # rho2crit = r2 * rhox

    # independent of rho, so we can precompute
    
    prefactor = -sample.cantilever.V_ts**2/(4 * np.pi * epsilon0 * sample.cantilever.omega_c)

    es = sample.epsilon_s
    a = ((es - 1)/(es + 1)).imag
    b = (- (2 * complex(0,1) * es.real)/((1 + es)**2)).imag
 
    h = sample.cantilever.z_c
    lim = ( prefactor * c0 * c0 / (4 * h**3)).to('pN s/m').magnitude

    return a.magnitude * np.ones_like(x) *  lim, (a.magnitude + b * x).magnitude * lim

def BLDSzerohigh(sample, x):
    """Loring's high-charge-density limit for the zero-frequency BLDS frequency shift
    in the perpendicular geometry. The dependent variable here is 
    :math:`x = (h/\lambda_{\mathrm{D}})^2`.
    """

    prefactor = -(sample.cantilever.f_c * sample.cantilever.V_ts**2)/(32 * np.pi * epsilon0 * sample.cantilever.k_c)

    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * sample.cantilever.R

    # array of limiting values

    h = sample.cantilever.z_c

    return np.ones_like(x) * (prefactor * c0 * c0 / h**3).to('Hz').magnitude

def BLDSzerolow(sample, x):
    """Loring's low-charge-density limit for the zero-frequency BLDS frequency shift
    in the perpendicular geometry.  The dependent variable here is 
    :math:`x = (h/\lambda_{\mathrm{D}})^2`.
    """

    er = sample.epsilon_s
    a = ((er - 1)/(er + 1)).real.magnitude

    return a * BLDSzerohigh(sample, x)

def BLDSapproxK2(sample, x):
    """Make a function to return Loring's low-density approximation for the :math:`K_2` term in the 
    BLDS frequency shift.  The dependent variable here is :math:`x = (h/\lambda_{\mathrm{D}})^2`. 
    The approximation is of the form :math:`c_1 + c_2 x` with :math:`c_1` and :mathr:`c_2` 
    constants.  The approximation increases without bound at large :math:`x`.  Only return values
    where the approximate BLDS frequency shift is less than the high-frequency limit.  Return a
    copy of the $x$ array where this is true; this truncated $x$ array will be handy for plotting.
    """

    f_high = BLDSzerohigh(sample, x)[:,0]

    er = sample.epsilon_s
    a = ((er - 1)/(er + 1)).real.magnitude
    b = (2 / (er + 1)**2).real.magnitude

    f_approx = (a + b * x) * BLDSzerohigh(sample, x)[:,0]
    mask = np.abs(f_approx) <= np.abs(f_high)

    return x[mask], f_approx[mask]

CantileverModelSpec = [
    ('f_c', float64),
    ('k_c', float64),
    ('V_ts', float64),
    ('R', float64),
    ('angle', float64),
    ('d', float64),
    ('z_c', float64)] 

@jitclass(CantileverModelSpec)
class CantileverModelJit(object):

    def __init__(self, f_c, k_c, V_ts, R, angle, d, z_c):

        self.f_c = f_c
        self.k_c = k_c
        self.V_ts = V_ts
        self.R = R
        self.angle = angle
        self.d = d
        self.z_c = z_c

    @property
    def omega_c(self):
        return 2 * np.pi * self.f_c

    def print(self):  # <=== can't use the __repr__ function :-(

        print("        cantilever freq = ", self.f_c, "Hz") # <== can't use formatting strings here
        print("                        = ", self.omega_c, "rad/s")
        print("        spring constant = ", self.k_c, "N/m")
        print("     tip-sample voltage = ", self.V_ts, "V")
        print("                 radius = ", self.R, "m")
        print("        cone half angle = ", self.angle, "degree")
        print("                 height = ", self.d, "m")
        print("  tip charge z location = ", self.z_c, "m")

CantileverModelJit_type = deferred_type()
CantileverModelJit_type.define(CantileverModelJit.class_type.instance_type)

kb_= kb.to('J/K').magnitude
T_ = 300.
qe_ = qe.to('C').magnitude
epsilon0_ = epsilon0.to('C/(V m)').magnitude

SampleModel1Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_s',complex128),
    ('h_s', float64),
    ('sigma',float64),
    ('rho',float64),
    ('epsilon_d',complex128),
    ('z_r',float64)]

@jitclass(SampleModel1Spec)
class SampleModel1Jit(object):
    
    def __init__(self, cantilever, epsilon_s, h_s, sigma, rho, epsilon_d, z_r):

        self.cantilever = cantilever
        self.epsilon_s = epsilon_s
        self.h_s = h_s
        self.sigma = sigma
        self.rho = rho
        self.epsilon_d = epsilon_d
        self.z_r = z_r

    @property
    def omega0(self):
        return self.sigma / epsilon0_

    @property
    def mu(self):
        return self.sigma / (qe_ * self.rho)
    
    @property
    def D(self):
        return (kb_ * T_ * self.mu) / qe_

    @property
    def Ld(self):
        return np.sqrt(self.D / self.cantilever.omega_c)
    
    @property
    def LD(self):
        return np.sqrt((epsilon0_ * kb_ * T_)/(self.rho * qe_ * qe_))

    @property
    def epsilon_eff(self):
        return self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2
    
    def print(self):

        print("cantilever")
        print("==========")
        self.cantilever.print()
        print("")
        print("semiconductor")
        print("=============")
        print("          epsilon (real) = ", self.epsilon_s.real)
        print("          epsilon (imag) = ", self.epsilon_s.imag)
        print("               thickness = ", self.h_s, "m")
        print("            conductivity = ", self.sigma, "S/m")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("        reference height = ", self.z_r, "m")
        print(" ")
        print("      roll-off frequency = ", self.omega0, "Hz")        
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("            Debye length = ", self.LD, "m")
        print("        diffusion length = ", self.Ld, "m")
        print("effective epsilon (real) = ", self.epsilon_eff.real)
        print("effective epsilon (imag) = ", self.epsilon_eff.imag)
        print("")
        print("dielectric")
        print("==========")
        print(" epsilon (real) = ", self.epsilon_d.real)
        print(" epsilon (imag) = ", self.epsilon_d.imag)
        print("      thickness = infinite")

SampleModel2Spec = [
    ('cantilever', CantileverModelJit_type),
    ('epsilon_d',complex128),
    ('h_d', float64),
    ('epsilon_s',complex128),
    ('sigma',float64),
    ('rho',float64),
    ('z_r',float64)]

@jitclass(SampleModel2Spec)
class SampleModel2Jit(object):
    
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
        return self.sigma / epsilon0_

    @property
    def mu(self):
        return self.sigma / (qe_ * self.rho)

    @property
    def D(self):
        return (kb_ * T_ * self.mu) / qe_

    @property
    def Ld(self):
        return np.sqrt(self.D / self.cantilever.omega_c)
    
    @property
    def LD(self):
        return np.sqrt((epsilon0_ * kb_ * T_)/(self.rho * qe_ * qe_))

    @property
    def epsilon_eff(self):
        return self.epsilon_s - complex(0,1) * self.Ld**2 / self.LD**2

    def print(self):

        print("cantilever")
        print("==========")
        self.cantilever.print()
        print("")
        print("dielectric")
        print("==========")
        print(" epsilon (real) = ", self.epsilon_d.real)
        print(" epsilon (imag) = ", self.epsilon_d.imag)
        print("      thickness = ", self.h_d, "m")
        print("")
        print("semiconductor")
        print("=============")
        print("          epsilon (real) = ", self.epsilon_s.real)
        print("          epsilon (imag) = ", self.epsilon_s.imag)
        print("               thickness = infinite")
        print("            conductivity = ", self.sigma, "S/m")        
        print("                mobility = ", self.mu, "m^2/(V s)")
        print("        reference height = ", self.z_r, "m")
        print(" ")
        print("      roll-off frequency = ", self.omega0, "Hz")
        print("      diffusion constant = ", self.D, "m^2/s")
        print("          charge density = ", self.rho, "m^{{-3}}")
        print("            Debye length = ", self.LD, "m")
        print("        diffusion length = ", self.Ld, "m")
        print("effective epsilon (real) = ", self.epsilon_eff.real)
        print("effective epsilon (imag) = ", self.epsilon_eff.imag)

# To compile the theta1norm_jit() and theta2norm_jit() functions, jit needs an actual 
#    instance of the function input "sample". 

sample1 = SampleModel1Jit(
    cantilever=CantileverModelJit(81.0e3, 2.8, 1., 80E-9, 20., 300E-9, 380E-9),
    epsilon_s=complex(11.9,-0.05),
    h_s=3000E-9,
    sigma = 1e-8,
    rho=1e21,
    epsilon_d=complex(11.9,-0.05),
    z_r=300E-9)

sample2 = SampleModel2Jit(
    cantilever=CantileverModelJit(81.0e3, 2.8, 1., 80E-9, 20., 300E-9, 380E-9),
    epsilon_d=complex(11.9,-0.05),
    h_d=0.,
    epsilon_s=complex(11.9,-0.05),
    sigma = 1e-8,
    rho=1e21,
    z_r=300E-9)

@jit(complex128(complex128), nopython=True)
def mysech_jit(x):
    """Define my own just-in-time-compiled ``sech()`` function to avoid overflow problems."""
    if x.real < 710.4:
        return 1/np.cosh(x)
    else:
        return complex(0., 0.)

@jit(complex128(complex128), nopython=True)
def mycsch_jit(x):
    """Define my own just-in-time-compiled ``csch()`` function to avoid overflow problems."""
    if x.real < 710.4:
        return 1/np.sinh(x)
    else:
        return complex(0., 0.,)

# https://github.com/numba/numba/issues/2922
# Inaccurate complex tanh implementation #2922
# "Numba complex tanh implementation returns nans for large real values, 
# probably due to some intermediate under/overflow."
#
# Replace np.tanh(a) with cmath.tanh(a), but take care to make the argument
# a is complex by replacing a -> a * complex(1,0) when a is real.
#

@jit(float64(float64,SampleModel1Jit.class_type.instance_type,float64,float64,boolean), nopython=True)
def theta1norm_jit(psi, sample1, omega, power, isImag):

    # Recompute Ld and epsilon_eff appropriate for the new frequency omega

    Ld = np.sqrt(sample1.D / omega)
    epsilon_eff = sample1.epsilon_s - complex(0,1) * Ld**2 / sample1.LD**2

    r1 = Ld**2 / (sample1.epsilon_s * sample1.LD**2)
    r2 = sample1.z_r**2 / (sample1.epsilon_s * sample1.LD**2)
    r3 = sample1.z_r**2 / Ld**2

    lambduh = complex(0,1) * r1 * psi / np.sqrt(psi**2 + r2 + complex(0,1) * r3)
 
    khs = complex(1,0) * psi * sample1.h_s / sample1.z_r

    r1 = sample1.h_s**2 / sample1.z_r**2
    r2 = sample1.h_s**2 / (sample1.epsilon_s * sample1.LD**2)
    r3 = sample1.h_s**2 / Ld**2

    etahs = complex(1,0) * np.sqrt(r1 * psi**2 + r2 + complex(0,1) * r3)

    alpha = epsilon_eff/sample1.epsilon_d
 
    r1 = 1/epsilon_eff
    t1 = - lambduh / cmath.tanh(etahs)
    t2 = cmath.tanh(khs) * cmath.tanh(etahs) \
         + alpha * cmath.tanh(etahs) \
         - lambduh \
         + 2 * lambduh * mysech_jit(khs) * mysech_jit(etahs) \
         - lambduh**2 * cmath.tanh(khs) * mysech_jit(etahs) * mycsch_jit(etahs)
    t3 = cmath.tanh(etahs) \
         + cmath.tanh(khs) * (-1 * lambduh + alpha * cmath.tanh(etahs))

    theta = r1 * (t1 + t2 / t3)

    ratio = (1 - theta) / (1 + theta)
    exponent = 2 * sample1.cantilever.z_c / sample1.z_r

    if isImag:
        integrand = psi**power * np.exp(-1 * psi * exponent) * np.imag(ratio)
    else:
        integrand = psi**power * np.exp(-1 * psi * exponent) * np.real(ratio)

    return integrand    

@jit(float64(float64,SampleModel2Jit.class_type.instance_type,float64,float64,boolean), nopython=True)
def theta2norm_jit(psi, sample2, omega, power, isImag):

    # Recompute Ld and epsilon_eff appropriate for the new frequency omega

    Ld = np.sqrt(sample2.D / omega)
    epsilon_eff = sample2.epsilon_s - complex(0,1) * Ld**2 / sample2.LD**2

    r1 = Ld**2 / (sample2.epsilon_s * sample2.LD**2)
    r2 = sample2.z_r**2 / (sample2.epsilon_s * sample2.LD**2)
    r3 = sample2.z_r**2 / Ld**2

    lambduh = complex(0,1) * r1 * psi / np.sqrt(psi**2 + r2 + complex(0,1) * r3)
     
    khd = complex(1,0) * psi * sample2.h_d / sample2.z_r

    r1 = 1/sample2.epsilon_d
    t1 = epsilon_eff * cmath.tanh(khd) + (1 - lambduh) * sample2.epsilon_d
    t2 = epsilon_eff + (1 - lambduh) * sample2.epsilon_d * cmath.tanh(khd)

    theta = (r1 * t1) / t2

    ratio = (1 - theta) / (1 + theta)
    exponent = 2 * sample2.cantilever.z_c / sample2.z_r

    if isImag:
        integrand = psi**power * np.exp(-1 * psi * exponent) * np.imag(ratio) 
    else:
        integrand = psi**power * np.exp(-1 * psi * exponent) * np.real(ratio)

    return integrand

def K_jit(power, theta, sample, omega, isImag):
    """Compute the integral :math:`K`.  The answer is returned with units. """
    
    prefactor = 1/np.power(ureg.Quantity(sample.z_r, 'm'), power+1)
    
    if isImag:
        integral = integrate.quad(theta, 0., np.inf, args=(sample, omega, power, True))[0]
    else:
        integral = integrate.quad(theta, 0., np.inf, args=(sample, omega, power, False))[0]
  
    return (prefactor * integral).to_base_units()

def gamma_perpendicular_jit(theta, sample):
    """Compute the friction for a cantilever oscillating in the
    perpendicular geometry.  Return a 3-element array, with units.
    The elements are the K2, K1, and K0 terms, respectively."""

    # The parameters sample_jit object lacks units, so add them back in manually

    prefactor = -ureg.Quantity(sample.cantilever.V_ts, 'V')**2 / \
                 (8 * np.pi * epsilon0 * ureg.Quantity(sample.cantilever.omega_c, 'Hz'))
    
    omega_c = sample.cantilever.omega_c

    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * ureg.Quantity(sample.cantilever.R, 'm')

    # Return a frequency shift with units

    return (prefactor * c0 * c0 * K_jit(2, theta, sample, omega_c, True)).to('pN s/m')

def blds_perpendicular_jit(theta, sample, omega_m):
    """Compute the BLDS frequency-shift signal for a cantilever oscillating in the 
    perpendicular geometry.  You are always going to want to compute a BLDS 
    frequency-shift *spectrum*, so assume omega_m is a `numpy` array, with units, 
    and loop over omega_m values. For example,
    
        omega_m = ureg.Quantity(2 * np.pi * np.logspace(1, 3, 3), 'Hz').
    
    """

    # The parameters sample_jit object lacks units, so add them back in manually

    prefactor = -(ureg.Quantity(sample.cantilever.f_c, 'Hz') * ureg.Quantity(sample.cantilever.V_ts, 'V')**2)/ \
                 (8 * np.pi * epsilon0 * ureg.Quantity(sample.cantilever.k_c, 'N/m'))

    # In Loring's new theory, c0 is the capacitance of the probe in the absence of the sample

    c0 = 4 * np.pi * epsilon0 * ureg.Quantity(sample.cantilever.R, 'm')

    # Return a frequency shift with units
     
    result = ureg.Quantity(np.zeros(len(omega_m)), 'Hz')
    for index, omega in enumerate(omega_m.to('Hz').magnitude):
    
        result[index] = (prefactor * c0 * c0 * K_jit(2, theta, sample, omega, False)).to('Hz')
        
    return result

def calculate_vs_conductivity(theta, sample, omega_m, rho, sigma):
    """Create a pandas dataframe row of useful results."""

    df = pd.DataFrame()

    for sigma_, rho_ in zip(sigma, rho):
        
        sample.rho = rho_.to('1/m^3').magnitude
        sample.sigma = sigma_.to('S/m').magnitude
    
        gamma = gamma_perpendicular_jit(theta, sample).to('pN s/m')
        f_BLDS = blds_perpendicular_jit(theta, sample, omega_m).to('Hz')
        
        ep = sample.epsilon_s.real   
        z_c = ureg.Quantity(sample.cantilever.z_c, 'm')
        LD = ureg.Quantity(sample.LD, 'm')
        rho = ureg.Quantity(sample.rho, '1/m^3')
        omega_0 = (ureg.Quantity(sample.sigma, 'S/m')/epsilon0).to('Hz')
        omega_c = ureg.Quantity(sample.cantilever.omega_c, 'Hz')
        
        new_row = pd.DataFrame([
            {'sigma [S/m]': sigma_.to('S/m').magnitude,
             'rho [1/cm^3]': rho_.to('1/cm^3').magnitude,
             'L_D [nm]': LD.to('nm').magnitude,
             'rho scaled 1': (z_c**2/(LD**2)).to('').magnitude,
             'rho scaled 2': (z_c**2/(ep * LD**2)).to('').magnitude,
             'rho scaled 3': (z_c**2/(7.742 * ep * LD**2)).to('').magnitude,  # see below
             'omega0 [Hz]': omega_0.to('Hz').magnitude,
             'omega_c [Hz]': omega_c.to('Hz').magnitude,
             'omega_c scaled': (omega_0/(ep * omega_c)).to('').magnitude,
             'omega_m [Hz]': omega_m.to('Hz').magnitude,
             'omega_m scaled': ((ep * omega_m)/omega_0).to('').magnitude,
             'f_BLDS [Hz]': f_BLDS.to('Hz').magnitude, 
             'gamma [pN s/m]': gamma.to('pN s/m').magnitude}])
    
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df