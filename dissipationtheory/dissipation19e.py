3# dissipationtheory18e.py
# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2026-02-11
# Summary: CPU code for computing computing dissipation and frequency shift for a point probe over a 
#          Type I, Type II, and Type III sample, encoding formulas provided by Roger Loring on 2026-01-17 
#          via Overleaf document "BLDS_1_2026".  In the 2026-01-17 treatment, as per Loring's email 
#          2026-01-09, the frequency shift formula does not include the problematic Bern and Frisch terms.
#  
#
# Classes:
#  - pointprobeCobject
#
# Functions:
#  - compare_results

import numpy as np
import scipy

from dissipationtheory.constants import ureg, epsilon0, qe
from dissipationtheory.dissipation13e import KmatrixI_jit, KmatrixII_jit,  KmatrixIII_jit, KmatrixIV_jit

import pandas as pd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


class pointprobeCobject():

    def __init__(self, sample):
        """Here sample is a SampleModel1Jit, SampleModel2Jit, SampleModel3Jit, or SampleModel4Jit object."""

        self.sample = sample

        self.Vr = ureg.Quantity(1, 'V')
        self.zr = ureg.Quantity(1, 'nm')

        self.results = {}
        self.results['Vts [V]'] = self.sample.cantilever.V_ts
        self.keys = ['Vts [V]']

        self.breakpoints = 10 

        self.results['breakpoints'] = self.breakpoints
        self.keys += ['breakpoints']
        
    def addsphere(self, h):
        """Model a sphere of radius $r$ above a ground plane, with a tip-sample
        separation of $h$.  Creates two unitless (1,3) arrays: (a) self.sj, a 
        voltage-test point at the center of the sphere, and (b) self.rk, 
        the location of the tip charge, also at the center of the sphere. 
        Coordinates are in nanometers.""" 
        
        r = ureg.Quantity(self.sample.cantilever.R, 'm')
        
        H = h.to('nm').magnitude
        R = r.to('nm').magnitude
        
        self.sj = np.array([[0., 0., H + R]])
        self.rk = np.array([[0., 0., H + R]])
        
        self.info = {'type': 'sphere', 
                     'r [nm]': R, 
                     'h [nm]': H}        
        
    def set_breakpoints(self, breakpoints):
        """Set the number of breakpoints to use in the numerical integration."""
        
        self.breakpoints = breakpoints 
        self.results['breakpoints'] = breakpoints    
        
    def solve(self, omega):
        """Compute the unitless K0, K1, and K2 integrals.  The units of the 
        integrals are 1/nm, 1/nm**2, and 1/nm**3 respectively.  The integrals
        are returned as numbers without units."""

        j0s = scipy.special.jn_zeros(0,100.)
        an, _ = scipy.integrate.newton_cotes(20, 1)

        if self.sample.type == 4:
            
            K0, K1, K2 = KmatrixIV_jit(self.sj, self.rk)
            
        elif self.sample.type == 3:

            args = {'omega': omega, 
                'omega0': self.sample.omega0,
                'kD': self.sample.kD, 
                'es': self.sample.epsilon_s, 
                'sj': self.sj, 
                'rk': self.rk, 
                'j0s': j0s, 
                'an': an,
                'breakpoints': self.breakpoints}
        
            K0, K1, K2 = KmatrixIII_jit(**args)

        elif self.sample.type == 2:
            
            args = {'omega': omega, 
                'omega0': self.sample.omega0,
                'kD': self.sample.kD,
                'hd': self.sample.h_d,
                'ed': self.sample.epsilon_d,
                'es': self.sample.epsilon_s, 
                'sj': self.sj, 
                'rk': self.rk, 
                'j0s': j0s, 
                'an': an,
                'breakpoints': self.breakpoints}
        
            K0, K1, K2 = KmatrixII_jit(**args)

        elif self.sample.type == 1:
 
            args = {'omega': omega, 
                'omega0': self.sample.omega0,
                'kD': self.sample.kD,
                'hs': self.sample.h_s,
                'es': self.sample.epsilon_s, 
                'ed': self.sample.epsilon_d,
                'sj': self.sj, 
                'rk': self.rk, 
                'j0s': j0s, 
                'an': an,
                'breakpoints': self.breakpoints}
        
            K0, K1, K2 = KmatrixI_jit(**args)

        else:

            raise Exception("unknown sample type")
        
        return K0[0][0], K1[0][0], K2[0][0]

    def properties(self):
        """Return useful quantities with units."""

        Rc = ureg.Quantity(self.sample.cantilever.R, 'm')
        Vc = ureg.Quantity(self.sample.cantilever.V_ts, 'V')
    
        C0 = 4 * np.pi * epsilon0 * Rc
        q0 = C0 * Vc   

        fc = ureg.Quantity(self.sample.cantilever.f_c, 'Hz')
        kc = ureg.Quantity(self.sample.cantilever.k_c, 'N/m')   
        wc = ureg.Quantity(self.sample.cantilever.omega_c, 'Hz')

        if 'C0 [aF]' not in self.keys:

            self.results['C0 [aF]'] = C0.to('aF').magnitude
            self.keys += ['C0 [aF]']

        if 'q0/qe' not in self.keys:
            
            self.results['q0/qe'] = (q0/qe).to('').magnitude
            self.keys += ['q0/qe']

        return q0, wc, fc, kc
    
    def friction(self):
        """Compute the frition."""

        _, _, K2wc = self.solve(self.sample.cantilever.omega_c)
        q0, wc, _, _ = self.properties()
        
        gamma = - (q0**2 * ureg.Quantity(K2wc, '1/nm**3').imag)/(8 * np.pi * epsilon0 * wc)  

        self.results['gamma [pN s/m]'] = gamma.to('pN s/m').magnitude
        self.keys += ['gamma [pN s/m]']

        return(gamma.to('pN s/m').magnitude)

    def DC(self):
        "Compute the dc frequency shift."

        _, _, K2dc = self.solve(0.)
        q0, _, fc, kc = self.properties()

        df_DC = - (fc * q0**2 * ureg.Quantity(K2dc, '1/nm**3').real)/(4 * np.pi * epsilon0 * kc)

        self.results['Delta f dc [Hz]'] = df_DC.to('Hz').magnitude
        self.keys += ['Delta f dc [Hz]']

        return(df_DC.to('Hz').magnitude)

    def LDS(self, omega_m):
        """Return an array of LDS frequency shifts.  Here omega_m is a numpy
        array of unitless voltage-modulation frequencies in Hz."""
        
        q0, _, fc, kc = self.properties()
        
        K2wm = ureg.Quantity(np.zeros(len(omega_m), dtype=complex), '1/nm^3')

        for index, omega_m__value in enumerate(omega_m): 
        
            _, _, K2wm__value = self.solve(omega_m__value)
            K2wm[index] = ureg.Quantity(K2wm__value, '1/nm**3')
            
        df_LDS = - (fc * q0**2 * K2wm.real)/(8 * np.pi * epsilon0 * kc)

        # Save a representative frequency shift, the first array element
        self.results['Delta f ac [Hz]'] = df_LDS[0].to('Hz').magnitude
        self.keys += ['Delta f ac [Hz]']

        return(df_LDS.to('Hz').magnitude)

    def BLDS(self, omega_m, omega_am):
        """Return an array of BLDS frequency shifts.  Here omega_m is a numpy
        array of unitless voltage-modulation frequencies in Hz, and omega_am
        is the amplitude-modulation frequency in Hz."""
        
        q0, _, fc, kc = self.properties()
        
        K2wm = ureg.Quantity(np.zeros(len(omega_m), dtype=complex), '1/nm^3')
        K2wmm = ureg.Quantity(np.zeros(len(omega_m), dtype=complex), '1/nm^3')
        K2wmp = ureg.Quantity(np.zeros(len(omega_m), dtype=complex), '1/nm^3')

        for index, omega_m__value in enumerate(omega_m): 
        
            _, _, K2wm__value = self.solve(omega_m__value)
            _, _, K2wmm__value = self.solve(omega_m__value - omega_am)
            _, _, K2wmp__value = self.solve(omega_m__value + omega_am)

            K2wm[index] = ureg.Quantity(K2wm__value, '1/nm**3')
            K2wmm[index] = ureg.Quantity(K2wmm__value, '1/nm**3')
            K2wmp[index] = ureg.Quantity(K2wmp__value, '1/nm**3')
        
        K2terms = K2wm + 0.25 * (K2wmm + K2wm + K2wmp)

        df_BLDS = - (fc * q0**2 * K2terms.real)/(32 * np.pi * epsilon0 * kc)
        
        # Save a representative frequency shift, the first array element
        self.results['Delta f am [Hz]'] = df_BLDS[0].to('Hz').magnitude
        self.keys += ['Delta f am [Hz]']

        return(df_BLDS.to('Hz').magnitude)

    def LDSO(self, V0, Vm, omega_m):
        """Return an array of LDS frequency shifts.  Here V0 is the unitless dc voltage, 
        Vm is the zero-to-peak ac voltage, and omega_m is a numpy array of unitless voltage-modulation
        frequencies in Hz."""
        
        _, _, fc, kc = self.properties()
        
        q0 = 4 * np.pi * epsilon0 * ureg.Quantity(self.sample.cantilever.R, 'm') * ureg.Quantity(V0, 'V')
        qm = 4 * np.pi * epsilon0 * ureg.Quantity(self.sample.cantilever.R, 'm') * ureg.Quantity(Vm, 'V')
        
        _, _, K2dc = self.solve(0.)
        
        K2wm = ureg.Quantity(np.zeros(len(omega_m), dtype=complex), '1/nm^3')

        for index, omega_m__value in enumerate(omega_m):
        
            _, _, K2wm__value = self.solve(omega_m__value)
            K2wm[index] = ureg.Quantity(K2wm__value, '1/nm**3')
            
        df_LDSO = - (fc * q0**2 * ureg.Quantity(K2dc, '1/nm^3').real +
                     fc * 0.50 * qm**2 * K2wm.real)/(4 * np.pi * epsilon0 * kc)

        return(df_LDSO.to('Hz').magnitude)

    def print_key_results(self):

        print('-'*50)
        for key in self.keys:
            print('{0:18s} {1:11.3f}    {1:+0.6e}'.format(
                key.rjust(18),
                self.results[key]))
        print('-'*50)

class ExptSweepConductivity(object):

    def __init__(self, msg):

        self.msg = msg
        self.df = pd.DataFrame()

    def calculate(self, sample, h, omega_am, omega_m, rho, sigma):
        """Create a pandas dataframe row of useful results."""

        for sigma_, rho_ in zip(sigma, rho):
            
            sample.rho = rho_.to('1/m^3').magnitude
            sample.sigma = sigma_.to('S/m').magnitude
        
            obj = pointprobeCobject(sample)
            obj.addsphere(h)
            obj.set_breakpoints(15)            
            
            gamma = obj.friction()
            f_LDS = obj.LDS(omega_m.to('Hz').magnitude)
            f_BLDS = obj.BLDS(omega_m.to('Hz').magnitude, omega_am.to('Hz').magnitude)

            ep = sample.epsilon_s.real   
            z_c = ureg.Quantity(obj.info['r [nm]'] + obj.info['h [nm]'], 'nm')
            
            LD = 1/ureg.Quantity(sample.kD, '1/m')
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
                'gamma [pN s/m]': gamma,
                'f_LDS [Hz]': f_LDS,
                'f_BLDS [Hz]': f_BLDS}])

            self.df = pd.concat([self.df, new_row], ignore_index=True)
            
        return obj # save a copy, with the last-used rho and sigma

def plot_LDS(self, n=1, scaled=False):

    rho = self.df['rho [1/cm^3]'][::n]

    lists = zip(self.df['omega_m [Hz]'][::n], 
                self.df['omega_m scaled'][::n],
                self.df['f_LDS [Hz]'][::n])
    
    colormap = plt.cm.magma_r
    color_list = [colormap(i) for i in np.linspace(0, 1, len(rho))]

    # color bar
    normalized_colors = mcolors.LogNorm(vmin=min(rho), vmax=max(rho))
    scalar_mappable = cm.ScalarMappable(norm=normalized_colors, cmap=colormap)
    scalar_mappable.set_array(len(color_list))

    fig, ax = plt.subplots(figsize=(3.50, 2.5))

    for index, (omega_m, omega_m_scaled, f_BLDS) in enumerate(lists):

        if scaled:
            x = omega_m_scaled
            xlabel = 'scaled modulation freq. $\Omega_{\mathrm{m}} = ' \
                    '(\epsilon_{\mathrm{s}}^{\prime} \omega_{\mathrm{m}})/\omega_0$'
        else:
            x = omega_m
            xlabel = 'modulation freq. $\omega_{\mathrm{m}}$ [rad/s]'

        plt.semilogx(x, np.abs(f_BLDS), '-', color=color_list[index])

    # color bar
    clb=plt.colorbar(scalar_mappable, ax=ax)
    clb.ax.set_title(r'$\rho \: [\mathrm{cm}^{-3}]$', fontsize=12)

    plt.ylabel(r'$\vert \Delta f_{\mathrm{LDS}} \vert$ [Hz]')
    plt.xlabel(xlabel)
    plt.tight_layout()

    return fig 

def dfdclims(obj):
    
    es = obj.sample.epsilon_s
    eta = (es - 1)/(es + 1)
    
    fc = ureg.Quantity(obj.sample.cantilever.f_c, 'Hz')
    kc = ureg.Quantity(obj.sample.cantilever.k_c, 'N/m')
    
    R = ureg.Quantity(obj.sample.cantilever.R, 'm')
    V = ureg.Quantity(obj.sample.cantilever.V_ts, 'V')
    q0 = ureg.Quantity(4 * np.pi * epsilon0 * R * V, 'C')

    zc = ureg.Quantity(obj.info['r [nm]'] + obj.info['h [nm]'], 'nm')
    
    dfdc = -(fc * q0**2)/(4 * np.pi * epsilon0 * kc * 4 * zc**3)
    
    return np.array([dfdc.to('Hz').magnitude,
                     eta.real * dfdc.to('Hz').magnitude])

def plot_LDS_zero(self, obj, abs=False):

    [df_metal, df_diel] = dfdclims(obj) / 2
    
    y = np.array([x[0] for x in self.df['f_LDS [Hz]'].values])
    if abs:
        y = np.abs(y)

    x1 = self.df['rho scaled 3'].values
    x2 = self.df['rho [1/cm^3]'].values

    # Define functions to convert from  $\hat{\rho}$ to $\rho$ and back again

    c = (x2/x1)[0]
    fwd = lambda x1: x1*c
    rev = lambda x2: x2/c 

    with plt.style.context('seaborn-v0_8'):

        fig, ax1 = plt.subplots(1, 1, figsize=(4.00, 2.75))

        ax1.semilogx(x1, y)
        
        ax1.hlines(df_diel, min(x1), 100.0, linestyles='-.', colors='tab:gray', label='dielectric') 
        ax1.hlines(df_metal, 0.01, max(x1), linestyles='--', colors='tab:gray', label='metal') 

        if abs:
            ax1.set_ylabel(r'$|\Delta f_{\mathrm{LDS}}(\omega_{\mathrm{m}}=0)|$ [Hz]')
        else:
            ax1.set_ylabel(r'$\Delta f_{\mathrm{LDS}}(\omega_{\mathrm{m}}=0)$ [Hz]')

        if plt.rcParams['text.usetex']:
            ax1.set_xlabel(
                r'scaled charge density $\hat{\rho}_3 = '
                r'z^2_{\mathrm{c}} \Big/ 7.742 \, \epsilon^{\prime}_{\mathrm{s}} \lambda^2_{\mathrm{D}}$')
        else:
             ax1.set_xlabel(
                r'scaled charge density $\hat{\rho}_3 = '
                r'z^2_{\mathrm{c}} \: / \: 7.742 \, \epsilon^{\prime}_{\mathrm{s}} \lambda^2_{\mathrm{D}}$')

        ax2 = ax1.secondary_xaxis("top", functions=(fwd,rev))
        ax2.set_xlabel(r'charge density $\rho$ [cm$^{-3}$]')
        
        plt.legend(frameon=True, framealpha=1, loc=6, handlelength=3.5)
        plt.tight_layout()

    return fig

def plot_friction(self):

    y = np.array([row for row in self.df['gamma [pN s/m]']])
    x1 = self.df['omega_c scaled'].values
    x2 = self.df['rho [1/cm^3]'].values

    # Define functions to convert from  $\hat{\rho}$ to $\rho$ and back again

    c = (x2/x1)[0]
    fwd = lambda x1: x1*c
    rev = lambda x2: x2/c 

    with plt.style.context('seaborn-v0_8'):

        fig, ax1 = plt.subplots(1, 1, figsize=(3.00, 2.50))

        ax1.semilogx(x1, y)
        ax1.set_ylabel((r'friction $\gamma_{\perp}$ [pN s/m]'))
        ax1.set_xlabel(r'scaled frequency $\Omega_0 = '
                       r'\omega_0/(\epsilon_{\mathrm{s}}^{\prime} \omega_{\mathrm{c}})$')

        ax2 = ax1.secondary_xaxis("top", functions=(fwd,rev))
        ax2.set_xlabel(r'charge density $\rho$ [cm$^{-3}$]')
        plt.tight_layout()

    return fig

def plot_friction_list(expt, keys, scaled=False):

    with plt.style.context('seaborn-v0_8'):

        fig, ax = plt.subplots(1, 1, figsize=(3.75, 2.50))

        for key in keys:

            if scaled:
                x = expt[key].df['omega_c scaled'].values
            else:    
                x = expt[key].df['rho [1/cm^3]'].values    
            
            y = np.array([row for row in expt[key].df['gamma [pN s/m]']])        

            plt.loglog(x, y)
        
        ax.set_ylabel((r'friction $\gamma_{\perp}$ [pN s/m]'))
        
        if scaled:
            ax.set_xlabel(r'scaled frequency $\Omega_0 = '
                          r'\omega_0/(\epsilon_{\mathrm{s}}^{\prime} \omega_{\mathrm{c}})$')
        else:
            ax.set_xlabel(r'charge density $\rho$ [cm$^{-3}$]')
        
        plt.tight_layout()

    return fig

def latex_float(f):
    """Example function call.

        latex_float(3e-7)
        => '$3.0 \\times 10^{-7}$'
        
    """
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str  

def compare_results(obj, key1, key2, keys):
    
    print('-'*71)
    print('{0:18s} {1:12s} {2:12s} {3:15s} {4:10s}'.format(
        'quantity'.rjust(18), 
        key1.rjust(12), 
        key2.rjust(12), 
        ('{:}/{:}'.format(key2, key1)).rjust(15),
        '|err| %'.rjust(10)))
    
    print('-'*71)
    for key in keys:
        val1 = obj[key1].results[key]
        val2 = obj[key2].results[key]

        print('{0:18s} {1:+12.4e} {2:+12.4e} {3:+15.4f} {4:10.2f}'.format(
            key.rjust(18),
            val1,
            val2,
            val2 / val1,
            100 * np.abs((val2 - val1)/val1)))

    print('-'*71)

if __name__ == "__main__":

    from dissipationtheory.dissipation9a import CantileverModel, SampleModel1
    from dissipationtheory.dissipation9b import SampleModel1Jit
    from dissipationtheory.dissipation13e import twodimCobject

    cantilever = CantileverModel(
        f_c = ureg.Quantity(60.360, 'kHz'),
        k_c = ureg.Quantity(2.8, 'N/m'), 
        V_ts = ureg.Quantity(1, 'V'), 
        R = ureg.Quantity(10, 'nm'),
        angle = ureg.Quantity(24.2, 'degree'),
        L = ureg.Quantity(2250, 'nm'))

    sample1 = SampleModel1(
        cantilever = cantilever,
        h_s = ureg.Quantity(400, 'nm'),
        epsilon_s = ureg.Quantity(complex(20.0, -0.01), ''),
        epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
        sigma = ureg.Quantity(9.7e-7, 'S/cm'),
        rho = ureg.Quantity(1.9e15, '1/cm^3'),
        z_r = ureg.Quantity(1, 'nm'))

    sample1_jit = SampleModel1Jit(**sample1.args())
    h = ureg.Quantity(1000, 'nm')
    wm = 1.0e5
    wam = 250.0

    obj = {}

    obj['sphere'] = twodimCobject(sample1_jit)
    obj['sphere'].addsphere(h, 20, 20)
    obj['sphere'].set_alpha(1.0e-6)
    obj['sphere'].set_breakpoints(15)
    obj['sphere'].properties_dc()
    obj['sphere'].properties_ac(omega_m=wm)
    obj['sphere'].properties_am(omega_m=wm, omega_am=wam)
    obj['sphere'].print_key_results()

    obj['point'] = pointprobeCobject(sample1_jit)
    obj['point'].addsphere(h)
    obj['point'].set_breakpoints(15)
    obj['point'].friction()
    obj['point'].DC()
    obj['point'].LDS(omega_m=np.array([wm]))
    obj['point'].BLDS(omega_m=np.array([wm]), omega_am=wam)
    obj['point'].print_key_results()

    compare_results(obj, 'sphere', 'point', 
        ['C0 [aF]', 'gamma [pN s/m]', 
         'Delta f dc [Hz]', 'Delta f ac [Hz]', 'Delta f am [Hz]'])