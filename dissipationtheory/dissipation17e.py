# dissipationtheory17e.py
# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-12-07
# Summary: CPU code for computing computing dissipation and frequency shift for a point probe over a 
#          Type I, Type II, and Type III sample, encoding formulas provided by Roger Loring on 2026-12-05.

import numpy as np
import scipy

from dissipationtheory.constants import ureg, epsilon0, qe
from dissipationtheory.dissipation13e import KmatrixI_jit, KmatrixII_jit,  KmatrixIII_jit, KmatrixIV_jit


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
        are returned as pint quantities with units."""
        
        if self.sample.type == 4:
            
            K0, K1, K2 = KmatrixIV_jit(self.sj, self.rk)
            
        elif self.sample.type == 3:

            j0s = scipy.special.jn_zeros(0,100.)
            an, _ = scipy.integrate.newton_cotes(20, 1)
            
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

            j0s = scipy.special.jn_zeros(0,100.)
            an, _ = scipy.integrate.newton_cotes(20, 1)
            
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

            j0s = scipy.special.jn_zeros(0,100.)
            an, _ = scipy.integrate.newton_cotes(20, 1)
            
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
        
        return ureg.Quantity(K0[0][0],'1/nm**1'), \
               ureg.Quantity(K1[0][0],'1/nm**2'), \
               ureg.Quantity(K2[0][0],'1/nm**3')
    
    def properties(self, omega_m):
        """Compute the friction when a DC voltage is applied to the cantilever.
        Compute the cantilever friction and frequency shift when 
        DC and AC voltages are applied to the cantilever.  Here omega_m
        is the unitless voltage-modulation frequency."""
        
        K0dc, K1dc, K2dc = self.solve(0.)
        K0ac, K1ac, K2ac = self.solve(self.sample.cantilever.omega_c)

        V0 = ureg.Quantity(self.sample.cantilever.V_ts, 'V')
        fc = ureg.Quantity(self.sample.cantilever.f_c, 'Hz')
        kc = ureg.Quantity(self.sample.cantilever.k_c, 'N/m')        
        
        C0 = 4 * np.pi * epsilon0 * ureg.Quantity(self.sample.cantilever.R, 'm')
        q0 = C0 * V0
        
        wc_units = ureg.Quantity(self.sample.cantilever.omega_c, 'Hz')
        gamma = - (q0**2 * K2ac.imag)/(8 * np.pi * epsilon0 * wc_units)
        
        dK2ac = K2ac - K2dc
        Kterms = K2dc.real + 0.25 * dK2ac.real
        
        dfdc = - (fc * q0**2 * Kterms.real)/(4 * np.pi * epsilon0 * kc)
        
        wc = self.sample.cantilever.omega_c
        wp = wc + omega_m
        wm = wc - omega_m
        
        K0p, K1p, K2p = self.solve(np.abs(wp))
        K0m, K1m, K2m = self.solve(np.abs(wm))
        
        dK2p = K2p - K2dc
        dK2m = K2m - K2dc
        Kterms = K2ac + wc * 0.25 * (dK2p / wp + dK2m / wm)
        
        dfac = - (fc * q0**2 * Kterms.real)/(8 * np.pi * epsilon0 * kc)
        
        self.results['C0 [aF]'] = C0.to('aF').magnitude
        self.results['q0/qe'] = (q0/qe).to('').magnitude
        self.results['gamma [pN s/m]'] = gamma.to('pN s/m').magnitude
        self.results['Delta f dc [Hz]'] = dfdc.to('Hz').magnitude
        self.results['Delta f ac [Hz]'] = dfac.to('Hz').magnitude
        
        self.keys += ['C0 [aF]', 'q0/qe', 'gamma [pN s/m]']
        self.keys += ['Delta f dc [Hz]', 'Delta f ac [Hz]']
        
    def print_key_results(self):

        print('-'*50)
        for key in self.keys:
            print('{0:18s} {1:11.3f}    {1:+0.6e}'.format(
                key.rjust(18),
                self.results[key]))
        print('-'*50)


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

    obj = {}

    obj['sphere'] = twodimCobject(sample1_jit)
    obj['sphere'].addsphere(h, 20, 20)
    obj['sphere'].set_alpha(1.0e-6)
    obj['sphere'].set_breakpoints(15)
    obj['sphere'].properties_dc()
    obj['sphere'].properties_ac(omega_m=wm)
    obj['sphere'].properties_am(omega_m=wm, omega_am = 250.)
    obj['sphere'].print_key_results()

    obj['point'] = pointprobeCobject(sample1_jit)
    obj['point'].addsphere(h)
    obj['point'].set_breakpoints(15)
    obj['point'].properties(wm)
    obj['point'].print_key_results()

    compare_results(obj, 'sphere', 'point', 
        ['C0 [aF]', 'gamma [pN s/m]', 'Delta f dc [Hz]', 'Delta f ac [Hz]'])