# python -m unittest -v test_dissipation12e.py
# python -m unittest -v test_dissipation12e.py --verbose
# python -m unittest -v tests/test_dissipation12e.py 
# python -m unittest -v tests/test_dissipation12e.py --verbose

# Author: John A. Marohn (jam99@cornell.edu)
# Date: 2025-09-01 
# Summary: Test the KmatrixIII_jit and KmatrixIV_jit functions in dissipation12e.py.

import unittest
import numpy as np
import scipy
import sys
import argparse
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation9a import CantileverModel, SampleModel1, SampleModel2, SampleModel3, SampleModel4
from dissipationtheory.dissipation9b import SampleModel1Jit, SampleModel2Jit, SampleModel3Jit, SampleModel4Jit
from dissipationtheory.dissipation9b import integrand1jit, integrand2jit, integrand3jit
from dissipationtheory.dissipation9b import K_jit
from dissipationtheory.dissipation12e import KmatrixIII_jit, KmatrixIV_jit

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', action='store_true', help='Enable verbose print statements')
args, unknown = parser.parse_known_args()
VERBOSE = args.verbose

class TestDissipation11eMethods(unittest.TestCase):
    """Compute $K_0$, $K_1$, and $K_2$ two ways, confirming that my bespoke, compiled
    integration routine in KmatrixIII_jit gives essentially the same result as the
    scipy.integrate result in K_jit for Type III and Type IV samples. 
    """

    def setUp(self):

        cantilever = CantileverModel(
            f_c = ureg.Quantity(62, 'kHz'),
            k_c = ureg.Quantity(2.8, 'N/m'), 
            V_ts = ureg.Quantity(1, 'V'), 
            R = ureg.Quantity(60, 'nm'),
            angle = ureg.Quantity(20, 'degree'),
            L = ureg.Quantity(1000, 'nm'))

        sample3 = SampleModel3(
            cantilever = cantilever,
            epsilon_s = ureg.Quantity(complex(20, 2), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(1, 'nm'))

        sample4 = SampleModel4(
            cantilever = cantilever,
            z_r = ureg.Quantity(1, 'nm'))
        
        sample5 = SampleModel3(
            cantilever = cantilever,
            epsilon_s = ureg.Quantity(complex(1e6, 0), ''),
            sigma = ureg.Quantity(1e7, 'S/m'),
            rho = ureg.Quantity(1e26, '1/m^3'),
            z_r = ureg.Quantity(1, 'nm'))

        self.sample3_jit = SampleModel3Jit(**sample3.args())
        self.sample4_jit = SampleModel4Jit(**sample4.args())
        self.sample5_jit = SampleModel3Jit(**sample5.args())

        self.loc1_nm_array = np.array(
            [np.array([  0,  0, 30], dtype=np.float64),
             np.array([  0, 20, 30], dtype=np.float64),
             np.array([  0, 30, 30], dtype=np.float64),
             np.array([  0,  0, 20], dtype=np.float64),
             np.array([  0, 10, 20], dtype=np.float64),
             np.array([  0, 20, 20], dtype=np.float64)])

        self.loc2_nm_array = np.array(
            [np.array([  0,  0, 30], dtype=np.float64),
             np.array([  0,  0, 30], dtype=np.float64),
             np.array([  0,  0, 30], dtype=np.float64),
             np.array([  0,  0, 20], dtype=np.float64),
             np.array([  0,  0, 20], dtype=np.float64),
             np.array([  0,  0, 20], dtype=np.float64)])
        
        self.omega_array = 2 * np.pi * np.array(
            [1e1, 1e2, 1e3, 1e4, 1e5, 1e6], dtype=np.float64)

    def test_Sample3(self):
        """Do K_jit and  KmatrixIII_jit return the same result for a Type III sample?"""

        results = np.zeros((len(self.omega_array), len(self.loc1_nm_array), 2), dtype=bool)

        if VERBOSE:
            print('')

        for idx1, omega in enumerate(self.omega_array):

            if VERBOSE:
                print('=' * 60)
                print('freq({:02d})'.format(idx1), 'f =', omega/(2*np.pi), 'Hz')

            for idx2, (loc1_nm, loc2_nm) in enumerate(zip(self.loc1_nm_array, self.loc2_nm_array)):

                if VERBOSE:
                    print('-' * 60)
                    print('pos({:02d}):'.format(idx2), 'loc1 =', loc1_nm, 'nm and loc2 =', loc2_nm, 'nm')
                
                loc1_m = 1e-9 * loc1_nm
                loc2_m = 1e-9 * loc2_nm

                params3_jit = {
                    'integrand': integrand3jit, 
                    'sample': self.sample3_jit, 
                    'omega': omega, 
                    'location1': loc1_m, 
                    'location2': loc2_m}

                K0a, K1a, K2a = K_jit(**params3_jit)
                a = np.array([K0a, K1a, K2a])

                j0s = scipy.special.jn_zeros(0, 100.0)
                an, _ = scipy.integrate.newton_cotes(20, 1)

                args = {
                    "omega": omega,
                    "omega0": params3_jit['sample'].omega0,
                    "kD": params3_jit['sample'].kD,
                    "es": params3_jit['sample'].epsilon_s,
                    "sj": np.array([loc1_nm]),
                    "rk": np.array([loc2_nm]),
                    "j0s": j0s,
                    "an": an}

                K0b, K1b, K2b = KmatrixIII_jit(**args)
                b = np.array([K0b[0][0], K1b[0][0], K2b[0][0]])
                
                # 0.01 percent relative tolerance
                results[idx1, idx2, 0] = np.allclose(b.real, a.real, atol=0., rtol=1e-4)
                results[idx1, idx2, 1] = np.allclose(b.imag, a.imag, atol=0., rtol=1e-4) 

                if VERBOSE or not results[idx1, idx2, 0]:

                    print(results[idx1, idx2, 0])  
                    for idx, (Ka, Kb) in enumerate(zip(a,b)):
                        
                        err_real = np.abs(Ka.real-Kb.real)/Ka.real        
                        print('Re[K[{:d}]] {:+0.6e} {:+0.6e}, err = {:+6.3e}'.format(idx, Ka.real, Kb.real, err_real))
                
                if VERBOSE or not results[idx1, idx2, 1]:

                    print(results[idx1, idx2, 1])
                    for idx, (Ka, Kb) in enumerate(zip(a,b)):
                        
                        err_imag = np.abs(Ka.imag-Kb.imag)/Ka.imag
                        print('Im[K[{:d}]] {:+0.6e} {:+0.6e}, err = {:+6.3e}'.format(idx, Ka.imag, Kb.imag, err_imag))

        if VERBOSE:
            print("")
            print("Type III sample test results")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("")
            print("freq  pos  real  imag")
            print("---- ---- ----- -----")
            for idx1, omega in enumerate(self.omega_array):
                for idx2, (loc1_nm, loc2_nm) in enumerate(zip(self.loc1_nm_array, self.loc2_nm_array)):

                    print('  {:02d}'.format(idx1), 
                        '  {:02d}'.format(idx2),
                        '{:}'.format(results[idx1, idx2, 0]).rjust(5), 
                        '{:}'.format(results[idx1, idx2, 1]).rjust(5))
                
        self.assertTrue(
            results[:, :, 0].all() and
            results[:, :, 1].all())