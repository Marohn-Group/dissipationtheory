# python -m unittest -v test_dissipation9b.py
# python -m unittest -v tests/test_dissipation9b.py 
#
# Ran 8 tests in 2.685s

import unittest
import numpy as np
from dissipationtheory.constants import ureg
from dissipationtheory.dissipation9a import CantileverModel, SampleModel1, SampleModel2, SampleModel3, SampleModel4
from dissipationtheory.dissipation9a import integrand1, integrand2, integrand3
from dissipationtheory.dissipation9a import K, Kunits, Kmetal, Kmetalunits
from dissipationtheory.dissipation9b import SampleModel1Jit, SampleModel2Jit, SampleModel3Jit, SampleModel4Jit
from dissipationtheory.dissipation9b import integrand1jit, integrand2jit, integrand3jit
from dissipationtheory.dissipation9b import K_jit, Kunits_jit, Kmetal_jit, Kmetalunits_jit

def stripKunits(Kn_tuple):
    """A wrapper function that strips the units from the Kn values returned by `Kunits` or `Kunits_jit.`"""

    units = ('1/nm','1/nm**2','1/nm**3')
    return(np.array([Kn.to(unit).magnitude for Kn, unit in zip(Kn_tuple, units)]))

class TestDissipation9bMethods(unittest.TestCase):
    """Verify that compiled code returns the same $K_0$, $K_1$, and $K_2$ values as pure Python code.
    
    Specifically, confirm that the functions K_jit, Kunits_jit, Kmetal_jit, and Kmetal_jit return the same
    $K_n$ values as K_jit, Kunits_jit, Kmetal_jit, Kmetal_jit for Type I, II, III, and IV samples.
    This amounts to a total of eight unit tests.
    """

    def setUp(self):

        cantilever = CantileverModel(
            f_c = ureg.Quantity(62, 'kHz'),
            k_c = ureg.Quantity(2.8, 'N/m'), 
            V_ts = ureg.Quantity(1, 'V'), 
            R = ureg.Quantity(55, 'nm'),
            angle = ureg.Quantity(20, 'degree'),
            L = ureg.Quantity(1000, 'nm')
        )

        sample1 = SampleModel1(
            cantilever = cantilever,
            h_s = ureg.Quantity(100, 'nm'),
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            epsilon_d = ureg.Quantity(complex(1e6, 0), ''),
            z_r = ureg.Quantity(100, 'nm')
        )

        sample2 = SampleModel2(
            cantilever = cantilever,
            epsilon_d = ureg.Quantity(complex(3, 0), ''),
            h_d = ureg.Quantity(20, 'nm'),
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        sample3 = SampleModel3(
            cantilever = cantilever,
            epsilon_s = ureg.Quantity(complex(20, 0), ''),
            sigma = ureg.Quantity(1e-7, 'S/m'),
            rho = ureg.Quantity(1e21, '1/m^3'),
            z_r = ureg.Quantity(100, 'nm')
        )

        sample4 = SampleModel4(
            cantilever = cantilever,
            z_r = ureg.Quantity(100, 'nm')
        )

        sample1_jit = SampleModel1Jit(**sample1.args())
        sample2_jit = SampleModel2Jit(**sample2.args())
        sample3_jit = SampleModel3Jit(**sample3.args())
        sample4_jit = SampleModel4Jit(**sample4.args())

        omega = 1e5
        loc1 = 1e-9 * np.array([ 10, 20, 50], dtype=np.float64)
        loc2 = 1e-9 * np.array([ 0,   0, 50], dtype=np.float64)

        self.params1_jit = {'integrand': integrand1jit, 
                            'sample': sample1_jit, 
                            'omega': omega, 
                            'location1': loc1, 
                            'location2': loc2}
        
        self.params2_jit = {'integrand': integrand2jit, 
                            'sample': sample2_jit, 
                            'omega': omega, 
                            'location1': loc1, 
                            'location2': loc2}

        self.params3_jit = {'integrand': integrand3jit, 
                            'sample': sample3_jit, 
                            'omega': omega, 
                            'location1': loc1, 
                            'location2': loc2}

        self.params4_jit = {'sample': sample4_jit, 
                            'location1': loc1, 
                            'location2': loc2}

        omega_u = ureg.Quantity(1e5, 'Hz')
        loc1_u = ureg.Quantity(np.array([ 10, 20, 50]), 'nm')
        loc2_u = ureg.Quantity(np.array([ 0,   0, 50]), 'nm')

        self.params1 = {'integrand': integrand1, 
                        'sample': sample1, 
                        'omega': omega_u, 
                        'location1': loc1_u, 
                        'location2': loc2_u}
        
        self.params2 = {'integrand': integrand2, 
                        'sample': sample2, 
                        'omega': omega_u, 
                        'location1': loc1_u, 
                        'location2': loc2_u}        

        self.params3 = {'integrand': integrand3, 
                        'sample': sample3, 
                        'omega': omega_u, 
                        'location1': loc1_u, 
                        'location2': loc2_u}       

        self.params4 = {'sample': sample4, 
                        'location1': loc1_u, 
                        'location2': loc2_u}

    def test_Sample1(self):
        """Do K and K_jit return the same result for a Type I sample?"""

        self.assertTrue(np.allclose(
            K_jit(**self.params1_jit), 
            K(**self.params1)))

    def test_Sample1units(self):
        """Do Kunits and Kunits_jit return the same result for a Type I sample?"""

        self.assertTrue(np.allclose(
            stripKunits(Kunits_jit(**self.params1_jit)),
            stripKunits(Kunits(**self.params1))))
        
    def test_Sample2(self):
        """Do K and K_jit return the same result for a Type II sample?"""

        self.assertTrue(np.allclose(
            K_jit(**self.params2_jit), 
            K(**self.params2)))

    def test_Sample2units(self):
        """Do Kunits and Kunits_jit return the same result for a Type II sample?"""

        self.assertTrue(np.allclose(
            stripKunits(Kunits_jit(**self.params2_jit)),
            stripKunits(Kunits(**self.params2))))
        
    def test_Sample3(self):
        """Do K and K_jit return the same result for a Type III sample?"""

        self.assertTrue(np.allclose(
            K_jit(**self.params3_jit), 
            K(**self.params3)))

    def test_Sample3units(self):
        """Do Kunits and Kunits_jit return the same result for a Type III sample?"""

        self.assertTrue(np.allclose(
            stripKunits(Kunits_jit(**self.params3_jit)),
            stripKunits(Kunits(**self.params3))))
        
    def test_Sample4(self):
        """Do K and K_jit return the same result for a Type III sample?"""

        self.assertTrue(np.allclose(
            Kmetal_jit(**self.params4_jit), 
            Kmetal(**self.params4)))
        
    def test_Sample4units(self):
        """Do K and K_jit return the same result for a Type III sample?"""

        self.assertTrue(np.allclose(
            stripKunits(Kmetalunits_jit(**self.params4_jit)), 
            stripKunits(Kmetalunits(**self.params4))))