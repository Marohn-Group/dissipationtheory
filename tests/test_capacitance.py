from dissipationtheory.constants import ureg, epsilon0
from dissipationtheory.capacitance import Csphere
import numpy as np
import pytest


def test_Csphere_infinity():
    """Csphere is approximated by 4*pi*epsilon_0*r at large separations."""

    r = ureg.Quantity(1, "m")
    h = ureg.Quantity([1000], "m")

    Cexact = (4 * np.pi * epsilon0 * r).to("pF")
    Capprox = Csphere(0, h, r).to("pF")

    assert Cexact.magnitude == pytest.approx(Capprox.magnitude, rel=1e-3)
