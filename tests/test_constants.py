from dissipationtheory.constants import kb, Troom, qe
import pytest


def test_Troom():
    """kb T/q is approximately 25.693 mV at room temperature"""
    c = (kb*Troom/qe).to('mV').magnitude
    assert 25.693 == pytest.approx(c, 0.001)