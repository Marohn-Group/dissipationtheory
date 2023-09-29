from sympy import symbols, acosh, sinh, lambdify
import numpy as np
from dissipationtheory.constants import ureg, epsilon0

height, R, power = symbols("d R n")
CsphereTermSymbolic = {}
CsphereTermSymbolic[0] = (
    R * sinh(acosh(1 + height / R)) / sinh(power * acosh(1 + height / R))
)
CsphereTermSymbolic[1] = CsphereTermSymbolic[0].diff(height)
CsphereTermSymbolic[2] = CsphereTermSymbolic[0].diff(height).diff(height)

CsphereTerm = {}
for key in CsphereTermSymbolic.keys():
    CsphereTerm[key] = lambdify([height, R, power], CsphereTermSymbolic[key])


def Csphere(index, height, radius, nterm=20):
    """Capacitance (and derivatives) of a sphere above a ground plane.

    :param integer index: 0 for capacitance, 1 for 1st derivative, 2 for 2nd derivative
    :param pint.util.Quantity height: sphere-to-plane separation
    :param pint.util.Quantity radius: sphere radius
    :param int nterm: number of terms in the expansion

    """
    H = ureg.Quantity(np.outer(height.magnitude, np.ones(nterm)), height.units)
    n = np.outer(np.ones(len(height)), np.arange(1, nterm + 1))
    return (
        4
        * np.pi
        * epsilon0
        * CsphereTerm[index](H, radius, n).sum(axis=1).to_base_units()
    )

def CsphereOverSemi(index, height, radius, epsilon, nterm=20):
    """Capacitance (and derivatives) of a sphere above a dielectric substrate.

    :param integer index: 0 for capacitance, 1 for 1st derivative, 2 for 2nd derivative
    :param pint.util.Quantity height: sphere-to-plane separation
    :param pint.util.Quantity radius: sphere radius
    :param float: real part of the substrates dielectric constant
    :param int nterm: number of terms in the expansion

    """
    H = ureg.Quantity(np.outer(height.magnitude, np.ones(nterm)), height.units)
    n = np.outer(np.ones(len(height)), np.arange(1, nterm + 1))
    return (
        4
        * np.pi
        * epsilon0
        * (np.power((1 - (1 / epsilon))/(1 + (1 / epsilon)), n - 1) * \
          CsphereTerm[index](H, radius, n)).sum(axis=1).to_base_units()
    )

def main():
    import matplotlib.pylab as plt

    h = ureg.Quantity(np.logspace(1, 4, num=500), "nm")
    r = ureg.Quantity(20, "nm")
    ureg.default_format = "~P"

    Cref = (4 * np.pi * epsilon0 * r).to("aF")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(3.25, 5))
    fig.suptitle("radius $r = ${:0.2f} {} sphere".format(r.magnitude, r.units))

    y = {}
    keys = CsphereTermSymbolic.keys()

    for unit, key in zip(["aF", "pF/m", "uF/(m**2)"], keys):
        y[key] = Csphere(key, h, r).to(unit)

    axs[0].semilogx(h.magnitude, y[0].magnitude)
    axs[1].loglog(h.magnitude, -y[1].magnitude)
    axs[2].loglog(h.magnitude, y[2].magnitude)

    axs[2].set_xlabel("height $h$ [nm]")
    for label, key in zip(["$C(h)$", "$-dC/dz$", "$d^2C/dz^2$"], keys):
        axs[key].set_ylabel(label + " [{:}]".format(y[key].units))

    fig.align_ylabels()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
