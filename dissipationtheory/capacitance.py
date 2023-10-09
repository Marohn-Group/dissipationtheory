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
    """Capacitance (and derivatives) of a sphere above a metallic ground plane.

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

def C2SphereCone(height, radius, theta, nterm=21):  
    """Second derivative of the capacitance of a sphere plus a cone above a ground plane.

    :param pint.util.Quantity height: sphere-to-plane separation
    :param pint.util.Quantity radius: tip sphere radius
    :param pint.util.Quantity theta: tip cone angle
    :param int nterm: number of terms in the sphere expansion

    """
    theta_rad = theta.to('radian').magnitude
    beta = np.log((1 + np.cos(theta_rad))/(1 - np.cos(theta_rad)))
    
    C2cone = (8*np.pi*epsilon0/(beta * beta * height)).to('F/m^2')
    C2sphere = Csphere(2, height, radius, nterm).to('F/m^2')
    
    return C2cone + C2sphere

def C2SphereConefit(height, radius, theta):
    """A curve-fitting version of C2SphereCone(), where the input parameters are unitless numbers.  
     
    :param float height: tip-sample separation in nanometers
    :param float radius: tip sphere radius in nanometers
    :param float theta: tip cone angle in degrees

    For simplicity, this function will use the default number of terms in the capacitance expansion.
    
    """
    
    return C2SphereCone(
        ureg.Quantity(height, 'nm'),
        ureg.Quantity(radius, 'nm'),
        ureg.Quantity(theta, 'degrees')).to('mF/m^2').magnitude

def FrictionCoefficientThickfit(height, a, R, fc):
    """A function for fitting perpendicular dissipation collected over a semi-infinite, high dielectric constant sample.  
    
    :param np.array x: an array of tip-sample separation in nm
    :param float a: unitless overall scale factor -- the fit parameter    
    :param float R: tip sphere radius in nm
    :param float fc: cantilever resonance frequency in Hz

    """

    h = ureg.Quantity(height, 'nm')
    r = ureg.Quantity(R, 'nm')
    c0 = Csphere(0, height=h, radius=r, nterm=21)
    c1 = Csphere(1, height=h, radius=r, nterm=21)
    d = h + r
    t1 = (c0 * c0)/d**3
    t2 = - (c0*c1)/d**2
    t3 = (c1 * c1)/d
    omega = 2 * np.pi * ureg.Quantity(fc, '1/s')
    prefactor = 1/(4 * np.pi * epsilon0 * omega)
    expression = (prefactor * (t1 + t2 + t3)).to('pN s/(V^2 m)').magnitude
    
    return a * expression

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
