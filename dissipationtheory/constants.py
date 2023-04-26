from pint import UnitRegistry

ureg = UnitRegistry()
kb = ureg.Quantity(1.380649e-23, 'J/K')
epsilon0 = ureg.Quantity(8.8541878128e-12, 'C/(V m)')
qe = ureg.Quantity(1.60217663e-19, 'C')
Troom = ureg.Quantity(298.15, 'K')