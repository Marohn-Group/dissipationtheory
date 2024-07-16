dissipationtheory
=================

``constants.py``
----------------

Use the ``pint`` package to add units to physical quantities.  Create a unit registry ``ureg`` and define some handy constants: ``kb``, Boltzmann's constant; ``epsilon0``, free-space permittivity; and ``qe``, the unit of charge, i.e., the absolute value of the electron charge.

.. automodule:: constants
    :members:
    :undoc-members:
    :show-inheritance:

``capacitance.py``
------------------

Formulas related to the capacitance of a sphere and a cone.  Sphere capacitance derivatives are computed using the `sympy` package to carry out term-wise analytic differentiation of a finite sum.

.. automodule:: capacitance
    :members:
    :undoc-members:
    :show-inheritance:

``dissipation.py``
------------------

The main show.

.. automodule:: dissipation
    :members:
    :undoc-members:
    :show-inheritance:
    
``dissipation2.py``
------------------

The same functions as in ``dissipation.py``, but with conductivity and charge density as the dependent variables, instead of mobility and charge density.  The classes ``SampleModel1``, ``SampleModel2``, ``SampleModel1Jit``, and ``SampleModel2Jit`` have been tweaked; the other functions remain the same.

.. automodule:: dissipation
    :members:
    :undoc-members:
    :show-inheritance:

``data.py``
-----------

Functions to help compare numerically computed friction and BLDS spectra to Marohn-group data. When fitting BLDS data, the free variables are charge density and charge mobility.

.. automodule:: data
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
    
``data2.py``
------------

Functions to help compare numerically computed BLDS spectra to Marohn-group data. When fitting BLDS data, the free variables are charge mobility and charge density.

.. automodule:: data
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__