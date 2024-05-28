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

``data.py``
-----------

Functions to help compare numerically computed friction to Marohn-group data.

.. automodule:: data
    :members:
    :undoc-members:
    :show-inheritance: