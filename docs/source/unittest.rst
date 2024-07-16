unit testing
============

``test_constants.py``
---------------------

Test the ``constant.py`` package by checking that combinations of constants, converted to a different unit, gives the expected answer.

.. automodule:: test_constants
    :members:
    :undoc-members:
    :show-inheritance:

``test_capacitance.py``
-----------------------

Test the ``capacitance.py`` package by checking that the capacitance of a sphere at close to infinite distance from the ground plane is approximately :math:`4 \pi \epsilon_0 r`, with :math:`r` the radius of the sphere.

.. automodule:: test_capacitance
    :members:
    :undoc-members:
    :show-inheritance:

``test_dissipation``
--------------------

Test the numerical calculations of non-contact friction in the ``dissipation.py`` package by comparing the calculated friction to (1) digitized prior calculations, performed in *Mathematica* by Lekkala, and (2) an approximate analytical solution for the friction valid at low charge density, derived by Marohn.

.. automodule:: test_dissipation
    :members:
    :undoc-members:
    :show-inheritance:
    
``test_dissipation2``
---------------------

Test the numerical calculations of non-contact friction in the ``dissipation2.py`` package analogously.

.. automodule:: test_dissipation
    :members:
    :undoc-members:
    :show-inheritance: