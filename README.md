# dissipationtheory
Compute atomic force microscope cantilever dissipation and frequency noise over metals and dielectrics.  Compute cantilever capacitance using a sphere plus a cone model.

## Theory

This package impliments models of tip-sample capacitance used in the following papers.

1. [**Hoepker2011oct**] Dielectric Fluctuations over Polymer Films Detected Using an Atomic Force Microscope. Nikolas Hoepker, Swapna Lekkala, Roger F. Loring, and John A. Marohn. *J. Phys. Chem. B* (2011) 115(49):14493-14500; https://doi.org/10.1021/jp207387d.  Equations 4 (frequency shift) and 22 (cone-plane capacitance).  The paper directs you to the supplement for the cone-plane capacitance, but the information is not there (sorry). 

2. [**Hoepker2013jan**] Fluctuations near Thin Films of Polymers, Organic Photovoltaics, and Organic Semiconductors Probed by Electric Force Microscopy. Nikolas C Hoepker. Cornell University, 2013; http://hdl.handle.net/1813/33910.  Equations 2.49 (cone-plane capacitance) and equations 2.51 and 2.52 (sphere-plane capacitance).

3. [**Cherniavskaya2003feb**] Quantitative Noncontact Electrostatic Force Imaging of Nanocrystal Polarizability. Oksana Cherniavskaya, Liwei Chen, Vivian Weng, Leonid Yuditsky, and Louis E. Brus. *J. Phys. Chem. B* (2003) 107(7):1525-1531; https://doi.org/10/fqzfmz.  Figure 4 is helpful.  Equation 19 (cone-plane capacitance second derivative).

## Installation

```python
import numpy as np
```