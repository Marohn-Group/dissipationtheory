# dissipationtheory

Compute atomic force microscope cantilever dissipation and frequency noise over metals and dielectrics.  Compute cantilever capacitance using a sphere plus a cone model.

## Theory

This package implements models of tip-sample capacitance, friction, frequency noise, and frequency shift published in the following manuscripts.

1. [**Cherniavskaya2003feb**] Cherniavskaya, O.; Chen, L.; Weng, V.; Yuditsky, L.; Brus, L. E. Quantitative Noncontact Electrostatic Force Imaging of Nanocrystal Polarizability. *J. Phys. Chem. B* **2003**, *107 (7)*, 1525. https://doi.org/10.1021/jp0265438. Figure 4 is helpful.  Equation 19 (cone-plane capacitance second derivative).

2. [**Kuehn2006apr**] Kuehn, S.; Loring, R. F.; Marohn, J. A. Dielectric Fluctuations and the Origins of Noncontact Friction. *Phys. Rev. Lett.* **2006**, *96 (15)*, 156103. https://doi.org/10.1103/PhysRevLett.96.156103.

3. [**Yazdanian2008jun**] Yazdanian, S. M.; Marohn, J. A.; Loring, R. F. Dielectric Fluctuations in Force Microscopy: {N}oncontact Friction and Frequency Jitter. *J. Chem. Phys.* **2008**, *128 (22)*, 224706. https://doi.org/10.1063/1.2932254.

4. [**Yazdanian2009jun**] Yazdanian, S. M.; Hoepker, N.; Kuehn, S.; Loring, R. F.; Marohn, J. A. Quantifying Electric Field Gradient Fluctuations over Polymers Using Ultrasensitive Cantilevers. *Nano Lett.* **2009**, *9 (6)*, 2273. https://doi.org/10.1021/nl9004332.  Supplement to: Quantifying Electric Field Gradient Fluctuations over Polymers Using Ultrasensitive Cantilevers. *Nano Lett.* **2009**, *9 (10)*, 3668. https://doi.org/10.1021/nl901850b.

5. [**Hoepker2011dec**] Hoepker, N.; Lekkala, S.; Loring, R. F.; Marohn, J. A. Dielectric Fluctuations over Polymer Films Detected Using an Atomic Force Microscope. *J. Phys. Chem. B* **2011**, *115 (49)*, 14493. https://doi.org/10.1021/jp207387d.  Equations 4 (frequency shift) and 22 (cone-plane capacitance).  The paper directs you to the supplement for the cone-plane capacitance, but the information is not there (sorry).

6. [**Hoepker2013jan**] Hoepker, N.  Fluctuations near Thin Films of Polymers, Organic Photovoltaics, and Organic Semiconductors Probed by Electric Force Microscopy.  PhD thesis, Cornell University **2013**. http://hdl.handle.net/1813/33910.  Equations 2.49 (cone-plane capacitance) and equations 2.51 and 2.52 (sphere-plane capacitance).

7. [**Lekkala2012sep**] Lekkala, S.; Hoepker, N.; Marohn, J. A.; Loring, R. F. Charge Carrier Dynamics and Interactions in Electric Force Microscopy. *J. Chem. Phys.* **2012**, *137 (12)*, 124701. https://doi.org/10.1063/1.4754602.

8. [**Lekkala2013nov**] Lekkala, S.; Marohn, J. A.; Loring, R. F. Electric Force Microscopy of Semiconductors: Theory of Cantilever Frequency Fluctuations and Noncontact Friction. *J. Chem. Phys.* **2013**, *139 (18)*, 184702. https://doi.org/10.1063/1.4828862.

We plan to incorporate equations from the following two papers in the near future.

9. [**Loring2022sep**] Loring, R. F. Noncontact Friction in Electric Force Microscopy over a Conductor with Nonlocal Dielectric Response. *J. Phys. Chem. A* **2022**, *126 (36)*, 6309. https://doi.org/10.1021/acs.jpca.2c04428.

10. [**Loring2023jul**] Loring, R. F. Voltage Fluctuations and Probe Frequency Jitter in Electric Force Microscopy of a Conductor. *J. Chem. Phys.* **2023**, *159 (4)*, 044703. https://doi.org/10.1063/5.0160556.


## Installation

## Install the development version

### Via poetry

I am using the poetry tool ([link](https://python-poetry.org/)) for dependency management and packaging.  So install the poetry tool.  I usually run the conda python distribution at the command line, deactivate conda before installing poetry.

```
$ conda deactivate
$ curl -sSL https://install.python-poetry.org | python3 -
$ open ${HOME}/.bash_profile
```

and in the `.bash_profile` file add

```
export PATH="/Users/jam99/.local/bin:$PATH"
```

As follows, run the `.bash_profile` file to update the path.  This will reactivate conda, so deactivate it again.  We should now be able to run poetry at the command line.

```
$ source ${HOME}/.bash_profile
$ conda deactivate
$ poetry --version
Poetry (version 1.4.2)
```

There is as yet no python available at the command line.  Next create a `pyproject.toml` file that will specify the virtual environment (i.e., tells poetry what version of python to run and what packages to load).  To create the `pyproject.toml` file I ran `poetry init` and answered the questions.  Afterwards, invoke the virtual environment as follows.  Verify that python is now running at the command line.

```
$ poetry shell
$ python --version
Python 3.8.5
```

While python is installed, none of the packages are:

```
$ python -c "import numpy as np"
ModuleNotFoundError: No module named 'numpy'
```

To install the packages run

```
$ poetry install --all-extras
```

and be prepared to wait while the many dependent packages are installed.  We now have a `poetry.lock` file present.  To run the unit tests,

```
$ poetry run pytest
```

After updating the package dependencies in `pyproject.toml`, I'll run

```
$ poetry lock
$ poetry install
```

Rememember to deactivate any conda environments before you start, to avoid errors resulting from package collisions.  If you are coming back to work on the package, the working commands are

```
$ conda deactivate
$ poetry lock
$ poetry install --all-extras
$ poetry run pytest
```

Note: I upgraded the python version in the `.toml` file from 3.8 to 3.9 but then got an error trying to launch `poetry shell`.  First I have to run

```
$ poetry env use 3.9
```

which sets up a new virtual environment.  Now I can proceed with adding a package that requires python 3.9, like sphinx

```
$ poetry add sphinx
$ poetry lock
$ poetry install --all-extras
$ poetry run pytest
```


### Adding and running jupyter

To install,

```
$ poetry add -D jupyter
$ jupyter notebook
```

Because I want to run the notebook in MS Visual Studio Code, I'll need to create a named Jupyter kernel based on this poetry environment.

```
$ poetry run ipython kernel install --user --name=dissipationtheory
Installed kernelspec dissipationtheory in /Users/jam99/Library/Jupyter/kernels/dissipationtheory
```

### Via conda

Alternatively

```
$ conda create -n dissipationtheory python=3.8
$ pip install .[test]
$ python -m pytest
```
