# dissipationtheory

Compute atomic force microscope cantilever dissipation and frequency noise over metals and dielectrics.  Compute cantilever capacitance using a sphere plus a cone model.

## Theory

This package impliments models of tip-sample capacitance used in the following sources.

1. [**Hoepker2011oct**] Dielectric Fluctuations over Polymer Films Detected Using an Atomic Force Microscope. Nikolas Hoepker, Swapna Lekkala, Roger F. Loring, and John A. Marohn. *J. Phys. Chem. B* (2011) 115(49):14493-14500; https://doi.org/10.1021/jp207387d.  Equations 4 (frequency shift) and 22 (cone-plane capacitance).  The paper directs you to the supplement for the cone-plane capacitance, but the information is not there (sorry). 

2. [**Hoepker2013jan**] Fluctuations near Thin Films of Polymers, Organic Photovoltaics, and Organic Semiconductors Probed by Electric Force Microscopy. Nikolas C Hoepker. Cornell University, 2013; http://hdl.handle.net/1813/33910.  Equations 2.49 (cone-plane capacitance) and equations 2.51 and 2.52 (sphere-plane capacitance).

3. [**Cherniavskaya2003feb**] Quantitative Noncontact Electrostatic Force Imaging of Nanocrystal Polarizability. Oksana Cherniavskaya, Liwei Chen, Vivian Weng, Leonid Yuditsky, and Louis E. Brus. *J. Phys. Chem. B* (2003) 107(7):1525-1531; https://doi.org/10/fqzfmz.  Figure 4 is helpful.  Equation 19 (cone-plane capacitance second derivative).

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

There is as yet no python available at the command line.  Next create a `pyproject.toml` file that will specify the virtual environment (i.e., tells poetry what version of python to run and what packages to load).  To create the `pyproject.toml` file I ran `poetry init` and answered the questions.  Afterwards, invoke the virtual enviroment as follows.  Verify that python is now running at the command line.

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

### Via conda

Alternatively

```
$ conda create -n dissipationtheory python=3.8
$ pip install .[test]
$ python -m pytest
```