[![Build Status](https://travis-ci.org/qiqi/fds.svg?branch=master)](https://travis-ci.org/qiqi/fds.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/algorithmic-music-exploration/amen/badge.svg?branch=master)](https://coveralls.io/github/algorithmic-music-exploration/amen?branch=master)
[![Dependency Status](https://dependencyci.com/github/algorithmic-music-exploration/amen/badge)](https://dependencyci.com/github/algorithmic-music-exploration/amen)
[![Documentation Status](https://readthedocs.org/projects/amen/badge/?version=latest)](http://amen.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/badge/license-BSD-blue.svg)](https://raw.githubusercontent.com/algorithmic-music-exploration/amen/master/LICENSE)

# What's it for

fds is a research tool for computational simulations that exhibis
chaotic dynamics. It computes sensitivity derivatives of time averaged
quantities, a.k.a. statistics, with respect simulation parameters.

For an introduction of chaotic dynamics, I highly recommend [Strogatz's
excellent book](https://www.amazon.com/gp/product/0813349109). For a
statistical view of chaotic dynamical systems, please refer to
[Berlinger's article](http://www.uvm.edu/~pdodds/files/papers/others/1992/berliner1992a.pdf).
Algorithm used in this software is described in [the upcoming AIAA
paper](https://dl.dropbox.com/s/2e9jxjmwh375i01/fds.pdf)

# Download and use

The best way to download fds is using pip. Pip is likely included in
your Python installation. If not, see [instruction
here](https://pip.pypa.io/en/stable/installing/). To install fds
using pip, simply type

```
    sudo pip install fds
```

# Tutorials

-  `First example -- Van der Pol oscillator in Python <tutorials/src/vanderpol_python/vanderpol>`__
-  `Lorenz attractor in C <docs/tutorials/lorenz_c.md>`__
-  `Lorenz 96 in MPI and C <docs/tutorials/lorenz96_mpi.md>`__

# Guides

-  `Chaos and statistical convergence <docs/guides/statistics.md>`__
-  `Lyapunov exponents and time
   segmentation <docs/guides/lyapunov.md>`__
-  `Save and restart <docs/guides/save_restart.md>`__

# Reference

-  `The least squares shadowing algorithm <docs/ref/lss_algorithm.md>`__
-  `Function reference <docs/ref/function_ref.md>`__
-  `License <LICENSE.md>`__
