# `simpleGCE`

A simple one-zone Galactic Chemical Evolution model, as used in [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901). 

`simpleGCE` solves the set of differential equations that represent a simplified model for an evolving galaxy.
This model allows for the infall of material from the Galactic halo (i.e. open box), and assumes homogeneous evolution (i.e. instantaneous mixing).
For a complete description of the equations and assumptions, see [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901), and [Kobayashi et al. (2000)](https://arxiv.org/abs/astro-ph/9908005).

## Installation

### From source
```
$ git clone https://github.com/jamesgrimmett/simple_gce.git /path/to/local/install
$ cd /path/to/local/install
$ pip install -e .
```

## Getting started
### Environment setup
Ensure that the package dependencies are installed within your local `python` environment. The easiest way to do this is to create a new conda environment, see the [anaconda docs](https://docs.conda.io/projects/conda/en/latest/index.html) if you have not used it before;
```
$ conda env update -f environment.yml
$ conda activate gce
```
### Quick start
To run with the example setup;
```
$ python -m simple_gce -t 13.e9 -o ./
```
For a description of the usage, type
```
$ python -m simple_gce --help
```
The output will be save in your current directory as a single `csv` file.

To run a different example, copy another example config file and run again. E.g.,
```
$ cp simple_gce/example_configs/grimmett-2020_config.py simple_gce/config.py
$ python -m simple_gce -t 13.e9 -o ./
```