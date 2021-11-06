# simple_gce

### Work in Progress. When I originally wrote this program, I was still new to coding and did not know very much about software development best practices. I am in the process of re-structuring the original code so that it is tidier, more robust, well-tested, properly documented, and of a higher quality in general.

A simple one-zone Galactic Chemical Evolution model, as used in [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901). 

simple_gce solves the set of differential equations that represent a simplified model for an evolving galaxy.
This model allows for the infall of material from the Galactic halo (i.e. open box), and assumes homogeneous evolution (i.e. instantaneous mixing).
For a complete description of the equations and assumptions, see [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901), and [Kobayashi et al. (2000)](https://arxiv.org/abs/astro-ph/9908005).

I have tried to create a program that is both readable and efficient, but where I have had to choose between them, I will generally opt for readability. This is so that the user can more easily understand the code structure and implement the software for their own purpose with a confident understanding of the parameters and functioning.

### Example Usage
```
conda env update -f environment.yml
conda activate gce
pip install -e .

python -m simple_gce -t 13.e9 -o ./
```
