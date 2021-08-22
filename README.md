# simple_gce

### Currently non-functional. The code was originally written to model a very specific system, I am in the process of re-structuring this code so that it can be applied more generally.

A simple one-zone Galactic Chemical Evolution model, as used in [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901). 

simple_gce solves the set of differential equations that represent a simplified model for an evolving galaxy.
This model allows for the infall of material from the Galactic halo (i.e. open box), and assumes homogeneous evolution (i.e. instantaneous mixing).
For a complete description of the equations and assumptions, see [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901), and [Kobayashi et al. (2000)](https://arxiv.org/abs/astro-ph/9908005).

I have tried to create a program that is both readable and efficient, but where I have had to choose between them, I opt for readability. This is so that the user can more easily understand the code structure and implement the software for their own purpose with a confident understanding of the parameters and functioning.


