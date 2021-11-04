# simple_gce

### Work in Progress. When I originally wrote this program, I did not know very much about software development best practices. I am in the process of re-structuring the original code so that it is tidier, more robust, well-tested, and of a higher quality in general.

A simple one-zone Galactic Chemical Evolution model, as used in [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901). 

simple_gce solves the set of differential equations that represent a simplified model for an evolving galaxy.
This model allows for the infall of material from the Galactic halo (i.e. open box), and assumes homogeneous evolution (i.e. instantaneous mixing).
For a complete description of the equations and assumptions, see [Grimmett et al. (2020)](https://arxiv.org/abs/1911.05901), and [Kobayashi et al. (2000)](https://arxiv.org/abs/astro-ph/9908005).

I have tried to create a program that is both readable and efficient, but where I have had to choose between them, I opt for readability. This is so that the user can more easily understand the code structure and implement the software for their own purpose with a confident understanding of the parameters and functioning.

### Example Run
```Python
from simple_gce.gce import galaxy

g = galaxy.Galaxy()

steps = 10000
gm = np.zeros(steps)
sm = np.zeros(steps)
t = np.zeros(steps)
x_idx = g.x_idx
zn = np.zeros(steps)
fe = np.zeros(steps)
h = np.zeros(steps)

for i in range(steps):
    g.evolve(dt=1.e6)
    gm[i] = float(g.gas_mass)
    sm[i] = float(g.star_mass)
    h[i] = float(g.x[x_idx['H']])
    fe[i] = float(g.x[x_idx['Fe']])
    zn[i] = float(g.x[x_idx['Zn']])
    t[i] = float(g.time)
    if g.time%1.e8 == 0:
        print(f'time: {g.time*1.e-9}')
```
