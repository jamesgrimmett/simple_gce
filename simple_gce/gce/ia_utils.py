"""A set of methods for calculate the rate of SNe Ia.

See Eq. 12 of Kobayashi (2000) - https://arxiv.org/abs/astro-ph/9908005
"""

import numpy as np
from scipy.integrate import fixed_quad as f_quad_int

from simple_gce.gce import approx_lifetime


lt = approx_lifetime.ApproxLifetime()


def calc_ia_rate_fast(galaxy: object) -> float:
    """Calculate the rate of SNe Ia occuring in the Galaxy.

    Accuracy is exchanged for speed in this function (fixed_quad and lt.mass_approx).
    """
    ia_system = galaxy.ia_system
    turnoff_mass = lt.mass_approx(lifetime=galaxy.time)
    # Progenitor (WD) component
    mpl = max(ia_system.mpl, turnoff_mass)
    mpu = ia_system.mpu
    if mpl >= mpu:
        rate_ia = 0.0
        return rate_ia

    progenitor_component = galaxy.imf._integrate_mod(lower=mpl, upper=mpu)

    # WD - RG scenario
    mdl_rg = max(ia_system.mdl_rg, turnoff_mass)
    mdu_rg = ia_system.mdu_rg
    b_rg = ia_system.b_rg
    if mdl_rg >= mdu_rg:
        rate_rg = 0.0
    else:
        rate_rg_1 = b_rg * progenitor_component
        rate_rg_2 = f_quad_int(
            lambda x: _ia_donor_integrand(x, ia_system.imf_ia_rg, galaxy),
            a=mdl_rg,
            b=mdu_rg,
            n=len(np.arange(mdl_rg, mdu_rg, 0.01)),
        )[0]
        rate_rg = rate_rg_1 * rate_rg_2

    # WD - MS scenario
    b_ms = ia_system.b_ms
    mdl_ms = max(ia_system.mdl_ms, turnoff_mass)
    mdu_ms = ia_system.mdu_ms
    if (mdl_ms >= mdu_ms) or (mpl >= mpu):
        rate_ms = 0.0
    else:
        rate_ms_1 = b_ms * progenitor_component
        rate_ms_2 = f_quad_int(
            lambda x: _ia_donor_integrand(x, ia_system.imf_ia_ms, galaxy),
            a=mdl_ms,
            b=mdu_ms,
            n=len(np.arange(mdl_ms, mdu_ms, 0.01)),
        )[0]
        rate_ms = rate_ms_1 * rate_ms_2

    rate_ia = rate_rg + rate_ms

    if np.isnan(rate_ia):
        raise RuntimeError("Error in calculation of Ia rate.")

    return rate_ia


def _ia_donor_integrand(m: float, imf: float, galaxy: object) -> float:
    """The integrand of the second intregal appearing in Eq. 12 of Kobayashi (2006).

    See Eq. 12 of https://arxiv.org/abs/astro-ph/9908005
    """
    imf_m = imf.functional_form(m)
    lifetime_m = lt.lifetime(mass=m, z=galaxy.z)
    t = galaxy.time - lifetime_m
    # TODO: sometimes slight difference between integration lower limit and
    #       true turnoff mass. Need to investigate.
    t = t.clip(min=0.0)
    sfr_m = galaxy._get_historical_value("sfr", t)

    integrand = 1.0 / m * sfr_m * imf_m
    if np.isnan(integrand).any():
        raise RuntimeError("Error in calculation of Ia rate.")

    return integrand
