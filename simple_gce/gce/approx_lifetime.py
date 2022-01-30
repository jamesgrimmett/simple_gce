"""
A class to calculate stellar lifetimes as a function of mass and metallicity.
Values are interpolated using the data provided by Portinari et al. 1997,
Table 14 (http://arxiv.org/abs/astro-ph/9711337).
This data has mass in range 0.6 Msun - 120 Msun, and Z in range 4.e-4 - 0.05.
Interpolating for values outside this range will return the boundary value of the data.

Example
-------
>>> lt = approx_lifetime.ApproxLifetime()
>>> lt.lifetime(mass=1.0, z=0.02)
>>> 1.e10
>>> lt.mass(lifetime=1.e10, z=0.02)
>>> 1.0
"""

import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy import interpolate

from .. import config


class ApproxLifetime(object):
    def __init__(self):
        # Load 2-D array of lifetime data in mass-metallicity space
        lifetime_data = self._read_portinari_lifetimes()
        # Create function to interpolate over data
        f_lifetime = self._interpolate_lifetimes(lifetime_data)
        self.f_lifetime = f_lifetime

        # Setup arrays for fast mass approximation
        mass_min = lifetime_data.M.min()
        mass_max = lifetime_data.M.max()
        # Add step size to mass_max in range in order to include end point
        self.mass_arr = np.array([m for m in np.arange(mass_min, mass_max + 0.1, 0.1)])
        self.lifetime_arr = np.array([self.lifetime(m, 0.0) for m in self.mass_arr])

    def lifetime(self, mass: float, z: float) -> float:
        """Returns the lifetime for a star of given mass and metallicity.

        Parameters
        ----------
        mass: float
            The stars mass in solar masses.
        z: float
            The stars metallicity.
        Returns
        -------
        float
            Stellar lifetime in years.
        """

        f_lifetime = self.f_lifetime

        t = f_lifetime(mass, z)

        return t

    def mass(self, lifetime: float, z: float) -> float:
        """Returns the expected stellar mass for a given lifetime and metallicity.

        Parameters
        ----------
        t: float
            Stellar lifetime in years.
        z: float
            Stellar metallicity.
        Returns
        -------
        float
            The stellar mass in solar masses.
        """

        m_arr = self.mass_arr
        arr = np.array([self.lifetime(m, z) for m in m_arr])

        m = m_arr[abs(arr - lifetime).argmin()]

        return m

    def mass_approx(self, lifetime: float) -> float:
        """
        Returns an approximate stellar mass for a given lifetime, assuming Z = 0.
        Faster but less accurate than self.mass()

        Parameters
        ----------
        t: float
            Stellar lifetime in years.
        Returns
        -------
        float
            The stellar mass in solar masses.
        """

        arr = self.lifetime_arr
        m_arr = self.mass_arr
        m = m_arr[abs(arr - lifetime).argmin()]

        return np.round(m, 2)

    def _read_portinari_lifetimes(self):
        """
        Reads the stellar lifetime data in CSV format and returns a dataframe.
        The lifetimes are taken from Portinari et al. 1997, Table 14.
        http://arxiv.org/abs/astro-ph/9711337
        """

        base_dir = config.INSTALL_DIR
        data_path = os.path.join(base_dir, "data/lifetimes_portinari.csv")

        data = pd.read_csv(data_path, comment="#", sep="\s+")

        return data

    def _interpolate_lifetimes(self, data: pd.DataFrame) -> Callable:
        """Interpolates between mass and metallicity for stellar lifetimes.

        Parameters
        ----------
        data: pd.DataFrame
            The stellar lifetime data at discrete mass/metallicity data points. Must be
            a pandas dataframe with the first column being the masses, and each column
            following being the lifetime in years for each metallicity, in ascending
            order. The column labels must be the metallicity value.
            E.g.
            M       0.0004      0.004       0.008       0.02        0.05
            0.6     4.28E+10    5.35E+10    6.47E+10    7.92E+10    7.18E+10
            0.7     2.37E+10    2.95E+10    3.54E+10    4.45E+10    4.00E+10
            ... etc.
        Returns
        -------
        Callable
            An interpolation function.
        """
        z_vals = np.array(data.columns[1:])

        x = np.array(data.M)
        y = np.array([float(z) for z in z_vals])
        z = np.array(data[z_vals])

        # f = interpolate.RectBivariateSpline(x,y,z)
        # TODO: Consider interpolating in log-space
        f = interpolate.interp2d(x, y, np.transpose(z))
        return f


def fill_lifetimes(df: pd.DataFrame):
    """Use the ApproxLifetime class to fill the lifetime column in a dataframe."""
    lt = ApproxLifetime()

    df.loc[:, "lifetime"] = [lt.lifetime(row["mass"], row["Z"]) for _, row in df.iterrows()]

    return df
