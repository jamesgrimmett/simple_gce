"""
An object to return stellar lifetimes as a function of mass and metallicity.
Returned values are interpolated using the data provided by Portinari et al. 1997, 
Table 14 (http://arxiv.org/abs/astro-ph/9711337).
This data has mass in range 0.6 Msun - 120 Msun, and Z in range 4.e-4 - 0.05. 
Interpolating for values outside this range will return the boundary value of the data.
"""

from .. import config
import os
import pandas as pd
import numpy as np
from scipy import interpolate

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
        self.mass_arr = np.array([m for m in np.arange(mass_min,mass_max+0.1,0.1)])
        self.lifetime_arr = np.array([self.lifetime(m,0.0) for m in self.mass_arr])
    

    def lifetime(self,mass,z):
        """
        Returns the lifetime for a star of given mass and metallicity
        
        Args:
            mass: The stars mass.
            z: The stars metallicity.
        Returns:
            Stellar lifetime in years.
        """

        f_lifetime = self.f_lifetime
    
        t = f_lifetime(mass,z)
    
        return t
    
    def mass(self,lifetime,z):
        """
        Returns the expected stellar mass for a given lifetime and metallicity.
        
        Args:
            t: Stellar lifetime.
            z: Stellar metallicity.
        Returns:
            The stellar mass in solar masses.
        """

        m_arr = self.mass_arr
        arr = np.array([self.lifetime(m,z) for m in m_arr])
    
        m = m_arr[abs(arr - lifetime).argmin()]
    
        return m

    def mass_approx(self,lifetime):
        """
        Returns an approximate stellar mass for a given lifetime, assuming Z = 0.
        Faster but less accurate than self.mass()
        
        Args:
            t: Stellar lifetime.
        Returns:
            The stellar mass in solar masses.
        """

        arr = self.lifetime_arr
        m_arr = self.mass_arr
        m = m_arr[abs(arr - lifetime).argmin()]
    
        return np.round(m,2)
    
    def _read_portinari_lifetimes(self):
        """
        Reads the stellar lifetime data in CSV format and returns a dataframe.
        The lifetimes are taken from Portinari et al. 1997, Table 14.
        http://arxiv.org/abs/astro-ph/9711337
        """
    
        base_dir = config.INSTALL_DIR
        data_path = os.path.join(base_dir,'data/lifetimes_portinari.csv')
    
        data = pd.read_csv(data_path,comment='#',sep='\s+')
    
        return data

    def _interpolate_lifetimes(self,data):
        """
        Interpolates between mass and metallicity for stellar lifetimes.
    
        Args:
            data: The stellar lifetime data at discrete mass/metallicity
                    data points. Must be a pandas dataframe with the first 
                    column being the masses, and each column following being
                    the lifetime in years for each metallicity, in ascending
                    order. The column labels must be the metallicity value.
                    E.g.
                    M       0.0004      0.004       0.008       0.02        0.05
                    0.6     4.28E+10    5.35E+10    6.47E+10    7.92E+10    7.18E+10
                    0.7     2.37E+10    2.95E+10    3.54E+10    4.45E+10    4.00E+10 
                    ... etc.
        Returns:
            An interpolation function.
        """
        z_vals = np.array(data.columns[1:])
    
        x = np.array(data.M)
        y = np.array([float(z) for z in z_vals])
        z = np.array(data[z_vals])
    
        #f = interpolate.RectBivariateSpline(x,y,z)
        # TODO: Consider interpolating in log-space
        f = interpolate.interp2d(x,y,np.transpose(z))
        return f
    

def fill_lifetimes(df):
    """ Use the ApproxLifetime class to fill the lifetime column in a 
        dataframe.
    """
    lt = ApproxLifetime()

    df.loc[:,'lifetime'] = [lt.lifetime(row['mass'],row['Z']) 
                                    for _,row in df.iterrows()] 

    return df
