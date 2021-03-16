"""Galaxy class object."""

import numpy as np

from ..io import load_stellar_models
from ..utils import chem_elements
from . import generate_imf, approx_agb
from .. import config

el2z = chem_elements.el2z 

INFALL_TIMESCALE = config.GALAXY_PARAMS['infall_timescale']
SFR_TIMESCALE = config.GALAXY_PARAMS['sfr_timescale']
TOTAL_MASS = config.GALAXY_PARAMS['total_mass']

class Galaxy(object):
    def __init__(self):
        """
        """
        # Load the stellar model yields (CC, AGB, etc.)
        self.stellar_models = load_stellar_models.read_stellar_csv()
        self.ia_model = load_stellar_models.read_ia_csv()

        # Initialise variables (from config where appropriate).
        self.time = 0.0
        self.z = 0.0
        self.gas_mass = 0.0
        self.star_mass = 0.0
        self.sfr = 0.0#config.GALAXY_PARAMS['sfr_init']
        self.infall_rate = self.calc_infall_rate(self.time)
        # Include only the elements listed in the dataset.
        self.elements = list(set(self.stellar_models.columns).intersection(set(el2z.keys())))
        # The elemental mass fractions in the gas to evolve.
        x = np.zeros(len(self.elements))
        # Store the index for each element in the array.
        self.x_idx = {el : int(i) for i,el in enumerate(self.elements)}
        x[self.x_idx['H']] = float(config.GALAXY_PARAMS['h_init'])
        x[self.x_idx['He']] = float(config.GALAXY_PARAMS['he_init'])
        self.x = x
        self.infall_x = np.array(x)
        # Store the evolution of metallicity and SFR in memory. The historical
        # values of these variables are needed in the equations of evolution,
        # other data will be output to file on disk. 
        self.historical_z = np.array([[self.time, self.z]])
        self.historical_sfr = np.array([[self.time, self.sfr]])
        self.imf = generate_imf.IMF(stellar_models = self.stellar_models, slope = config.IMF_PARAMS['slope'])


    def evolve(self, dt):
        """
        """

        elements = self.elements
        time = self.time

        stellar_models = self.stellar_models

        imf = self.imf
        historical_z = self.historical_z
        historical_sfr = self.historical_sfr
        gas_mass = self.gas_mass
        star_mass = self.star_mass
        z = self.z
        x_idx = self.x_idx
        x = self.x
        sfr = self.sfr
        infall_rate = self.infall_rate
        infall_x = self.infall_x

        # fill the composition of AGB/wind models if they have approximated.
        # this will recycle the composition of the ISM from the time of formation.
        if stellar_models[(stellar_models.type.isin(['agb','wind'])) & 
                        (stellar_models.Z <= z)][elements].isna().any(axis = None):
            stellar_models = approx_agb.fill_composition(stellar_models, z, x, x_idx)
            self.stellar_models = stellar_models

        stellar_models = stellar_models[stellar_models.lifetime <= time].copy()

        if stellar_models.empty:
            ej_cc = 0.0
            ej_cc_x = (self.stellar_models[elements] * 0.0).sum()
            ej_wind = 0.0
            ej_wind_x = (self.stellar_models[elements] * 0.0).sum()
            #ej_ia = 0.0
            #ej_ia_x = (self.stellar_models[elements] * 0.0).sum()
        else:
            # time and metallicity at the point of formation for each stellar model
            stellar_models.loc[:,'t_birth'] = time - np.array(stellar_models.lifetime)
            Z_birth = self._get_historical_value(historical_z,stellar_models.t_birth) 
            # extract the models with Z == Z_birth for each stellar mass
            models = self._filter_stellar_models(stellar_models,Z_birth)
            if not models.mass.is_unique:
                raise error_handling.ProgramError("Unable to filter stellar models")

            cc_models = models[(models.type == 'cc')].copy()
            wind_models = models[models.type.isin(['agb','wind'])].copy()

            cc_imfdm = imf.imfdm(mass_list = cc_models.mass)
            wind_imfdm = imf.imfdm(mass_list = wind_models.mass)

            cc_sfr_birth = self._get_historical_value(historical_sfr,cc_models.t_birth) 
            wind_sfr_birth = self._get_historical_value(historical_sfr,wind_models.t_birth) 
            # ejecta mass from massive stars (core collapse/winds) for each mass range
            ej_cc_ = (cc_models.mass_final - cc_models.remnant_mass) * cc_sfr_birth * cc_imfdm
            # total ejecta mass from massive stars (core collapse/winds)
            ej_cc = ej_cc_.sum()
            # ejecta mass (per element) from massive stars (core collapse/winds)
            ej_cc_x = cc_models[elements].mul(ej_cc_, axis = 'rows').sum()
            # ejecta mass from winds for each mass range 
            ej_wind_ = (wind_models.mass - wind_models.mass_final) * wind_sfr_birth * wind_imfdm
            # total ejecta mass from winds
            ej_wind = ej_wind_.sum()
            # ejecta mass (per element) from winds
            ej_wind_x = wind_models[elements].mul(ej_wind_, axis = 'rows').sum()

            # Ia
            #rate_ia = ...
            #ej_ia = ia_models....

        ej_x = np.array(ej_cc_x + ej_wind_x)
        ej = ej_cc + ej_wind
        
        dm_s_dt = sfr - ej
        dm_g_dt = infall_rate - sfr + ej
        dm_mx_dt = (infall_x * infall_rate) - (x * sfr) + ej_x 

        self.star_mass = star_mass + dm_s_dt * dt
        self.gas_mass = gas_mass + dm_g_dt * dt
        mx = x * gas_mass + dm_mx_dt * dt
        x = mx / self.gas_mass
        self.x = x
        if abs(np.sum(x) - 1.0) > 1.e-8:
            error_handling.ProgramError("Error in evolution. SUM(X) != 1.0")
        self.z = np.sum(x) - x[x_idx['H']] - x[x_idx['He']]
        self.time = time + dt

        self.update_sfr()
        self.update_infall_rate()
        self.update_historical_sfr()
        self.update_historical_z()


    def update_infall_rate(self):
        """
        """

        time = self.time
        infall_rate = self.calc_infall_rate(time)
        self.infall_rate = infall_rate

    @staticmethod
    def calc_infall_rate(time):
        """
        """

        infall_rate = 1.0 / INFALL_TIMESCALE * np.exp(-time / INFALL_TIMESCALE) * TOTAL_MASS

        return infall_rate

    def update_sfr(self):
        """
        """
        gas_mass = self.gas_mass
        sfr = self.calc_sfr(gas_mass)
        self.sfr = sfr

    @staticmethod
    def calc_sfr(gas_mass):
        """
        """

        sfr = 1.0 / SFR_TIMESCALE * gas_mass

        return sfr
        
    def update_historical_sfr(self):#,t_min):
        """
        """
        historical_sfr = self.historical_sfr
        t = self.time
        sfr = self.sfr
        #historical_sfr = historical_sfr[historical_sfr[:,0] >= t_min]
        historical_sfr = np.append(historical_sfr, [[t,sfr]], axis = 0)
        self.historical_sfr = historical_sfr

    def update_historical_z(self):#,t_min):
        """
        """
        historical_z = self.historical_z
        t = self.time
        z = self.z
        #historical_z = historical_z[historical_z[:,0] >= t_min]
        historical_z = np.append(historical_z, [[t,z]], axis = 0)
        self.historical_z = historical_z

    @staticmethod
    def _get_historical_value(historical_array,time_array):
        """
        Search array of historical values for values at specified time(s).

        Args:
            historical_array: 2D array of [time,value] evolution. Must be
                orderder by time (ascending).
            time_array: 1D array of time values for extraction from 
                historical datapoints. 
        Returns:
            values: Values extracted from the historical array at the specified
                times. Ordering matches the time_array provided.
        """

        values = np.array([historical_array[historical_array[:,0] <= t][-1][-1] for t in time_array])
            
        return values
    
    def _filter_stellar_models(self,stellar_models,z_birth):
        """
        Extract model with Z == Z_birth for each stellar mass.
        """

        time = self.time
        z = self.z
        stellar_models.loc[:,'Z_diff'] = abs(stellar_models.Z - z_birth)
        stellar_models = stellar_models[(stellar_models.lifetime <= time) & (stellar_models.Z <= z)]
        idx = stellar_models.groupby('mass').Z_diff.idxmin().to_list()
        stellar_models = stellar_models[stellar_models.index.isin(idx)] 
        stellar_models.drop(columns = ['Z_diff'], inplace = True)

        return stellar_models
