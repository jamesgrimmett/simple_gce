"""Galaxy class object."""
import numpy as np

from ..io import load_stellar_models
from ..utils import chem_elements
from .. import config

el2z = chem_elements.el2z 

INFALL_TIMESCALE = config.GALAXY_PARAMS['infall_timescale']
SFR_TIMESCALE = config.GALAXY_PARAMS['sfr_timescale']

class Galaxy(object):
    def __init__(self, test = False):
        """
        """
        # Load the stellar model yields (CC, AGB, etc.)
        self.stellar_models = load_stellar_models.read_csv(test = test)

        # Initialise variables (from config where appropriate).
        self.time = 0.0
        self.Z = 0.0
        self.sfr = config.GALAXY_PARAMS['sfr_init']
        self.infall_rate = self.calc_infall_rate(self.time)
        # Include only the elements listed in the dataset.
        self.elements = list(set(self.stellar_models.columns).intersection(set(el2z.keys())))
        # The elemental mass fractions in the gas to evolve.
        X = np.zeros(len(self.elements))
        # Store the index for each element in the array.
        self.x_idx = {el : int(i) for i,el in enumerate(self.elements)}
        X[self.x_idx['H']] = float(config.GALAXY_PARAMS['h_init'])
        X[self.x_idx['He']] = float(config.GALAXY_PARAMS['he_init'])
        self.X = X
        # Store the evolution of metallicity and SFR in memory. The historical
        # values of these variables are needed in the equations of evolution,
        # other data will be output to file on disk. 
        self.historical_Z = np.array([[self.time, self.Z]])
        self.historical_sfr = np.array([[self.time, self.sfr]])


    def evolve(self, dt):
        """
        """

        time = self.time

        time = 1.e6

        stellar_models = self.stellar_models
        historical_Z = self.historical_Z
        historical_sfr = self.historical_sfr
        Z = self.Z
        sfr = self.sfr
        CC_models = stellar_models[(stellar_models.type == 'CC') & (stellar_models.lifetime <= time) & (stellar_models.Z <= Z)]
        # Will this work with high resolution stellar lifetime and Z?
        CC_models = CC_models[CC_models.Z == CC_models.Z.max()]
        CC_birthtimes = time - np.array(CC_models.lifetime)
        CC_sfr_birth = self._get_historical_value(historical_sfr,CC_birthtimes) 
        CC_Z_birth = self._get_historical_value(historical_Z,CC_birthtimes) 
        E_CC = (CC_models.mass_final - CC_models.remnant_mass)  
        zz





        # Last?
        self.update_sfr()
        self.update_infall_rate()


    def update_infall_rate(self):
        """
        """

        time = self.time
        infall_rate = self.calc_infall_rate(time)
        self.infall_rate = infall_rate

    def calc_infall_rate(self, time):
        """
        """

        infall_rate = 1.0 / INFALL_TIMESCALE * np.exp(-time / INFALL_TIMESCALE)

        return infall_rate

    def update_sfr(self):
        """
        """
        gas_mass = self.gas_mass
        sfr = self.calc_sfr(gas_mass)
        self.sfr = sfr

    def calc_sfr(self,gas_mass):
        """
        """

        sfr = 1.0 / SFR_TIMESCALE * gas_mass

        return sfr
        
    def update_historical_sfr(self,t,sfr):#,t_min):
        """
        """
        historical_sfr = self.historical_sfr
        #historical_sfr = historical_sfr[historical_sfr[:,0] >= t_min]
        historical_sfr = np.append(historical_sfr, [[t,sfr]], axis = 0)
        self.historical_sfr = historical_sfr

    def update_historical_Z(self,t,Z):#,t_min):
        """
        """
        historical_Z = self.historical_Z
        #historical_Z = historical_Z[historical_Z[:,0] >= t_min]
        historical_Z = np.append(historical_Z, [[t,Z]], axis = 0)
        self.historical_Z = historical_Z

    def _get_historical_value(self,historical_array,time_array):
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

            values = np.array([historical_array[historical_aray[:,0] <= t][-1][-1] for t in time_array])
            
            return values