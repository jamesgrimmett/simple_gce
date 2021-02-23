"""The initial mass function."""

import numpy as np
from .. import config
from ..utils import error_handling

class IMF(object):
    """
    """
    def __init__(self, stellar_models, slope):
        self.mass_min = config.IMF_PARAMS['mass_min']
        self.mass_max = config.IMF_PARAMS['mass_max']
        self.mass_min_cc = config.STELLAR_MODELS['mass_min_cc']
        self.slope = slope
        # normalising total mass to 1.0
        self.imf_norm = (1.0 - self.slope) / (self.mass_max**(1.0 - self.slope) - self.mass_min**(1.0 - self.slope))
        self.stellar_models = stellar_models
        self._check_models_compatible()
        self.masses = stellar_models.mass.unique()
        self.mass_bins = self._generate_mass_bins()
        self.dms = self.mass_bins[:,1] - self.mass_bins[:,0]
        self._test_imf()

    def imfdm(self,m):
        """
        """
        try:
            iter(m)
        except TypeError as err:
            m = [m]
        masses = self.masses
        mass_bins = self.mass_bins
        if not set(m).issubset(masses):
            raise error_handling.ProgramError(f"Unable to find mass {m} in the discretised IMF")
        # there must be a cleaner way to get these indices
        idx = [int(np.squeeze(np.where(masses == m_))) for m_ in m]
        result = self.integrate(lower = mass_bins[idx,0], upper = mass_bins[idx,1])

        return result

    
    def integrate(self, lower, upper):
        """
        """
        imf_norm = self.imf_norm
        p = self.slope
        result = imf_norm / (1 - p) * (upper**(1 - p) - lower**(1 - p))

        return result

    def _generate_mass_bins(self):
        """
        """
        masses = self.masses
        mass_min = self.mass_min 
        mass_max = self.mass_max 
        mass_min_cc = self.mass_min_cc 

        lo_mass_models = masses[(masses >= mass_min) & (masses < mass_min_cc)]
        hi_mass_models = masses[(masses >= mass_min_cc) & (masses <= mass_max)]
        lo_mass_bins = [(lo_mass_models[i+1] + m)/2 for i,m in enumerate(lo_mass_models[:-1])]        
        lo_mass_bins.insert(0,mass_min)        
        lo_mass_bins.append(mass_min_cc)        
        hi_mass_bins = [(hi_mass_models[i+1] + m)/2 for i,m in enumerate(hi_mass_models[:-1])]        
        hi_mass_bins.append(mass_max)
        mass_bins = np.concatenate((np.array(lo_mass_bins), np.array(hi_mass_bins)))        
        mass_bins = np.transpose([mass_bins[:-1],mass_bins[1:]])

        return mass_bins
    
    def _check_models_compatible(self):
        """
        Ensure that the stellar model dataset is compatible with this
        implementation of the IMF. 
        """
        stellar_models = self.stellar_models

        Z_unique = stellar_models.Z.unique()
        mass_unique = stellar_models.mass.unique()

        for m in mass_unique:
            Z_vals = stellar_models[stellar_models.mass == m].Z.unique()
            check = sorted(Z_vals) == sorted(Z_unique)
            if not check:
                raise error_handling.UnknownCaseError("Error building the IMF; a constistent set of masses must be provided for each value of metallicity in the stellar model set.")
            
    def _test_imf(self):
        """
        """
        check = np.sum(self.imfdm(m = self.masses)) - 1.0
        if abs(check) >= 1.e-5:
            raise error_handling.ProgramError("Error in the IMF implementation. Does not sum to unity.")

