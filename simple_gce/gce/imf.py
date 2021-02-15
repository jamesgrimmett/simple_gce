"""The initial mass function."""

from .. import config

class IMF(object):
    """
    """
    def __init__(self, stellar_models, slope):
        self.mass_min = config.IMF['mass_min']
        self.mass_max = config.IMF['mass_max']
        self.mass_min_cc = config.STELLAR_MODELS['mass_min_cc']
        self.slope = slope
        # normalising total mass to 1.0
        self.imf_norm = (self.slope - 1.0) / (self.mass_min**(1.0 - self.slope) - self.mass_max**(1.0 - self.slope))
        self.stellar_models = stellar_models
        self._check_compatible()
        self.mass = stellar_models.mass.unique()
        self.mass_bin = self._generate_mass_bins()
        self.dm = self.mass_bin[:,1] - self.mass_bin[:,0]
    
    def integrate(self, lower, upper):
        """
        """
        imf_norm = self.imf_norm
        result =  (imf_norm / 0.35) * (lower**-0.35 - upper**-0.35)

        return result

    def _generate_mass_bins(self):
        """
        """
        mass = self.mass
        mass_min = self.mass_min 
        mass_max = self.mass_max 
        mass_min_cc = self.mass_min_cc 

        lo_mass_models = mass[(mass >= mass_min) & (mass < mass_min_cc)]
        hi_mass_models = mass[(mass >= mass_min_cc) & (mass <= mass_max)]
        lo_mass_bins = [(lo_mass_models[i+1] + m)/2 for i,m in enumerate(lo_mass_models[:-1])]        
        lo_mass_bins.insert(0,mass_min)        
        lo_mass_bins.append(mass_min_cc)        
        hi_mass_bins = [(hi_mass_models[i+1] + m)/2 for i,m in enumerate(hi_mass_models[:-1])]        
        hi_mass_bins.append(mass_max)
        mass_bins = np.concatenate((np.array(lo_mass_bins), np.array(hi_mass_bins)))        
        mass_bins = np.transpose([mass_bins[:-1],mass_bins[1:]])

        return mass_bins
    
    def _check_compatible(self):
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