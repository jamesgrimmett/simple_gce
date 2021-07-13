"""Galaxy class object."""

import numpy as np
from scipy import interpolate

from ..io import load_stellar_models
from ..utils import chem_elements, error_handling
from . import imf, approx_agb, approx_lifetime
from .. import config

el2z = chem_elements.el2z 
lt = approx_lifetime.ApproxLifetime()

INFALL_TIMESCALE = config.GALAXY_PARAMS['infall_timescale']
SFR_TIMESCALE = config.GALAXY_PARAMS['sfr_timescale']
TOTAL_MASS = config.GALAXY_PARAMS['total_mass']

class Galaxy(object):
    def __init__(self):
        """
        """
        # Load the stellar model yields (CC, AGB, etc.)
        self.stellar_models = load_stellar_models.read_stellar_csv()
        # Include only the elements listed in the dataset.
        self.elements = list(set(self.stellar_models.columns).intersection(set(el2z.keys())))
        # Sort by charge number
        self.elements.sort(key = lambda x : el2z[x])
        arrays = load_stellar_models.fill_arrays(self.stellar_models)
        self.mass_dim = arrays['mass_dim']
        self.z_dim = arrays['z_dim']
        self.x_idx = arrays['x_idx']
        self.lifetime = arrays['lifetime']
        self.mass_final = arrays['mass_final']
        self.mass_remnant = arrays['mass_remnant']
        self.x_cc = arrays['x_cc']
        self.x_wind = arrays['x_wind'] 

        # Initialise variables (from config where appropriate).
        self.time = 0.0
        self.z = 0.0
        self.gas_mass = 0.0
        self.star_mass = 0.0
        self.sfr = 0.0#config.GALAXY_PARAMS['sfr_init']
        self.infall_rate = self.calc_infall_rate(self.time)
        # The elemental mass fractions in the gas to evolve.
        x = np.zeros(len(self.elements))
        x[self.x_idx['H']] = float(config.GALAXY_PARAMS['h_init'])
        x[self.x_idx['He']] = float(config.GALAXY_PARAMS['he_init'])
        self.x = x
        self.infall_x = np.array(x)
        # Store the evolution of metallicity and SFR in memory. The historical
        # values of these variables are needed in the equations of evolution,
        # other data will be output to file on disk. 
        self.historical_z = np.array([[self.time, self.z]])
        self.historical_sfr = np.array([[self.time, self.sfr]])
        self.imf = imf.IMF(masses = self.mass_dim, 
                                    slope = config.IMF_PARAMS['slope'],
                                    mass_min = config.IMF_PARAMS['mass_min'],
                                    mass_max = config.IMF_PARAMS['mass_max'],
                                    mass_min_cc = config.STELLAR_MODELS['mass_min_cc']
                                    )
        # Initialise the Ia model parameters
        self.ia_model = load_stellar_models.read_ia_csv()
        if not set(self.ia_model.columns).issubset(set(self.elements)):
            raise error_handling.NotImplementedError(
                "Composition of Ia ejecta must\
                     be a subset of the composition of stellar yields (CCSNe/winds)")
        x_ia = np.zeros_like(self.elements)
        for el, idx in self.x_idx.items():
            if el in self.ia_model.columns:
                x_ia[idx] = float(self.ia_model[el]) 
        self.x_ia = x_ia
        self.mass_co = float(config.IA_PARAMS['mass_co'])
        self.imf_donor_slope = float(config.IA_PARAMS['imf_donor_slope'])
        self.mdl_rg = float(config.IA_PARAMS['mdl_rg'])
        self.mdu_rg = float(config.IA_PARAMS['mdu_rg'])
        self.mdl_ms = float(config.IA_PARAMS['mdl_ms'])
        self.mdu_ms = float(config.IA_PARAMS['mdu_ms'])
        self.mpl = float(config.IA_PARAMS['mpl'])
        self.mpu = float(config.IA_PARAMS['mpu'])
        self.b_rg = float(config.IA_PARAMS['b_rg'])
        self.b_ms = float(config.IA_PARAMS['b_ms'])

        #self.masses_rg = np.arange(self.mdl_rg, self.mdu_rg, 0.1)
        #self.imf_ia_rg = imf.IMF(masses = self.masses_rg, 
        #                            slope = self.imf_donor_slope,
        #                            mass_min = self.mdl_rg,
        #                            mass_max = self.mdu_rg,
        #                            )
        #self.masses_ms = np.arange(self.mdl_ms, self.mdu_ms, 0.1), 
        #self.imf_ia_ms = imf.IMF(masses = self.masses_ms,
        #                            slope = self.imf_donor_slope,
        #                            mass_min = self.mdl_ms,
        #                            mass_max = self.mdu_ms,
        #                            )


    def evolve(self, dt):
        """
        """

        elements = self.elements
        time = self.time

        mass_dim = self.mass_dim 
        z_dim = self.z_dim 
        lifetime = self.lifetime 
        mass_final = self.mass_final 
        mass_remnant = self.mass_remnant
        x_cc = self.x_cc
        x_wind = self.x_wind
        mass_co = self.mass_co

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

        # fill the composition of AGB/wind models if they have been approximated.
        # this will recycle the composition of the ISM from the time of formation.
        if np.isnan(x_wind[:,z_dim <= z]).any():
            x_wind[:,z_dim <= z] = approx_agb.fill_composition(x_wind[:,z_dim <= z], z, x, x_idx)
            self.x_wind = x_wind

        # Trim arrays 
        lifetime = lifetime[:,z_dim <= z]
        mass_final = mass_final[:,z_dim <= z]
        mass_remnant = mass_remnant[:,z_dim <= z]
        x_cc = x_cc[:,z_dim <= z,:]
        x_wind = x_wind[:,z_dim <= z,:]
        z_dim = z_dim[z_dim <= z]

        if not (lifetime <= time).any():
            ej_cc = 0.0
            ej_x_cc = np.zeros_like(x)
            ej_wind = 0.0
            ej_x_wind = np.zeros_like(x)
            ej_ia = 0.0
            ej_ia_x = np.zeros_like(x)
        else:
            # Time and metallicity at the point of formation for each stellar model
            t_birth = time - lifetime
            z_birth = self._get_historical_value(historical_z, t_birth) 
            # Create a mask for the models with Z == Z_birth for each stellar mass
            mask = self._filter_stellar_models(z_birth, z_dim)
            # Ensure only one model (metallicity) per mass is kept
            if mask.sum(axis = 1).max() != 1:
                raise error_handling.ProgramError("Unable to filter stellar models")
            # Filter the model arrays to include only those shedding mass
            x_cc = x_cc[mask]
            x_wind = x_wind[mask]
            mass_final = mass_final[mask]
            mass_remnant = mass_remnant[mask]
            mass_dim = mass_dim[mask.sum(axis = 1).astype(bool)]
            lifetime = lifetime[mask]
            t_birth = t_birth[mask]
            z_birth = z_birth[mask]
            if (lifetime > time).any():
                #arrays need more trimming
                xxx

            imfdm = imf.imfdm(mass_list = mass_dim)

            sfr_birth = self._get_historical_value(historical_sfr,t_birth) 
            
            # ejecta mass from stars (core collapse/winds) for each mass range
            ej_cc_ = (mass_final - mass_remnant) / mass_dim * sfr_birth * imfdm
            # total ejecta mass from massive stars (core collapse/winds)
            ej_cc = ej_cc_.sum()
            # ejecta mass (per element) from massive stars (core collapse/winds)
            ej_x_cc = np.matmul(ej_cc_, x_cc)
            # ejecta mass from winds for each mass range 
            ej_wind_ = (mass_dim - mass_final) / mass_dim * sfr_birth * imfdm
            # total ejecta mass from winds
            ej_wind = ej_wind_.sum()
            # ejecta mass (per element) from winds
            ej_x_wind = np.matmul(ej_wind_, x_wind)
            # Ia
            #rate_ia = ...
            #ej_ia = ia_models....

        ej_x = np.array(ej_x_cc + ej_x_wind)
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
            raise error_handling.ProgramError("Error in evolution. SUM(X) != 1.0")
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
    
    def calc_ia_rate(self):
        """
        """

        # approximate the turnoff mass using the current metallicity
        m_turnoff = lt.mass(lifetime = self.time, z = self.z) 
        # WD - RG scenario
        mdl_rg = max(self.mdl_rg, m_turnoff)
        mdu_rg = self.mdu_rg
        mpl = max(self.mpl, m_turnoff)
        mpu = self.mpu
        b_rg = self.b_rg
        m_ms = self.b_ms
        if (mdl_rg >= mdu_rg) or (mpl >= mpu):
            rate_rg = 0.0
        else:
            rate_rg_ = b_rg * self.imf.integrate_ia(lower = mpl, upper = mpu)
            
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
            time_array: 2D array of time values for extraction from 
                historical datapoints. 
        Returns:
            values: Values extracted from the historical array at the specified
                times. Ordering matches the time_array provided.
        """

        f = interpolate.interp1d(historical_array[:,0], historical_array[:,1], 
                                        bounds_error = False, fill_value = np.nan)
        values = f(time_array)
            
        return values
    
    @staticmethod
    def _filter_stellar_models(z_birth,z_dim):
        """
        Extract model with Z == Z_birth for each stellar mass.
        """

        #time = self.time
        #z = self.z
        #stellar_models.loc[:,'Z_diff'] = abs(stellar_models.Z - z_birth)
        #stellar_models = stellar_models[(stellar_models.lifetime <= time) & (stellar_models.Z <= z)]
        #idx = stellar_models.groupby('mass').Z_diff.idxmin().to_list()
        #stellar_models = stellar_models[stellar_models.index.isin(idx)] 
        #stellar_models.drop(columns = ['Z_diff'], inplace = True)
        
        #z_diff = np.array([abs(row - z_dim) for row in z_birth])
        #mask = np.array([row == row.min() for row in z_diff])

        mask = np.empty_like(z_birth)
        for i,row in enumerate(z_birth):
            z_diff = abs(row - z_dim)
            mask[i] = z_diff == z_diff.min()

        return mask.astype(bool)
