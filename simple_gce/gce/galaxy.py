"""Galaxy class object."""

import numpy as np
from scipy import interpolate
from scipy.integrate import fixed_quad as f_quad_int
from typing import Sequence, Dict, Type, Callable

from .. import config
from ..io import load_stellar_models
from ..utils import chem_elements, error_handling
from . import approx_agb, approx_lifetime, imf

el2z = chem_elements.el2z
lt = approx_lifetime.ApproxLifetime()

INFALL_TIMESCALE = config.GALAXY_PARAMS["infall_timescale"]
SFR_TIMESCALE = config.GALAXY_PARAMS["sfr_timescale"]
TOTAL_MASS = config.GALAXY_PARAMS["total_mass"]


class Galaxy(object):
    """A Galaxy object.

    Attributes
    ----------
    include_hn: bool
        True if HNe are to be included as a source of chemical enrichment. If False,
        only SNe will be included as a chemical end product for massive stars.
    stellar_models: pd.DataFrame
        The properties of stellar models and their end products to included in the evolution.
    elements: Sequence[str]
        The names of all elements included in the evolution.
    mass_dim: Sequence[np.number]
        The initial mass of each stellar model. Corresponds to the first axis of
        each ndarray that contains properties of stellar models in mass-metallicity space.
    z_dim: Sequence[np.number]
        The metallicity of each stellar models. Corresponds to the second axis of
        each ndarray that contains properties of stellar models in mass-metallicity space.
    x_idx: Dict[str, int]
        Each element and the corresponding index to use to access its abundance value.
    lifetime: np.ndarray
        The lifetime (years) of each stellar models in mass-metallicty space.
    w_cc: np.ndarray
        The total mass fraction of ejecta material from CCSNe in mass-metallicity space.
        Given as the fraction of the initial stellar mass.
    w_wind: np.ndarray
        The total mass fraction of ejecta material from winds in mass-metallicity space.
        Given as a fraction of the initial stellar mass.
    x_cc: np.ndarray
        The mass fraction of each element in the ejecta of CCSNe in mass-metallicity-element
        space. Given as a fraction of the CCSNe ejecta mass.
    x_wind: np.ndarray
        The mass fraction of each element in the ejecta of winds in mass-metallicity-element
        space. Given as a fraction of the wind ejecta mass.
    w_hn: np.ndarray
        The total mass fraction of ejecta material from HNe in mass-metallicity space.
        Given as the fraction of the initial stellar mass.
    w_hn_wind: np.ndarray
        The total mass fraction of ejecta material from winds in mass-metallicity space.
        Given as a fraction of the initial stellar mass.
    x_hn: np.ndarray
        The mass fraction of each element in the ejecta of HNe in mass-metallicity-element
        space. Given as a fraction of the HNe ejecta mass.
    x_hn_wind: np.ndarray
        The mass fraction of each element in the ejecta of winds in mass-metallicity-element
        space. Given as a fraction of the wind ejecta mass.
    hn_frac: np.number
        The fraction of massive stars destined to explode as HNe (i.e. a weighting factor to
        combine HNe yields with CCSNe yields.) Must be between 0.0 and 1.0
    min_mass_hn: np.number
        The minimum (initial) mass of HNe models. E.g., if 20, then the chemical contribution of
        stars with mass >= 20 will be a weighted average between HNe and SNe.
    time: np.number
        The age of the Galaxy in years.
    z: np.number
        The metallicity of the Galaxy.
    gas_mass: np.number
        The mass of gas within the Galaxy in solar masses
    star_mass: np.number
        The total mass of stars within the Galaxy in solar masses
    galaxy_mass: np.number
        The total mass of the Galaxy (i.e., gas_mass + star_mass).
    sfr: np.number
        The star formation rate in the Galaxy (solar masses / year)
    infall_rate: np.number
        The rate of gas falling into the Galaxy from a halo reservoir (solar masses / year)
    x: Sequence[np.number]
        The mass fractions of elements in the gas. The indices of elements can be found with x_idx.
    infall_x: Sequence[np.number]
        The mass fraction abundances of the material infalling from the halo.
    historical_z: np.ndarray
        The time and metallicity for each previous timestep.
    historical_sfr: np.ndarray
        The time and SFR for each previous timestep.
    imf: Type[IMF]
        An IMF object containing information about the initial mass function of stars in the
        Galaxy. The IMF has been discretised over the stellar masses provided to the model.
    lifetime_min: np.ndarray
        The minimum lifetime for the representative range of each stellar models, in
        mass-metallicity space. The IMF has been discretised over the stellar masses, so this
        is the lifetime of a star at the upper mass range of each mass bin.
    ia_model: pd.DataFrame
        The chemical abundances ejected from the SNe Ia model to be included in the evolution.
    x_ia: Sequence[np.number]
        The chemical abundances ejected from the SNe Ia model to be included in the evolution.
    mass_co: np.number
        The mass of white dwarf (WD) exploding as a SNe Ia.
    imf_donor_slope: np.number
        The slope of the IMF representing donor stars in the SNe Ia system.
    mdl_rg: np.number
        The minimum initial mass of donor stars in red giant - white dwarf SNe Ia systems.
    mdu_rg: np.number
        The maximum initial mass of donor stars in red giant - white dwarf SNe Ia systems.
    mdl_ms: np.number
        The minimum initial mass of donor stars in main sequence - white dwarf SNe Ia systems.
    mdu_ms: np.number
        The maximum intial mass of donor stars in main sequence - white dwarf SNe Ia systems.
    mpl: np.number
        The minimum intiial mass of stars which may become white dwarves in SNe Ia systems.
    mpu: np.number
        The maximum intiial mass of stars which may become white dwarves in SNe Ia systems.
    b_rg: np.number
        The total fraction of red giant stars that end up in a SNe Ia system.
    b_ms: np.number
        The total fraction of main sequence stars that end up in a SNe Ia system.
    masses_rg: Sequence[np.number]
        A discrete list of red giant masses used to contruct an IMF for these stars.
    imf_ia_rg: Type[IMF]
        An IMF object representing the initial mass function of red giant stars within Ia systems.
    masses_ms: Sequence[np.number]
        A discrete list of main sequence masses used to contruct an IMF for these stars.
    imf_ia_ms: Type[IMF]
        An IMF object representing the initial mass function of main sequence stars within Ia
        systems.
    f_sfr: Callable
        A function used to interpolate for the historical SFR at a given time.
    f_z: Callable
        A function used to interpolate for the historical metallicity at a given time.

    Methods
    -------
    evolve(dt)
        Evolves the Galaxy for a timestep of specified length `dt`.
    """

    def __init__(self):

        self.include_hn = bool(config.STELLAR_MODELS["include_hn"])
        # Load the stellar model yields (CC, AGB, etc.)
        stellar_models = load_stellar_models.read_stellar_csv()
        self.stellar_models = stellar_models
        # Include only the elements listed in the dataset.
        self.elements = list(set(stellar_models.columns).intersection(set(el2z.keys())))
        # Sort by charge number
        self.elements.sort(key=lambda x: el2z[x])
        # Load stellar model arrays
        arrays = load_stellar_models.fill_arrays(self.stellar_models)
        self.mass_dim = arrays["mass_dim"]
        self.z_dim = arrays["z_dim"]
        self.x_idx = arrays["x_idx"]
        self.lifetime = arrays["lifetime"]
        self.w_cc = arrays["w_cc"]
        self.w_wind = arrays["w_wind"]
        self.x_cc = arrays["x_cc"]
        self.x_wind = arrays["x_wind"]
        if self.include_hn:
            self.x_hn = arrays["x_hn"]
            self.x_hn_wind = arrays["x_hn_wind"]
            self.w_hn = arrays["w_hn"]
            self.w_hn_wind = arrays["w_hn_wind"]
            self.hn_frac = config.STELLAR_MODELS["hn_frac"]
            self.min_mass_hn = stellar_models[stellar_models.type == "hn"].mass.min()

        # Initialise variables (from config where appropriate).
        self.time = 0.0
        self.z = config.GALAXY_PARAMS["z_init"]
        self.gas_mass = 0.0
        self.star_mass = 0.0
        self.galaxy_mass = 0.0
        self.sfr = config.GALAXY_PARAMS["sfr_init"]
        self.infall_rate = self.calc_infall_rate(self.time)
        # The elemental mass fractions in the gas to evolve.
        x = np.zeros(len(self.elements))
        x[self.x_idx["H"]] = float(config.GALAXY_PARAMS["h_init"])
        x[self.x_idx["He"]] = float(config.GALAXY_PARAMS["he_init"])
        self.x = x
        self.infall_x = np.array(x)
        # Store the evolution of metallicity and SFR in memory. The historical
        # values of these variables are needed in the equations of evolution,
        # other data will be output to file on disk.
        self.historical_z = np.array([[self.time, self.z]])
        self.historical_sfr = np.array([[self.time, self.sfr]])
        self.imf = imf.IMF(
            masses=self.mass_dim,
            slope=config.IMF_PARAMS["slope"],
            mass_min=config.IMF_PARAMS["mass_min"],
            mass_max=config.IMF_PARAMS["mass_max"],
        )
        # Use the minimum lifetime (maximum mass) for each discretised mass range
        self.lifetime_min = np.squeeze(
            [
                [lt.lifetime(self.imf.mass_bins[i, 1], z) for z in self.z_dim]
                for i, m in enumerate(self.mass_dim)
            ]
        )
        # Initialise the Ia model parameters
        self.ia_model = load_stellar_models.read_ia_csv()
        if not set(self.ia_model.columns).issubset(set(self.elements)):
            raise error_handling.NotImplementedError(
                "Composition of Ia ejecta must\
                     be a subset of the composition of stellar yields (CCSNe/winds)"
            )
        x_ia = np.zeros(len(self.elements))
        for el, idx in self.x_idx.items():
            if el in self.ia_model.columns:
                x_ia[idx] = float(self.ia_model[el])
        self.x_ia = x_ia
        self.mass_co = float(config.IA_PARAMS["mass_co"])
        self.imf_donor_slope = float(config.IA_PARAMS["imf_donor_slope"])
        self.mdl_rg = float(config.IA_PARAMS["mdl_rg"])
        self.mdu_rg = float(config.IA_PARAMS["mdu_rg"])
        self.mdl_ms = float(config.IA_PARAMS["mdl_ms"])
        self.mdu_ms = float(config.IA_PARAMS["mdu_ms"])
        self.mpl = float(config.IA_PARAMS["mpl"])
        self.mpu = float(config.IA_PARAMS["mpu"])
        self.b_rg = float(config.IA_PARAMS["b_rg"])
        self.b_ms = float(config.IA_PARAMS["b_ms"])

        self.masses_rg = np.arange(self.mdl_rg + 0.1, self.mdu_rg, 0.1)
        self.imf_ia_rg = imf.IMF(
            masses=self.masses_rg,
            slope=self.imf_donor_slope,
            mass_min=self.mdl_rg,
            mass_max=self.mdu_rg,
        )
        self.masses_ms = np.arange(self.mdl_ms + 0.1, self.mdu_ms, 0.1)
        self.imf_ia_ms = imf.IMF(
            masses=self.masses_ms,
            slope=self.imf_donor_slope,
            mass_min=self.mdl_ms,
            mass_max=self.mdu_ms,
        )

        self.f_sfr = lambda x: np.ones_like(x) * config.GALAXY_PARAMS["sfr_init"]
        self.f_z = lambda x: np.ones_like(x) * config.GALAXY_PARAMS["z_init"]

    def evolve(self, dt):
        """ """

        time = float(self.time)

        # Use copies of mutable objects for each timestep, to avoid modifying
        # the class attribute (e.g., when updating x_wind)
        mass_dim = self.mass_dim.copy()
        z_dim = self.z_dim.copy()
        lifetime_min = self.lifetime_min.copy()
        w_cc = self.w_cc.copy()
        w_wind = self.w_wind.copy()
        x_cc = self.x_cc.copy()
        x_wind = self.x_wind.copy()
        if self.include_hn:
            x_hn = self.x_hn.copy()
            x_hn_wind = self.x_hn_wind.copy()
            w_hn = self.w_hn.copy()
            w_hn_wind = self.w_hn_wind.copy()
            hn_frac = float(self.hn_frac)
        mass_co = float(self.mass_co)
        x_ia = self.x_ia.copy()

        imf = self.imf
        gas_mass = float(self.gas_mass)
        star_mass = float(self.star_mass)
        z = float(self.z)
        x_idx = self.x_idx.copy()
        x = self.x.copy()
        sfr = float(self.sfr)
        infall_rate = float(self.infall_rate)
        infall_x = self.infall_x.copy()

        # fill the composition of AGB/wind models if they are to be approximated.
        # this will recycle the composition of the ISM from the time of formation.
        if np.isnan(x_wind[:, z_dim <= z]).any():
            x_wind[:, z_dim <= z] = approx_agb.fill_composition(x_wind[:, z_dim <= z], z, x, x_idx)
            self.x_wind = np.copy(x_wind)
        x_wind[:, z_dim > z] = approx_agb.fill_composition(x_wind[:, z_dim > z], z, x, x_idx)

        if self.include_hn:
            if np.isnan(x_hn_wind[:, z_dim <= z]).any():
                x_hn_wind[:, z_dim <= z] = approx_agb.fill_composition(
                    x_hn_wind[:, z_dim <= z], z, x, x_idx
                )
                self.x_hn_wind = np.copy(x_hn_wind)
            x_hn_wind[:, z_dim > z] = approx_agb.fill_composition(
                x_hn_wind[:, z_dim > z], z, x, x_idx
            )

        if not (lifetime_min[:, z_dim <= z] <= time).any():
            ej_cc = 0.0
            ej_x_cc = np.zeros_like(x)
            ej_wind = 0.0
            ej_x_wind = np.zeros_like(x)
            ej_ia = 0.0
            ej_x_ia = np.zeros_like(x)
        else:
            # Time and metallicity at the point of formation for each stellar model
            t_birth = time - lifetime_min
            z_birth = self._get_historical_value("z", t_birth)
            # Rescale ejecta mass fractions to be fraction of total initial stellar mass,
            # to simplify interpolation
            x_cc = (w_cc.T * x_cc.T).T
            x_wind = (w_wind.T * x_wind.T).T

            # Create a mask for the models with Z ~= Z_birth for each stellar mass
            mask, weight = self._filter_stellar_models(z_birth, z_dim)
            # Filter the model arrays to include only those shedding mass
            mass_dim = mass_dim[mask]
            w_cc = (w_cc.T * weight.T).T.sum(axis=1)[mask]
            w_wind = (w_wind.T * weight.T).T.sum(axis=1)[mask]

            x_cc = (x_cc.T * weight.T).T.sum(axis=1)[mask]
            x_wind = (x_wind.T * weight.T).T.sum(axis=1)[mask]
            lifetime_min = (lifetime_min.T * weight.T).T.sum(axis=1)[mask]
            t_birth = (t_birth.T * weight.T).T.sum(axis=1)[mask]
            z_birth = (z_birth.T * weight.T).T.sum(axis=1)[mask]

            if self.include_hn:
                x_hn = (w_hn.T * x_hn.T).T
                x_hn_wind = (w_hn_wind.T * x_hn_wind.T).T
                w_hn = (w_hn.T * weight.T).T.sum(axis=1)[mask]
                w_hn_wind = (w_hn_wind.T * weight.T).T.sum(axis=1)[mask]
                x_hn = (x_hn.T * weight.T).T.sum(axis=1)[mask]
                x_hn_wind = (x_hn_wind.T * weight.T).T.sum(axis=1)[mask]
            if (lifetime_min > time).any():
                # TODO: add checks for correctness of stellar models
                raise error_handling.ProgramError(
                    "Error in evolution. Stellar models are incorrect"
                )

            imfdm = imf.imfdm(mass_list=mass_dim)

            sfr_birth = self._get_historical_value("sfr", t_birth)

            if not self.include_hn:
                # total mass of stars of mass `m` born from the gas
                mass_m = sfr_birth * imfdm
                # mass ejected from stars (core collapse/winds) for each mass range
                ej_cc_ = w_cc * mass_m
                # total ejecta mass from massive stars (core collapse/winds)
                ej_cc = ej_cc_.sum()
                # ejecta mass (per element) from massive stars (core collapse/winds)
                ej_x_cc = np.matmul(mass_m, x_cc)
                # ejecta mass from winds for each mass range
                ej_wind_ = w_wind * mass_m
                # total ejecta mass from winds
                ej_wind = ej_wind_.sum()
                # ejecta mass (per element) from winds
                ej_x_wind = np.matmul(mass_m, x_wind)
            else:
                hn_mask = mass_dim >= self.min_mass_hn
                # TODO: Can the HN values be separated earlier?
                ej_hn_ = w_hn[hn_mask] * sfr_birth[hn_mask] * imfdm[hn_mask]
                ej_hn_wind_ = w_hn_wind[hn_mask] * sfr_birth[hn_mask] * imfdm[hn_mask]
                x_hn = x_hn[hn_mask]
                x_hn_wind = x_hn_wind[hn_mask]
                w_hn = w_hn[hn_mask]
                w_hn_wind = w_hn_wind[hn_mask]

                if not all(hn_mask is True):
                    # total mass of stars of mass `m` born from the gas
                    mass_m = sfr_birth * imfdm
                    # ejecta mass from stars (core collacse/winds) for each mass range
                    ej_cc_1_ = w_cc[~hn_mask] * mass_m[~hn_mask]
                    ej_cc_2_ = w_cc[hn_mask] * mass_m[hn_mask]
                    # total ejecta mass from massive stars (core collapse/winds)
                    ej_cc = ej_cc_1_.sum() + (1 - hn_frac) * ej_cc_2_.sum() + hn_frac * ej_hn_.sum()
                    # ejecta mass (per element) from massive stars (core collapse/winds)
                    ej_x_cc = (
                        np.matmul(mass_m[~hn_mask], x_cc[~hn_mask])
                        + (1 - hn_frac) * np.matmul(mass_m[hn_mask], x_cc[hn_mask])
                        + hn_frac * np.matmul(mass_m[hn_mask], x_hn)
                    )
                    # ejecta mass from winds for each mass range
                    ej_wind_1_ = w_wind[~hn_mask] * mass_m[~hn_mask]
                    ej_wind_2_ = w_wind[hn_mask] * mass_m[hn_mask]
                    # total ejecta mass from winds
                    ej_wind = (
                        ej_wind_1_.sum()
                        + (1 - hn_frac) * ej_wind_2_.sum()
                        + hn_frac * ej_hn_wind_.sum()
                    )
                    # ejecta mass (per element) from winds
                    ej_x_wind = (
                        np.matmul(mass_m[~hn_mask], x_wind[~hn_mask])
                        + (1 - hn_frac) * np.matmul(mass_m[hn_mask], x_wind[hn_mask])
                        + hn_frac * np.matmul(mass_m[hn_mask], x_hn_wind)
                    )
                else:
                    # ejecta mass from stars (core collapse/winds) for each mass range
                    ej_cc_ = w_cc * sfr_birth * imfdm
                    # total ejecta mass from massive stars (core collapse/winds)
                    ej_cc = (1 - hn_frac) * ej_cc_.sum() + hn_frac * ej_hn_.sum()
                    # ejecta mass (per element) from massive stars (core collapse/winds)
                    ej_cc_ = sfr_birth * imfdm
                    ej_hn_ = sfr_birth[hn_mask] * imfdm[hn_mask]
                    ej_x_cc = (1 - hn_frac) * np.matmul(ej_cc_, x_cc) + hn_frac * np.matmul(
                        ej_hn_, x_hn
                    )
                    # ejecta mass from winds for each mass range
                    ej_wind_ = w_wind * sfr_birth * imfdm
                    # total ejecta mass from winds
                    ej_wind = (1 - hn_frac) * ej_wind_.sum() + hn_frac * ej_hn_wind_.sum()
                    # ejecta mass (per element) from winds
                    ej_wind_ = sfr_birth * imfdm
                    ej_hn_wind_ = sfr_birth[hn_mask] * imfdm[hn_mask]
                    ej_x_wind = np.matmul(ej_wind_, x_wind) + np.matmul(ej_hn_wind_, x_hn_wind)

            # Ia
            # TODO: use fe from avg. stellar value rather than gas, then use fe_h >= -1.1
            fe_h = np.log10(x[x_idx["Fe"]] / x[x_idx["H"]]) - -2.7519036043868
            if fe_h >= -1.0:
                rate_ia = self.calc_ia_rate_fast()
            else:
                rate_ia = 0.0

            ej_ia = mass_co * rate_ia
            ej_x_ia = ej_ia * x_ia

        ej_x = np.array(ej_x_cc + ej_x_wind + ej_x_ia)
        ej = ej_cc + ej_wind + ej_ia

        dm_s_dt = sfr - ej
        dm_g_dt = infall_rate - sfr + ej
        dm_mx_dt = (infall_x * infall_rate) - (x * sfr) + ej_x

        self.galaxy_mass += infall_rate * dt
        self.star_mass = star_mass + dm_s_dt * dt
        self.gas_mass = gas_mass + dm_g_dt * dt
        mx = x * gas_mass + dm_mx_dt * dt
        if self.gas_mass > 0:
            x = mx / self.gas_mass
        else:
            x = self.x

        if any(np.isnan(x)):
            raise RuntimeError("Error in evolution. NaNs in the chemical abundances in the gas.")
        self.x = x
        self.z = (
            np.sum(x)
            - x[x_idx["H"]]
            - x[x_idx["He"]]
            - x[x_idx["Li"]]
            - x[x_idx["Be"]]
            - x[x_idx["B"]]
        )

        self.time = time + dt

        self._conservation_checks()
        self.update_sfr()
        self.update_infall_rate()
        self.update_historical_sfr()
        self.update_historical_z()

    def update_infall_rate(self):
        """ """

        time = self.time
        infall_rate = self.calc_infall_rate(time)
        self.infall_rate = infall_rate

    @staticmethod
    def calc_infall_rate(time):
        """ """

        # infall_rate = 1.0 / INFALL_TIMESCALE * np.exp(-time / INFALL_TIMESCALE) * TOTAL_MASS

        infall_rate = (
            1.0 * (time / (INFALL_TIMESCALE ** 2)) * np.exp(-time / INFALL_TIMESCALE) * TOTAL_MASS
        )

        return infall_rate

    def update_sfr(self):
        """ """
        gas_mass = self.gas_mass
        sfr = self.calc_sfr(gas_mass)
        self.sfr = sfr

    @staticmethod
    def calc_sfr(gas_mass):
        """ """

        sfr = (1.0 / SFR_TIMESCALE) * gas_mass

        return sfr

    def calc_ia_rate_fast(self):
        """
        ...
        Accuracy is exchanged for speed in this function (fixed_quad and lt.mass_approx).
        Should make calc_ia_rate_slow() and compare results every n timesteps
        """

        # approximate the turnoff mass
        m_turnoff = lt.mass_approx(lifetime=self.time)
        # Progenitor (WD) component
        mpl = max(self.mpl, m_turnoff)
        mpu = self.mpu
        if mpl >= mpu:
            rate_ia = 0.0
            return rate_ia

        progenitor_component = self.imf._integrate_mod(lower=mpl, upper=mpu)

        # WD - RG scenario
        mdl_rg = max(self.mdl_rg, m_turnoff)
        mdu_rg = self.mdu_rg
        b_rg = self.b_rg
        if mdl_rg >= mdu_rg:
            rate_rg = 0.0
        else:
            rate_rg_1 = b_rg * progenitor_component
            rate_rg_2 = f_quad_int(
                lambda x: self._ia_donor_integrand(x, type="rg"),
                a=mdl_rg,
                b=mdu_rg,
                n=len(np.arange(mdl_rg, mdu_rg, 0.01)),
            )[0]
            rate_rg = rate_rg_1 * rate_rg_2

        # WD - MS scenario
        b_ms = self.b_ms
        mdl_ms = max(self.mdl_ms, m_turnoff)
        mdu_ms = self.mdu_ms
        if (mdl_ms >= mdu_ms) or (mpl >= mpu):
            rate_ms = 0.0
        else:
            rate_ms_1 = b_ms * progenitor_component
            rate_ms_2 = f_quad_int(
                lambda x: self._ia_donor_integrand(x, type="ms"),
                a=mdl_ms,
                b=mdu_ms,
                n=len(np.arange(mdl_ms, mdu_ms, 0.01)),
            )[0]
            rate_ms = rate_ms_1 * rate_ms_2

        rate_ia = rate_rg + rate_ms

        if np.isnan(rate_ia):
            raise RuntimeError("Error in calculation of Ia rate.")

        return rate_ia

    def _ia_donor_integrand(self, m, type):
        """ """
        if type == "rg":
            imf_m = self.imf_ia_rg.functional_form(m)
        elif type == "ms":
            imf_m = self.imf_ia_ms.functional_form(m)
        lifetime_m = lt.lifetime(mass=m, z=self.z)
        t = self.time - lifetime_m
        # TODO: sometimes slight difference between integration lower limit and
        #       true turnoff mass. Need to investigate.
        t = t.clip(min=0.0)
        sfr_m = self._get_historical_value("sfr", t)

        integrand = 1.0 / m * sfr_m * imf_m
        if np.isnan(integrand).any():
            raise RuntimeError("Error in calculation of Ia rate.")

        return integrand

    def update_historical_sfr(self):  # ,t_min):
        """ """
        historical_sfr = self.historical_sfr
        t = self.time
        sfr = self.sfr
        # historical_sfr = historical_sfr[historical_sfr[:,0] >= t_min]
        historical_sfr = np.append(historical_sfr, [[t, sfr]], axis=0)
        self.historical_sfr = historical_sfr
        self.f_sfr = interpolate.interp1d(
            historical_sfr[:, 0],
            historical_sfr[:, 1],
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )

    def update_historical_z(self):  # ,t_min):
        """ """
        historical_z = self.historical_z
        t = self.time
        z = self.z
        # historical_z = historical_z[historical_z[:,0] >= t_min]
        historical_z = np.append(historical_z, [[t, z]], axis=0)
        self.historical_z = historical_z
        self.f_z = interpolate.interp1d(
            historical_z[:, 0],
            historical_z[:, 1],
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )

    def _get_historical_value(self, property, time_array):
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

        if property == "sfr":
            f = self.f_sfr
        elif property == "z":
            f = self.f_z

        values = f(time_array)

        return values

    @staticmethod
    def _filter_stellar_models(z_birth, z_dim):
        """
        Choose the best fitting models to represent this timestep, and
        calculating weighting factors to interpolate in between metallicities
        """

        # TODO: consider interpolating in log-space
        mask = np.zeros_like(z_birth)
        weight = np.zeros_like(z_birth)
        for i, row in enumerate(z_birth):
            if np.isnan(row).any():
                continue
            z_diff = row - z_dim
            if 0.0 in np.round(z_diff, 15):
                mask_z = np.round(z_diff, 15) == 0.0
                mask[i] = mask_z
                weight[i] = mask_z.astype(float)

            else:
                best_idx = np.nanargmin(abs(z_diff))
                if row[best_idx] > z_dim.max():
                    mask_z = z_dim == z_dim.max()
                    mask[i] = mask_z
                    weight[i] = mask_z.astype(float)
                    continue
                if row[best_idx] < z_dim.min():
                    mask_z = z_dim == z_dim.min()
                    mask[i] = mask_z
                    weight[i] = mask_z.astype(float)
                    continue
                diff = z_dim - row[best_idx]

                diff_pos = diff[diff >= 0]
                diff_neg = diff[diff < 0]
                min_pos_idx = np.nanargmin(abs(diff_pos))
                min_pos = diff_pos[min_pos_idx]
                min_pos_idx = int(np.argwhere(diff == min_pos))
                mask_upper = diff == min_pos
                min_neg_idx = np.nanargmin(abs(diff_neg))
                min_neg = diff[min_neg_idx]
                min_neg_idx = int(np.argwhere(diff == min_neg))
                if abs(min_neg_idx - min_pos_idx) != 1:
                    raise RuntimeError(
                        "Error in interpolation between stellar models. Unable to select "
                        "appropriate models to interpolate between"
                    )
                mask_lower = diff == min_neg
                mask_z = mask_upper.astype(bool) | mask_lower.astype(bool)

                w = min_pos / np.ptp(z_dim[mask_z])
                weight[i, mask_upper] = 1.0 - w
                weight[i, mask_lower] = w

                mask[i] = mask_z

        if not np.sum(weight, axis=1).all() in [0.0, 1.0]:
            raise ValueError(
                "Error in interpolation between stellar models. Weights sum to neither 0 nor 1."
            )

        mask = mask.sum(axis=1).astype(bool)

        return mask, weight

    def _conservation_checks(self):
        """ """
        # TODO: add check for all(np.sum(x_cc + x_wind, axis = 1) == 1)
        if abs(np.sum(self.x) - 1.0) > 1.0e-8:
            raise error_handling.ProgramError("Error in evolution. SUM(X) != 1.0")
        if self.galaxy_mass == 0.0:
            if self.star_mass != 0.0 or self.gas_mass != 0.0:
                raise error_handling.ProgramError("Error in evolution. Total mass not conserved")
        elif abs(((self.star_mass + self.gas_mass) / self.galaxy_mass) - 1.0) > 1.0e-8:
            raise error_handling.ProgramError("Error in evolution. Total mass not conserved")
        if self.galaxy_mass > TOTAL_MASS:
            raise error_handling.ProgramError(
                "Error in evolution. Galaxy mass exceeds mass available in system"
            )
        if np.isnan([self.galaxy_mass, self.star_mass, self.galaxy_mass]).any():
            raise error_handling.ProgramError("Nan in evolution")
