import copy
from dataclasses import dataclass
from typing import List, Tuple
import warnings

import numpy as np
from scipy import interpolate

from .. import config
from ..io import load_stellar_models
from ..io.load_stellar_models import StellarModels
from ..utils import error_handling
from ..utils.chem_elements import get_abundance_of_element
from . import approx_winds, ia_utils, imf

INFALL_TIMESCALE = config.GALAXY_PARAMS["infall_timescale"]
SFR_TIMESCALE = config.GALAXY_PARAMS["sfr_timescale"]
TOTAL_MASS = config.GALAXY_PARAMS["total_mass"]


class Galaxy(object):
    """The main Galaxy object.

    This class contains all of the information needed to describe the Galaxy, and methods
    for evolving the Galaxy through time. No arguments are required to create a Galaxy
    object; all configurable parameters will be loaded from config.py

    Attributes
    ----------
    stellar_models: StellarModels
        A dataclass holding the properties of the stellar models existing in the Galaxy.
    include_hn: bool
        Whether or not hypernovae are included in the model.
    hn_frac: float
        The fraction of massive stars that explode as hypernovae (between 0.0 and 1.0).
    time: float
        The current time in the evolution, i.e., the age of the Galaxy in years.
    z: float
        The current metallicity of the Galaxy (sum of metal mass fractions).
    gas_mass: float
        The current mass of gas in the Galaxy in solar masses.
    star_mass: float
        The current total mass of stars in the Galaxy in solar masses.
    galaxy_mass: float
        The current total mass of the Galaxy in solar masses.
    sfr: float
        The current star formation rate in the Galaxy in solar masses per year.
    infall_rate: float
        The current rate of mass falling into the Galaxy in solar masses per year.
    x: List[float]
        The chemical composition of the gas in the Galaxy (mass fraction per element).
    x_idx: Dict[str, int]
        The index of each element in self.x
    non_metal_idx: dict[Tuple, int]
        Dictionary mapping non-metal elements to their index in self.x
    infall_x: List[float]
        The chemical compsition of gas falling into the Galaxy.
    historical_z: (N, 2) np.ndarray
        The [time, metallicity] at each of N historical timesteps.
    historical_sfr: (N, 2) np.ndarray
        The [time, sfr] at each of N historical timesteps.
    imf: IMF
        An object describing the initial mass function of stars forming in the Galaxy.
    ia_system: IaSystem
        A dataclass holding the properties of Type Ia systems exisiting in the Galaxy.
    f_sfr: Callable
        An interpolation over self.historical_sfr to quickly get SFR for a given time.
    f_z: Callable
        An interpolation over self.historical_z to quickly get Z for a given time.
    """

    def __init__(self):

        # Load the stellar model yields (CC, AGB, etc.)
        self.stellar_models = load_stellar_models.generate_stellarmodels_dataclass()
        # Initialise the Ia model parameters
        self.ia_system = load_stellar_models.generate_iasystem_dataclass()

        if not np.all(self.stellar_models.elements == self.ia_system.elements):
            missing_el_stellar = list(set(self.ia_system.elements) - set(self.stellar_models.elements))
            missing_el_ia = list(set(self.stellar_models.elements) - set(self.ia_system.elements))
            warnings.warn(
                "Elements included in Ia ejecta must be the same as "
                "those in the stellar model yields (CCSNe/winds)."
            )
            self.stellar_models = load_stellar_models.generate_stellarmodels_dataclass(additional_elements=missing_el_stellar)
            self.ia_system = load_stellar_models.generate_iasystem_dataclass(additional_elements=missing_el_ia)

        # Initialise variables (from config where appropriate).
        self.include_hn = bool(config.STELLAR_MODELS["include_hn"])
        self.time = 0.0
        self.z = config.GALAXY_PARAMS["z_init"]
        self.gas_mass = 0.0
        self.star_mass = 0.0
        self.galaxy_mass = 0.0
        self.sfr = config.GALAXY_PARAMS["sfr_init"]
        self.infall_rate = self.calc_infall_rate(self.time)
        self.hn_frac_init = config.GALAXY_PARAMS["hn_frac_init"]
        self.hn_frac = config.GALAXY_PARAMS["hn_frac_init"]
        # The elemental mass fractions in the gas to evolve.
        x = np.zeros(len(self.stellar_models.elements))
        self.x_idx = self.stellar_models.x_idx
        try:
            x[self.x_idx[("H", None)]] = float(config.GALAXY_PARAMS["h_init"])
        except KeyError:
            x[self.x_idx[("H", 1)]] = float(config.GALAXY_PARAMS["h_init"])
        try:
            x[self.x_idx[("He", None)]] = float(config.GALAXY_PARAMS["he_init"])
        except KeyError:
            x[self.x_idx[("He", 4)]] = float(config.GALAXY_PARAMS["he_init"])
        self.non_metal_idx = self.stellar_models.non_metal_idx
        self.x = x
        self.infall_x = np.array(x)
        # Store the evolution of metallicity and SFR in memory. The historical
        # values of these variables are needed in the equations of evolution,
        # other data will be output to file on disk.
        self.historical_z = np.array([[self.time, self.z]])
        self.historical_sfr = np.array([[self.time, self.sfr]])
        self.imf = imf.IMF(
            masses=self.stellar_models.mass_dim,
            slope=config.IMF_PARAMS["slope"],
            mass_min=config.IMF_PARAMS["mass_min"],
            mass_max=config.IMF_PARAMS["mass_max"],
        )

        self.f_sfr = lambda x: np.ones_like(x) * config.GALAXY_PARAMS["sfr_init"]
        self.f_z = lambda x: np.ones_like(x) * config.GALAXY_PARAMS["z_init"]

    def evolve(self, dt: float):
        """Evolve the Galaxy through a single timestep.

        Calculates the rates of star formation, mass ejection from stars, and gas inflow,
        and integrates the evolution of the Galaxy through a single timestep.

        Parameters
        ----------
        dt: float
            The size of the timestep in years.
        """

        time = float(self.time)
        # Use a copy to avoid risk of permanently modifying stellar model properties
        stellar_models = copy.deepcopy(self.stellar_models)

        imf = self.imf
        gas_mass = float(self.gas_mass)
        star_mass = float(self.star_mass)
        z = float(self.z)
        x_idx = self.x_idx
        x = self.x
        sfr = float(self.sfr)
        infall_rate = float(self.infall_rate)
        infall_x = self.infall_x

        if self.include_hn is False:
            (
                self.stellar_models.x_wind,
                stellar_models.x_wind,
            ) = approx_winds.recycle_ism_composition(
                stellar_models.x_wind, stellar_models.z_dim, z, x, x_idx, self.include_hn
            )
        else:
            (
                self.stellar_models.x_wind,
                stellar_models.x_wind,
                self.stellar_models.x_hn_wind,
                stellar_models.x_hn_wind,
            ) = approx_winds.recycle_ism_composition(
                stellar_models.x_wind,
                stellar_models.z_dim,
                z,
                x,
                x_idx,
                self.include_hn,
                stellar_models.x_hn_wind,
            )

        if np.all(stellar_models.lifetime[:, stellar_models.z_dim <= z] > time):
            ejecta = self.EnrichmentSources(
                ej_cc=0.0,
                ej_x_cc=np.zeros_like(x),
                ej_wind=0.0,
                ej_x_wind=np.zeros_like(x),
                ej_ia=0.0,
                ej_x_ia=np.zeros_like(x),
            )
        else:
            # Time and metallicity at the point of formation for each stellar model
            t_birth = time - stellar_models.lifetime
            z_birth = self._get_historical_value("z", t_birth)

            z_birth, t_birth, stellar_models = self._interpolate_between_model_metallicity(
                z_birth, t_birth, stellar_models
            )

            imfdm = imf.imfdm(mass_list=stellar_models.mass_dim)
            sfr_birth = self._get_historical_value("sfr", t_birth)

            ejecta = self._calculate_enrichment_sources(sfr_birth, imfdm, stellar_models)

        ej_x = np.array(ejecta.ej_x_cc + ejecta.ej_x_wind + ejecta.ej_x_ia)
        ej = ejecta.ej_cc + ejecta.ej_wind + ejecta.ej_ia

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
        self.z = np.sum([xi for i, xi in enumerate(x) if i not in self.non_metal_idx.values()])

        self.time = time + dt

        self._conservation_checks()
        self.update_sfr()
        self.update_infall_rate()
        self.update_hn_rate()
        self.update_historical_sfr()
        self.update_historical_z()

    def update_infall_rate(self):
        time = self.time
        infall_rate = self.calc_infall_rate(time)
        self.infall_rate = infall_rate

    @staticmethod
    def calc_infall_rate(time: float) -> float:
        # infall_rate = 1.0 / INFALL_TIMESCALE * np.exp(-time / INFALL_TIMESCALE) * TOTAL_MASS

        infall_rate = (
            1.0 * (time / (INFALL_TIMESCALE**2)) * np.exp(-time / INFALL_TIMESCALE) * TOTAL_MASS
        )

        return infall_rate

    def update_hn_rate(self):
        hn_frac = self.calc_hn_rate(self.hn_frac_init, self.z)
        self.hn_frac = hn_frac

    @staticmethod
    def calc_hn_rate(hn_frac_init: float, z_gas: float) -> float:
        return max(0.001, hn_frac_init * np.exp(-z_gas / 0.001))

    def update_sfr(self):
        gas_mass = self.gas_mass
        sfr = self.calc_sfr(gas_mass)
        self.sfr = sfr

    @staticmethod
    def calc_sfr(gas_mass: float) -> float:
        sfr = (1.0 / SFR_TIMESCALE) * gas_mass

        return sfr

    def update_historical_sfr(self):
        historical_sfr = self.historical_sfr
        t = self.time
        sfr = self.sfr
        historical_sfr = np.append(historical_sfr, [[t, sfr]], axis=0)
        self.historical_sfr = historical_sfr
        self.f_sfr = interpolate.interp1d(
            historical_sfr[:, 0],
            historical_sfr[:, 1],
            kind="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )

    def update_historical_z(self):
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

    def _get_historical_value(self, property: str, time_array: np.ndarray) -> np.ndarray:
        """
        Search array of historical values for values at specified time(s).

        Parameters
        ----------
        property: str
            The property of the Galaxy to find historical values for. Must be one of
            "sfr" or "z".
        time_array: np.ndarray
            An array of time values to extract historical values for.

        Returns
        -------
        values: np.ndarray
            Historical values of the property for each of the times specified.
        """

        valid_properties = {
            "sfr": self.f_sfr,
            "z": self.f_z,
        }
        if property not in valid_properties.keys():
            return ValueError(
                f"Invalid property ({property}) to extract historical "
                f"values for. Must be one of {valid_properties.keys()}"
            )

        values = valid_properties[property](time_array)

        return values

    @staticmethod
    def _filter_stellar_models(
        z_birth: np.ndarray, z_dim: List[float]
    ) -> Tuple[List[bool], np.ndarray]:
        """Create a mask and weighting matrix to filter stellar models by metallicity.

        For an array of metallicities at birth for each stellar model, create a mask and
        weighting matrix used to interpolate between models of discrete metallicity.

        Parameters
        ----------
        z_birth: (M, Z) np.ndarray
            The metallicity at birth for each model of mass M and metallicity Z, where
            the time of birth is the current time minus stellar lifetime.
        z_dim: List[float]
            The metallicity of each model along the second axis of the z_birth array.

        Returns
        -------
        mask: List[Bool]
            A boolean mask along the mass axis of the stellar models, to select models
            with a lifetime less than the current time in the evolution.
        weight: np.ndarray
            The weight to apply to each model, to interpolate along the metallicity axis.
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

    def _interpolate_between_model_metallicity(
        self, z_birth: np.ndarray, t_birth: np.ndarray, stellar_models: StellarModels
    ) -> Tuple[List[float], List[float], StellarModels]:
        """Reduce the stellar models to a single model per mass range.

        For each mass in the mass axis; remove the row if the mass-dependent lifetime is
        longer than the current time in evolution, otherwise, interpolate along the
        metallicity axis to the metallicity at the time of birth.

        Parameters
        ----------
        z_birth: (M, Z) np.ndarray
            The metallicity of the Galaxy at the time of birth (based on the
            mass-metallicity dependent lifetime) for each model.
        t_birth: (M, Z) np.ndarray
            The age of the Galaxy at the time of birth (based on the mass-metallicity
            dependent lifetime) for each model.
        stellar_models: StellarModels
            A dataclass containing information that describes the stellar models.

        Returns
        -------
        z_birth: List[float]
            The metallicity of the Galaxy at the time of birth (based on the
            mass-metallicity dependent lifetime) for each model. Reduced to
            a single model per mass.
        t_birth: List[float]
            The age of the Galaxy at the time of birth (based on the mass-metallicity
            dependent lifetime) for each model. Reduced to a single model per mass.
        stellar_models: StellarModels
            A dataclass containing the properties of the stellar models, reduced to only
            include those which will be ejecting matter at the current timestep.
        """
        # Rescale ejecta mass fractions to be fraction of total initial stellar mass,
        # to simplify interpolation
        stellar_models.x_cc = (stellar_models.w_cc.T * stellar_models.x_cc.T).T
        stellar_models.x_wind = (stellar_models.w_wind.T * stellar_models.x_wind.T).T
        if self.include_hn:
            stellar_models.x_hn = (stellar_models.w_hn.T * stellar_models.x_hn.T).T
            stellar_models.x_hn_wind = (stellar_models.w_hn_wind.T * stellar_models.x_hn_wind.T).T
        # Create a mask for the models with Z ~= Z_birth for each stellar mass
        mask, weight = self._filter_stellar_models(z_birth, stellar_models.z_dim)
        # Filter the model arrays to include only those shedding mass
        stellar_models.mass_dim = stellar_models.mass_dim[mask]
        stellar_models.w_cc = (stellar_models.w_cc.T * weight.T).T.sum(axis=1)[mask]
        stellar_models.w_wind = (stellar_models.w_wind.T * weight.T).T.sum(axis=1)[mask]

        stellar_models.x_cc = (stellar_models.x_cc.T * weight.T).T.sum(axis=1)[mask]
        stellar_models.x_wind = (stellar_models.x_wind.T * weight.T).T.sum(axis=1)[mask]
        t_birth = (t_birth.T * weight.T).T.sum(axis=1)[mask]
        z_birth = (z_birth.T * weight.T).T.sum(axis=1)[mask]

        if self.include_hn:
            stellar_models.w_hn = (stellar_models.w_hn.T * weight.T).T.sum(axis=1)[mask]
            stellar_models.w_hn_wind = (stellar_models.w_hn_wind.T * weight.T).T.sum(axis=1)[mask]
            stellar_models.x_hn = (stellar_models.x_hn.T * weight.T).T.sum(axis=1)[mask]
            stellar_models.x_hn_wind = (stellar_models.x_hn_wind.T * weight.T).T.sum(axis=1)[mask]

        return z_birth, t_birth, stellar_models

    def _calculate_enrichment_sources(
        self, sfr_birth: List[float], imfdm: List[float], stellar_models: StellarModels
    ) -> dataclass:
        hn_frac = self.hn_frac
        if not self.include_hn:
            # total mass of stars of mass `m` born from the gas
            mass_m = sfr_birth * imfdm
            # mass ejected from stars (core collapse/winds) for each mass range
            ej_cc_ = stellar_models.w_cc * mass_m
            # total ejecta mass from massive stars (core collapse/winds)
            ej_cc = ej_cc_.sum()
            # ejecta mass (per element) from massive stars (core collapse/winds)
            ej_x_cc = np.matmul(mass_m, stellar_models.x_cc)
            # ejecta mass from winds for each mass range
            ej_wind_ = stellar_models.w_wind * mass_m
            # total ejecta mass from winds
            ej_wind = ej_wind_.sum()
            # ejecta mass (per element) from winds
            ej_x_wind = np.matmul(mass_m, stellar_models.x_wind)
        else:
            hn_mask = stellar_models.mass_dim >= stellar_models.min_mass_hn
            # TODO: Can the HN values be separated earlier?
            ej_hn_ = stellar_models.w_hn[hn_mask] * sfr_birth[hn_mask] * imfdm[hn_mask]
            ej_hn_wind_ = stellar_models.w_hn_wind[hn_mask] * sfr_birth[hn_mask] * imfdm[hn_mask]
            stellar_models.x_hn = stellar_models.x_hn[hn_mask]
            stellar_models.x_hn_wind = stellar_models.x_hn_wind[hn_mask]
            stellar_models.w_hn = stellar_models.w_hn[hn_mask]
            stellar_models.w_hn_wind = stellar_models.w_hn_wind[hn_mask]

            # total mass of stars of mass `m` born from the gas
            mass_m = sfr_birth * imfdm
            # ejecta mass from stars (core collacse/winds) for each mass range
            ej_cc_1_ = stellar_models.w_cc[~hn_mask] * mass_m[~hn_mask]
            ej_cc_2_ = stellar_models.w_cc[hn_mask] * mass_m[hn_mask]
            # total ejecta mass from massive stars (core collapse/winds)
            ej_cc = ej_cc_1_.sum() + (1 - hn_frac) * ej_cc_2_.sum() + hn_frac * ej_hn_.sum()
            # ejecta mass (per element) from massive stars (core collapse/winds)
            ej_x_cc = (
                np.matmul(mass_m[~hn_mask], stellar_models.x_cc[~hn_mask])
                + (1 - hn_frac) * np.matmul(mass_m[hn_mask], stellar_models.x_cc[hn_mask])
                + hn_frac * np.matmul(mass_m[hn_mask], stellar_models.x_hn)
            )
            # ejecta mass from winds for each mass range
            ej_wind_1_ = stellar_models.w_wind[~hn_mask] * mass_m[~hn_mask]
            ej_wind_2_ = stellar_models.w_wind[hn_mask] * mass_m[hn_mask]
            # total ejecta mass from winds
            ej_wind = (
                ej_wind_1_.sum() + (1 - hn_frac) * ej_wind_2_.sum() + hn_frac * ej_hn_wind_.sum()
            )
            # ejecta mass (per element) from winds
            ej_x_wind = (
                np.matmul(mass_m[~hn_mask], stellar_models.x_wind[~hn_mask])
                + (1 - hn_frac) * np.matmul(mass_m[hn_mask], stellar_models.x_wind[hn_mask])
                + hn_frac * np.matmul(mass_m[hn_mask], stellar_models.x_hn_wind)
            )

        # Ia
        # TODO: use fe from avg. stellar value rather than gas, then use fe_h >= -1.1
        fe_abu = get_abundance_of_element(x=self.x, x_idx=self.x_idx, element="Fe")
        h_abu = get_abundance_of_element(x=self.x, x_idx=self.x_idx, element="H")
        fe_h = np.log10(fe_abu / h_abu) - -2.7519036043868
        if fe_h >= -1.0:
            rate_ia = ia_utils.calc_ia_rate_fast(self)
        else:
            rate_ia = 0.0

        ej_ia = self.ia_system.mass_co * rate_ia
        ej_x_ia = ej_ia * self.ia_system.x_ia

        ejecta = self.EnrichmentSources(
            ej_x_cc=ej_x_cc,
            ej_x_wind=ej_x_wind,
            ej_x_ia=ej_x_ia,
            ej_cc=ej_cc,
            ej_wind=ej_wind,
            ej_ia=ej_ia,
        )

        return ejecta

    @dataclass
    class EnrichmentSources:
        """
        An object containing the source components of chemical enrichment.

        Attributes
        ----------
        ej_x_cc: np.array
            The mass of each element ejected from core-collapse SNe (including HNe if applicable).
        ej_x_wind: np.array
            The mass of each element ejected via stellar winds from massive stars and AGB.
        ej_x_ia: np.array
            The mass of each element ejected via stellar winds from massive stars and AGB.
        ej_cc: float
            The total mass ejected from core-collapse SNe (including HNe if applicable).
        ej_wind: float
            The total mass ejected via stellar winds from massive stars and AGB.
        ej_ia: float
            The total mass ejected via stellar winds from massive stars and AGB.
        """

        ej_x_cc: np.array
        ej_x_wind: np.array
        ej_x_ia: np.array
        ej_cc: float
        ej_wind: float
        ej_ia: float

    def _conservation_checks(self):
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
