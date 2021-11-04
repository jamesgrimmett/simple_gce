"""The initial mass function."""

import numpy as np

from .. import config
from ..utils import error_handling


class IMF(object):
    """
    Constructs a discretised IMF. For the specified masses, a normalised IMF
    is instantiated, where each mass represents a section `dm` of the
    continuous IMF.
    Assumed to be of the form phi(m) = N * m **-p, where N is a normalisation
    constant.

    Attributes
    ----------
    slope: numeric
        The exponent of the IMF.
    masses: Sequence[numeric]
        The list of masses to set the discretisation
    mass_min: numeric, optional
        The minimum mass existing in the distribution
    mass_max: numeric, optional
        The maximum mass existing in the distribution

    Methods
    -------

    """

    def __init__(self, slope, masses, mass_min=None, mass_max=None):
        if mass_min is None:
            mass_min = config.IMF_PARAMS["mass_min"]
        if mass_max is None:
            mass_max = config.IMF_PARAMS["mass_max"]
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.mass_min_cc = config.STELLAR_MODELS["mass_min_cc"]
        self.slope = abs(slope)  # minus sign is added explicitly throughout the class
        # normalising total mass to 1.0
        self.imf_norm = (1.0 - self.slope) / (
            self.mass_max ** (1.0 - self.slope) - self.mass_min ** (1.0 - self.slope)
        )
        self.masses = masses
        self.mass_bins = self._generate_mass_bins()
        self.dms = self.mass_bins[:, 1] - self.mass_bins[:, 0]

        self._validate_imf()

    def functional_form(self, m):
        """
        Returns the value of the IMF at the specified m
        """
        result = self.imf_norm * m ** (-1.0 * self.slope)

        return result

    def imfdm(self, mass_list):
        """ """
        try:
            iter(mass_list)
        except TypeError:
            mass_list = [mass_list]
        masses = self.masses
        mass_bins = self.mass_bins
        if not set(mass_list).issubset(masses):
            raise error_handling.ProgramError(
                f"Unable to find mass {mass_list} in the discretised IMF"
            )
        # there must be a cleaner way to get these indices
        idx = [int(np.squeeze(np.where(masses == m_))) for m_ in mass_list]
        result = self.integrate(lower=mass_bins[idx, 0], upper=mass_bins[idx, 1])

        return result

    def integrate(self, lower, upper):
        """
        Integrates the IMF between the specified lower and upper values.
        """
        imf_norm = self.imf_norm
        p = self.slope
        result = imf_norm / (1 - p) * (upper ** (1 - p) - lower ** (1 - p))

        return result

    def _integrate_mod(self, lower, upper):
        """
        Integrates over 1/m * phi(m) dm. This is only called for integrating
        over the Ia progenitor function when finding the Ia rate.
        """

        imf_norm = self.imf_norm
        p = self.slope + 1.0
        result = imf_norm / (1 - p) * (upper ** (1 - p) - lower ** (1 - p))

        return result

    def _generate_mass_bins(self):
        """ """
        masses = self.masses
        mass_min = self.mass_min
        mass_max = self.mass_max
        mass_min_cc = self.mass_min_cc

        lo_mass_models = masses[(masses >= mass_min) & (masses < mass_min_cc)]
        lo_mass_bins = [m for i, m in enumerate(lo_mass_models[:-1])]
        if mass_min < np.min(masses):
            lo_mass_bins.insert(0, mass_min)
        if (masses > mass_min_cc).any():
            lo_mass_bins.append(mass_min_cc)
            hi_mass_models = masses[(masses >= mass_min_cc) & (masses <= mass_max)]
            hi_mass_bins = [m for i, m in enumerate(hi_mass_models[:-1])]
            if mass_max > np.max(masses):
                hi_mass_bins.append(mass_max)
            mass_bins = np.concatenate((np.array(lo_mass_bins), np.array(hi_mass_bins)))
        else:
            lo_mass_bins.append(mass_max)
            mass_bins = np.array(lo_mass_bins)

        mass_bins = np.transpose([mass_bins[:-1], mass_bins[1:]])

        return mass_bins

    def _validate_imf(self):
        """ """
        check = 1.0 - np.sum(self.imfdm(mass_list=self.masses))
        if abs(check) >= 1.0e-5:
            raise error_handling.ProgramError(
                "Error in the IMF implementation. Does not sum to unity."
            )
