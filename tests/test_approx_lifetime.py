from itertools import product

import numpy as np
import pytest

from simple_gce.gce import approx_lifetime

lt = approx_lifetime.ApproxLifetime()
data = lt._read_portinari_lifetimes()
mass_list = data.M.to_list()
z_list = [0.0004, 0.004, 0.008, 0.02, 0.05]


def test_recover_portinari_lifetime():
    """Test that the interpolated values match the original data"""

    for m, z in product(mass_list, z_list):
        interp_val = float(lt.lifetime(mass=m, z=z))
        actual_val = float(data[data.M == m][str(z)])

        assert interp_val == actual_val


def test_recover_portinari_mass():
    """Test that the interpolated values match the original data"""

    for z in z_list:
        for i, row in data.iterrows():
            l = float(row[str(z)])
            interp_val = float(lt.mass(lifetime=l, z=z))
            actual_val = float(row["M"])

            assert np.round(interp_val, 1) == actual_val


TEST_INPUTS = [
    (0.0, 0.0),
    (1.0, -1.0),
    (0.0, -1.0),
]
TEST_IDS = ["invalidmass", "invalidz", "invalidboth"]


@pytest.mark.parametrize("m,z", TEST_INPUTS, ids=TEST_IDS)
def test_valuerror_lifetime(m, z):
    """Test the passing unphysical values into interpolation raises
    the expected error.
    """
    err_msg = f"Trying to interpolate with unphysical values; " f"mass = {m} and Z = {z}"

    with pytest.raises(ValueError, match=err_msg):
        _ = lt.lifetime(mass=m, z=z)


TEST_IDS = ["invalidlifetime", "invalidz", "invalidboth"]


@pytest.mark.parametrize("lifetime,z", TEST_INPUTS, ids=TEST_IDS)
def test_valuerror_mass(lifetime, z):
    """Test the passing unphysical values into interpolation raises
    the expected error.
    """
    err_msg = f"Trying to interpolate with unphysical values; " f"lifetime = {lifetime} and Z = {z}"

    with pytest.raises(ValueError, match=err_msg):
        _ = lt.mass(lifetime=lifetime, z=z)
