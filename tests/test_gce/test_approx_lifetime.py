from itertools import product

import numpy as np

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
