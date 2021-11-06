__version__ = "0.1.0"

import argparse
import numpy as np
import os
import pandas as pd

from simple_gce.gce import galaxy


def parse_cli_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-o", "--output-path", type=str, default="./", help="Desired path to output csv."
    )

    parser.add_argument(
        "-t", "--tmax", type=float, default=13.0e9, help="Length of evolution time."
    )
    parser.set_defaults(func=run_basic_evolution)

    return parser


def run_basic_evolution(args):
    """Run a basic GCE model."""
    output_file = os.path.join(args.output_path, "gce_output.csv")
    g = galaxy.Galaxy()

    dt = 1.0e6
    tmax = args.tmax
    steps = int(np.ceil(tmax / dt))
    x_idx = g.x_idx
    elements = g.elements

    results = {
        "time": np.zeros(steps),
        "gas_mass": np.zeros(steps),
        "star_mass": np.zeros(steps),
        "sfr": np.zeros(steps),
        "z": np.zeros(steps),
    }
    for el in elements:
        results[el] = np.zeros(steps)

    for i in range(steps):
        g.evolve(dt=1.0e6)
        results["time"][i] = float(g.time)
        results["sfr"][i] = float(g.sfr)
        results["z"][i] = float(g.z)
        results["gas_mass"][i] = float(g.gas_mass)
        results["star_mass"][i] = float(g.star_mass)
        for el in elements:
            results[el][i] = float(g.x[x_idx[el]])
        if g.time % 1.0e8 == 0:
            print(f"time: {g.time*1.e-9}")

    pd.DataFrame(results).to_csv(output_file)
