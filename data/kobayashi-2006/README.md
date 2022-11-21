Data for stellar chemical yields, from Kobayashi, C. et al 2006 ApJ 653 1145.
Downloaded from the machine readable tables found here; https://iopscience.iop.org/article/10.1086/508914
Reformatted for compatibility using Python;
```
import pandas as pd

df = pd.read_csv(
    "raw/table1.csv",
    sep="\s+",
    skiprows=24,
    header=None,
    index_col=0,
)
z_vals = df.index.unique()
mass_row = {
    1: ["mass"],
    2: [13.0],
    3: [15.0],
    4: [18.0],
    5: [20.0],
    6: [25.0],
    7: [30.0],
    8: [40.0],
}

models = []
for z in z_vals:
    model_grp = df.loc[z]
    new_row = pd.DataFrame(mass_row, index=[z])
    model_grp = pd.concat([model_grp, new_row])
    model_grp = model_grp.set_index(1, drop=True).T
    model_grp.columns.name = None
    model_grp["Z"] = z
    model_grp["type"] = "cc"
    models.append(model_grp)

models = pd.concat(models)
models = models.rename(
    columns={"M_final_": "mass_final", "M_cut_": "remnant_mass", "p": "^1^H", "d": "^2^H"}
)
models.to_csv("kobayashi-2006_table1.csv", index=False)
```

Descriptions:
  * kobayashi-2006_sn.csv is the data from Table 1 of Kobayashi 2006.
  * kobayashi-2006_hn.csv is the data from Table 2 of Kobayashi 2006.
  * kobayashi-2006_ia.csv is the data extracted from Table 3 of Kobayashi 2006.
  * kobayashi-2006_imfweighted.csv is data the extracted from Table 3 of Kobayashi 2006.

The chemical yield data taken from Tables 2 and 3 are presented as fraction of the total initial stellar mass, and the Ia data from Table 3 is given in solar masses.
This data has been rescaled and stored in each of the files with the `_scaled.csv` suffix, containing the same yield data presented as mass fractions in the ejecta (i.e., the mass fractions sum to unity)
