"""Approximate AGB models."""

def approximate_agb():
    """
    """

    mass_min_agb = 0.8  # solar masses, minimum stellar mass for AGB
    mass_max_wd = 1.4   # solar masses, maximum white dwarf mass
    z_vals = models.Z.unique()
    m_vals = np.array([max(min_agb,1.5*cfg.IMF['mass_min'])] + [i for i in np.arange(1,10)])
    cols = models.columns
    models_agb = pd.DataFrame(np.zeros((len(m_vals)*len(z_vals),len(cols))),columns=cols)

    m_z = itertools.product(m_vals,z_vals)

    for i in models_agb.index:
        m,z = next(m_z)
        models_agb.loc[i,'mass'] = m
        models_agb.loc[i,'Z'] = z
        models_agb.loc[i,'expl_energy'] = 0.0
        models_agb.loc[i,'type'] = str('AGB')
#        if m <= min_agb:
#            models_agb.loc[i,'remnant_mass'] = m
#        else: 
#            models_agb.loc[i,'remnant_mass'] = min(max_wd,0.25 * m)
#   Iben & Tutukov 1984, in Pagel 2009 after eqn 7.10
        if (m <= 0.506):
            models_agb.loc[i,'remnant_mass'] = m
            models_agb.loc[i,'mass_presn'] = m
        elif (m <= 9.5):
            models_agb.loc[i,'remnant_mass'] = 0.45 + 0.11 * m
            models_agb.loc[i,'mass_presn'] = 0.45 + 0.11 * m
        else:
            models_agb.loc[i,'remnant_mass'] = 1.5
            models_agb.loc[i,'mass_presn'] = 1.5

        models_agb.loc[i,'lifetime'] = lt.lifetime_m(m,z)
