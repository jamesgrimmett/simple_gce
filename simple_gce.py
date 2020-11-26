"""A simple one-zone GCE model.

TODO: add a description of where the config can be changed, 
and where data will be loaded from.

Typical usage example:

TODO: add a typical usage example, including setup, evolve, and plotting
results.
"""

class SimpleGCE(object):
    """The GCE object.

    Attributes:
        models_cc: A dataframe containing the core-collapse model information
                    and yields.
        models_ia: A dataframe containing the Type Ia model information and
                    yields.
        models_agb: A dataframe containing the AGB model information and yields.
        