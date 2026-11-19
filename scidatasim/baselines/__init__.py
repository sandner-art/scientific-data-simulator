"""Standard detection baselines, with their false-alarm probabilities, for the science packs.

* :mod:`.periodogram` - generalised Lomb-Scargle periodogram (floating mean, weights) and the
  Baluev (2008) false-alarm probability of its highest peak
* :mod:`.bls` - box least squares for transits, with an empirical false-alarm probability
* :mod:`.significance` - p-values and Z, trials factors and the look-elsewhere effect (Sidak,
  Gross-Vitells), empirical p-values from null simulations
* :mod:`.counting` - binned Poisson likelihood fits, a signal scan (the bump hunt), its global
  significance, Asimov significance

All are plain numpy/scipy; where astropy has the same method, the tests check the two agree.
A method is measured on planted truth before it is trusted on real data: run it on the
simulator's null and injected series and compare its stated false-alarm rate with the observed one.
"""
from .bls import bls, bls_null
from .counting import BinnedFit, asimov_z, global_significance
from .periodogram import lomb_scargle, lomb_scargle_fap, frequency_grid
from .significance import (empirical_p, gross_vitells, p_to_z, sidak, trials_factor, z_to_p)

__all__ = ["BinnedFit", "asimov_z", "global_significance", "bls", "bls_null", "lomb_scargle", "lomb_scargle_fap", "frequency_grid", "empirical_p",
           "gross_vitells", "p_to_z", "sidak", "trials_factor", "z_to_p"]
