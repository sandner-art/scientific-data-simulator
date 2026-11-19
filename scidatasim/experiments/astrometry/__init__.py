"""Gaia-like epoch astrometry: single stars and astrometric binaries, a detection baseline, and
reading and writing the Gaia DR4 ``epoch_astrometry`` layout.

* :mod:`.model` - the along-scan measurement equation, Keplerian orbits (Thiele-Innes), sampling
* :mod:`.fit` - five-parameter fit, circular-orbit periodogram, Keplerian fit, excess noise
* :mod:`.epochs` - DR4 epoch astrometry tables (read; write simulated sources in the same layout)
* :mod:`.logic` - :class:`AstrometricBinaryExperiment`, the experiment run by the engine
* :mod:`.dr4`, :mod:`.dr4_analysis` - the Gaia DR4 selection, fetch and analysis pre-registered in
  10.5281/zenodo.24370371
"""
