"""Real data as a model's outputs: :class:`~.logic.ExternalSeriesExperiment` reads a measured series
(any format :mod:`scidatasim.ingest` reads) and serves it with role ``observation``, so real and
simulated data go through the same estimators, questions and reports."""
