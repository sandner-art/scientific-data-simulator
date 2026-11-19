"""A measured series served as a run's outputs.

Configuration::

    experiment_type: scidatasim.experiments.external.logic.ExternalSeriesExperiment
    source: data/tic_261136679_lc.csv       # any format scidatasim.ingest reads
    value: pdcsap_flux                      # column names in the file
    time: time
    error: pdcsap_flux_err                  # optional
    quality: quality                        # optional: rows with quality not in good_quality -> NaN
    good_quality: [0]
    normalize: median                       # optional: 'median' (relative flux) or 'subtract_median'
    name: flux                              # output name (default: the value column)
    units: {time: d, value: relative flux}
    drop_missing: true                      # drop rows that are not usable (default true)

Outputs: the coordinate (``time``), the series (role ``observation``: it was measured), its
``<name>_error`` (auxiliary) when given. The file's path, SHA-256, size and the rows used go into
the run's ground-truth record under ``source.*``; the fidelity level is ``observed``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from scidatasim import ingest
from scidatasim.base import ExperimentLogic
from scidatasim.utils import DataDescriptor, DataType


class ExternalSeriesExperiment(ExperimentLogic):
    synthetic = False
    fidelity_level = "observed"
    fidelity_note = "A measured series read from a file (see source.* in the record); nothing is simulated."

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        for key in ("source", "value"):
            if not config.get(key):
                raise ValueError(f"ExternalSeriesExperiment needs '{key}'")
        self.series = None

    def initialize(self, config):
        c = self.config
        self.series = ingest.read_series(c["source"], value=c["value"], time=c.get("time"), error=c.get("error"),
                                         quality=c.get("quality"), good_quality=c.get("good_quality", (0,)),
                                         normalize=c.get("normalize"), fmt=c.get("format"), key=c.get("key"))
        return {}

    def run_step(self, state, step):
        return state

    def _arrays(self):
        s = self.series
        keep = s.mask if self.config.get("drop_missing", True) else np.ones(len(s), bool)
        order = np.argsort(s.time[keep], kind="stable")
        t = s.time[keep][order]
        v = s.value[keep][order]
        e = None if s.error is None else s.error[keep][order]
        return t, v, e

    def get_results(self):
        if self.series is None:
            self.initialize(self.config)
        t, v, e = self._arrays()
        units = self.config.get("units") or {}
        name = self.config.get("name") or self.config["value"]
        tname = self.config.get("time") or "index"
        out = {
            tname: {"data": t, "descriptor": DataDescriptor(tname, DataType.NDARRAY, shape=t.shape,
                                                             units=units.get("time"), group="time_series",
                                                             role="coordinate")},
            name: {"data": v, "descriptor": DataDescriptor(name, DataType.NDARRAY, shape=v.shape,
                                                            units=units.get("value"), group="time_series",
                                                            plot_type="scatter", x_axis=tname, role="observation")},
        }
        if e is not None:
            out[f"{name}_error"] = {"data": e, "descriptor": DataDescriptor(
                f"{name}_error", DataType.NDARRAY, shape=e.shape, units=units.get("value"), group="time_series",
                x_axis=tname, role="auxiliary")}
        return out

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        if self.series is None:
            self.initialize(self.config)
        prov = self.series.provenance
        params.update({f"source.{k}": prov[k] for k in ("path", "sha256", "bytes", "format", "rows")})
        params["source.rows_used"] = int(self._arrays()[0].size)
        params.update({f"source.{k}": v for k, v in self.series.meta.items() if v is not None
                       and isinstance(v, (int, float, str))})
        return params
