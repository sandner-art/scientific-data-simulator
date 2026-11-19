"""Fetch the pre-registered Gaia DR4 sample (10.5281/zenodo.24370371) from the Gaia archive.

Runs the registered selection (percentiles and controls per G bin, DR3 orbit companions matched
through ``dr3_neighbourhood``, Gaia BH1-3), the auxiliary DR4 rows (``gaia_source``,
``nss_two_body_orbit``, ``spurious_signal`` via TAP_UPLOAD) and the epoch astrometry (DataLink,
``EPOCH_ASTROMETRY``) in batches, into ``--out``. Every query, row count and file (SHA-256) goes into
``manifest.json``. Interrupted? Run it again: finished steps are skipped.

Needs ``pip install astroquery`` (with astropy). From 2 December 2026, 12:00 CET::

    python tools/gaia_dr4_fetch.py --record preregistration_gaia_dr4.json --probe     # 2 sources, check the format
    python tools/gaia_dr4_fetch.py --record preregistration_gaia_dr4.json --out gaia_dr4_data
    python tools/gaia_dr4_fetch.py --record preregistration_gaia_dr4.json --dry-run   # print the queries

Log in (recommended: job space, no row limits) with ``--login`` (asks for user and password) or
``--credentials FILE`` (user on the first line, password on the second).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None):
    from scidatasim.experiments.astrometry import dr4
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", required=True, help="the deposited pre-registration JSON")
    parser.add_argument("--out", default="gaia_dr4_data")
    parser.add_argument("--batch", type=int, default=1000, help="sources per DataLink call (at most 5000)")
    parser.add_argument("--data-structure", default="RAW", help="DataLink data structure (check on release day)")
    parser.add_argument("--format", default="ecsv")
    parser.add_argument("--login", action="store_true")
    parser.add_argument("--credentials")
    parser.add_argument("--dry-run", action="store_true", help="print the queries and stop")
    parser.add_argument("--probe", action="store_true", help="fetch two named sources' epochs and read them")
    args = parser.parse_args(argv)
    record = dr4.load_record(args.record)
    if args.dry_run:
        for g_lo, g_hi in record["selection"]["controls"]["g_bins"]:
            print(f"-- percentiles G {g_lo}-{g_hi}\n{dr4.percentile_query(record, g_lo, g_hi)}\n")
        print("-- controls: the record's ADQL with the percentiles filled in\n")
        print(f"-- DR3 companions\n{dr4.dr3_companion_query()}\n")
        print(f"-- named\n{dr4.named_query(list(dr4.KNOWN_BLACK_HOLES.values()))}\n")
        for name, q in dr4.auxiliary_queries().items():
            print(f"-- {name}\n{q}\n")
        return 0
    from astroquery.gaia import Gaia
    if args.credentials:
        Gaia.login(credentials_file=args.credentials)
    elif args.login:
        Gaia.login()
    if args.probe:
        from scidatasim.experiments.astrometry.epochs import read_epoch_astrometry, source_epochs, summary
        probe = os.path.join(args.out, "probe")
        os.makedirs(probe, exist_ok=True)
        named = dr4._table(Gaia.launch_job_async(dr4.named_query(list(dr4.KNOWN_BLACK_HOLES.values())[:2])))
        ids = [int(i) for i in dr4.best_match(named)["source_id"]]
        path = os.path.join(probe, "probe.zip")
        result = Gaia.load_data(ids=ids, data_release="Gaia DR4", retrieval_type="EPOCH_ASTROMETRY",
                                data_structure=args.data_structure, format=args.format, output_file=path)
        if not os.path.exists(path):
            dr4._write_tables(result, path)
        frame = read_epoch_astrometry(path)
        print("columns:", sorted(frame.columns))
        print("summary:", summary(frame))
        for sid in ids:
            ep, w, info = source_epochs(frame, sid)
            print(sid, info)
        return 0
    fetch = dr4.Fetch(Gaia, record, args.out, batch=args.batch, data_structure=args.data_structure, fmt=args.format)
    manifest = fetch.run()
    print(f"done: {len(manifest['files'])} files in {args.out}; manifest.json lists them with SHA-256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
