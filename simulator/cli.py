# simulator/cli.py
"""The ``sds`` command (also ``python -m simulator``).

    sds doctor                       check the installation
    sds examples list [--field F] [--level L] [--json]
    sds examples show NAME           title, summary, parameters, outputs, command
    sds examples run NAME [--quick] [--set name=value ...] [--output DIR] [--timeout S]
    sds examples check [--run-quick] [--only NAME ...]   validate manifests (and run quick modes)
    sds run CONFIG.yaml [--output DIR]   run one experiment configuration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional


def _ensure_models_importable() -> None:
    """In a source checkout, make the bundled models importable even if an older install did not
    package them (the console script does not put the current folder on sys.path)."""
    import importlib.util
    import os
    if importlib.util.find_spec("experiments") is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isdir(os.path.join(root, "experiments")):
            sys.path.insert(0, root)


def _examples(args):
    from .examples import discover
    return discover(args.examples_dir)


def _cmd_list(args) -> int:
    found = _examples(args)
    rows = [e for e in found.values()
            if (not args.field or args.field in e.fields) and (not args.level or e.level == args.level)]
    if args.json:
        print(json.dumps([e.to_dict() for e in rows], indent=2, default=str))
        return 0
    width = max((len(e.name) for e in rows), default=10)
    for e in rows:
        runtime = e.runtime.get("quick") or e.runtime.get("full") or ""
        print(f"{e.name:<{width}}  {e.level:<8} {','.join(e.fields):<22} {runtime:>8}  {e.title}")
    print(f"\n{len(rows)} example(s). Details: sds examples show NAME")
    return 0


def _cmd_show(args) -> int:
    from .examples import build_command
    found = _examples(args)
    if args.name not in found:
        print(f"unknown example '{args.name}'. Available: {', '.join(found)}", file=sys.stderr)
        return 2
    e = found[args.name]
    print(f"{e.title}\n{'=' * len(e.title)}\n{e.summary}\n")
    print(f"level: {e.level}   fields: {', '.join(e.fields) or '-'}   runtime: "
          + ", ".join(f"{k} {v}" for k, v in e.runtime.items()))
    if e.extras:
        print(f"needs extras: {', '.join(e.extras)}  (pip install -e \".[{','.join(e.extras)}]\")")
    if e.params:
        print("\nparameters (sds examples run NAME --set name=value):")
        for p in e.params:
            extra = f" choices {p.choices}" if p.choices else (
                f" range [{p.min}, {p.max}]" if p.min is not None or p.max is not None else "")
            print(f"  {p.name:<16} {p.type:<6} default {p.default!r}{extra}  {p.help}")
    if e.outputs:
        print("\nshows: " + ", ".join(o["path"] for o in e.outputs))
    print("\ncommand: " + " ".join(build_command(e, {}, False, "<output>", "python")))
    if e.quick_args:
        print("quick:   + " + " ".join(e.quick_args))
    return 0


def _parse_sets(pairs: List[str]) -> dict:
    values = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects name=value, got '{pair}'")
        k, v = pair.split("=", 1)
        values[k.strip()] = v
    return values


def _cmd_run(args) -> int:
    from .examples import run_example
    found = _examples(args)
    if args.name not in found:
        print(f"unknown example '{args.name}'. Available: {', '.join(found)}", file=sys.stderr)
        return 2
    example = found[args.name]
    try:
        values = {k: example.param(k).coerce(v) for k, v in _parse_sets(args.set).items()}
    except (KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    run = run_example(example, values, quick=args.quick, output_dir=args.output, timeout=args.timeout,
                      on_line=None if args.silent else _safe_print)
    status = "done" if run.ok else ("timed out" if run.returncode is None else f"failed ({run.returncode})")
    print(f"\n[sds] {example.name}: {status} in {run.seconds:.1f} s. Output: {run.output_dir}")
    for o in run.outputs[:10]:
        print(f"[sds]   {o['kind']}: {o['path']}")
    print(f"[sds] repeat with (in {run.cwd}, with PYTHONPATH={os.path.dirname(os.path.dirname(example.path))}):")
    print(f"[sds]   {run.command_line}")
    return 0 if run.ok else 1


def _safe_print(line: str) -> None:
    """print() that survives characters the console encoding cannot show."""
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _cmd_check(args) -> int:
    from .examples import check_all, discover, run_example
    problems = check_all(args.examples_dir)
    for p in problems:
        print(f"[FAIL] {p}")
    found = discover(args.examples_dir)
    print(f"{len(found)} valid manifest(s), {len(problems)} problem(s).")
    failed = len(problems)
    if args.run_quick:
        import tempfile
        names = args.only or list(found)
        unknown = [n for n in names if n not in found]
        if unknown:
            print(f"unknown example(s): {', '.join(unknown)}. Available: {', '.join(found)}", file=sys.stderr)
            return 2
        for name in names:
            with tempfile.TemporaryDirectory() as tmp:
                run = run_example(found[name], quick=True, output_dir=tmp, timeout=args.timeout)
                tail = ""
                if not run.ok:
                    with open(run.log_path, encoding="utf-8") as f:
                        tail = "".join(f.readlines()[-15:])
                    failed += 1
                print(f"[{' ok ' if run.ok else 'FAIL'}] {name} ({run.seconds:.1f} s)" + (f"\n{tail}" if tail else ""))
    return 1 if failed else 0


def _cmd_run_config(args) -> int:
    from .config import load_config
    from .engine import SimulatorEngine
    run = SimulatorEngine(args.output).run_config(load_config(args.config))
    print(f"[sds] experiment {run.experiment_id}: {run.experiment_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sds", description="Scientific Data Simulator")
    parser.add_argument("--examples-dir", default=None, help="examples folder (default: the checkout's)")
    sub = parser.add_subparsers(dest="command", required=True)

    doc = sub.add_parser("doctor", help="check the installation")
    doc.add_argument("--json", action="store_true")
    doc.add_argument("--plots", action="store_true", help="also try a static plot export")
    doc.add_argument("--output_dir", default="experiments_output", help="folder to check for writing")

    ex = sub.add_parser("examples", help="list, show, run and check examples")
    exs = ex.add_subparsers(dest="action", required=True)
    p = exs.add_parser("list")
    p.add_argument("--field")
    p.add_argument("--level", choices=["student", "research"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_list)
    p = exs.add_parser("show")
    p.add_argument("name")
    p.set_defaults(func=_cmd_show)
    p = exs.add_parser("run")
    p.add_argument("name")
    p.add_argument("--quick", action="store_true", help="the example's fast settings")
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="override a parameter (repeatable)")
    p.add_argument("--output", help="output folder (default: sds_runs/NAME/TIMESTAMP)")
    p.add_argument("--timeout", type=float, default=None, help="stop after this many seconds")
    p.add_argument("--silent", action="store_true", help="do not echo the example's output")
    p.set_defaults(func=_cmd_run)
    p = exs.add_parser("check")
    p.add_argument("--run-quick", action="store_true", help="also run every example in quick mode")
    p.add_argument("--only", nargs="*", help="limit --run-quick to these examples")
    p.add_argument("--timeout", type=float, default=600)
    p.set_defaults(func=_cmd_check)

    run = sub.add_parser("run", help="run one experiment configuration (YAML)")
    run.add_argument("config")
    run.add_argument("--output", default="experiments_output")
    run.set_defaults(func=_cmd_run_config)

    args = parser.parse_args(argv)
    _ensure_models_importable()
    if args.command == "doctor":
        from .doctor import main as doctor_main
        return doctor_main((["--json"] if args.json else []) + (["--plots"] if args.plots else [])
                           + ["--output_dir", args.output_dir])
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
