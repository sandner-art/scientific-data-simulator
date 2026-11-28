#!/usr/bin/env python3
"""A report on InfiniteScienceGym generation runs, from their output folders.

Point it at one or more outputs of ``audit_isg.py generate`` (a Kaggle version's output, unzipped,
or a local ``--fs-dir``; folders are searched for ``fs/``, ``regeneration_manifest*.json`` and
``logs/``). It prints Markdown with file:line references, so a part can be pointed at::

    python tools/isg_run_report.py C:\\Downloads\\isg-generate-v1 > report.md
    python tools/isg_run_report.py out1 out2 --seed 21          # one repository: manifest and log lines
    python tools/isg_run_report.py out1 --grep "Attempt 1/3"     # every matching line, with file:line

Sections: totals (the pooled manifests), throughput (per stream and for the machine with every
stream busy, and how much of a repository's time the model takes), repositories (status, attempts,
calls, tokens, hours, speed, context), failures by kind, and the vLLM or Ollama servers (loading
time, KV cache, concurrency, the periodic throughput lines, warnings). Standard library and
``audit_isg.py`` next to it only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_isg import merge_manifests  # noqa: E402

CALL = re.compile(r"model call (\d+): (?:prompt (\d+) \+ )?(-?\d+) tokens, (\d+) s")
START = re.compile(r"^\[(\d+)/(\d+)\] seed (\d+)(?: \(attempt (\d+))?")
END = re.compile(r"^\[(\d+)/(\d+)\] seed (\d+) (done|FAILED) in (\d+) s")
STATS = re.compile(r"generation throughput: ([\d.]+) tokens/s.*?Running: (\d+) reqs.*?Waiting: (\d+) reqs"
                   r"(?:.*?KV cache usage: ([\d.]+)%)?")
SERVER_FACTS = ("Loading weights took", "Model loading took", "Available KV cache memory", "GPU KV cache size",
                "Maximum concurrency", "init engine", "torch.compile took", "Graph capturing finished",
                "preempt", "out of memory", "Error", "Traceback")


def find(paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Manifests (the fullest copy of each name) and log files under the given folders."""
    manifests: Dict[str, str] = {}
    logs: List[str] = []

    def size(p):
        try:
            with open(p, encoding="utf-8") as f:
                m = json.load(f)
            return len(m.get("seconds", {})) + len(m.get("failed", {})) + sum(m.get("attempts", {}).values())
        except (OSError, ValueError):
            return -1
    for root in paths:
        for p in glob.glob(os.path.join(root, "**", "regeneration_manifest*.json"), recursive=True):
            name = os.path.basename(p)
            if name not in manifests or size(p) > size(manifests[name]):
                manifests[name] = p
        logs += glob.glob(os.path.join(root, "**", "logs", "*.log"), recursive=True)
    return sorted(manifests.values()), sorted(set(logs))


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r", "\n").split("\n")


def shard_logs(logs: Sequence[str]) -> Dict[str, List[dict]]:
    """Per seed, its attempts as found in shard logs: calls, status, file and line numbers."""
    seeds: Dict[str, List[dict]] = defaultdict(list)
    for path in logs:
        if not os.path.basename(path).startswith("shard-"):
            continue
        current: Optional[dict] = None
        for n, line in enumerate(read_lines(path), 1):
            m = END.match(line)
            if m and current and current["seed"] == m.group(3):
                current.update(status=m.group(4).lower(), seconds=int(m.group(5)), end=n)
                current = None
                continue
            m = START.match(line)
            if m:
                current = {"seed": m.group(3), "file": path, "start": n, "end": None, "status": "in progress",
                           "calls": []}
                seeds[m.group(3)].append(current)
                continue
            m = CALL.search(line)
            if m and current:
                current["calls"].append({"prompt": int(m.group(2)) if m.group(2) else None, "tokens": int(m.group(3)),
                                         "seconds": int(m.group(4)), "line": n})
    return seeds


def pct(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))] if v else float("nan")


def report(paths: Sequence[str], seed: Optional[str] = None, grep: Optional[str] = None) -> str:
    manifests, logs = find(paths)
    L: List[str] = ["# ISG generation report", "", f"Folders: {', '.join(paths)}",
                    f"Manifests: {len(manifests)}; logs: {len(logs)}", ""]
    if grep:
        rx = re.compile(grep)
        L += [f"## Lines matching `{grep}`", ""]
        for path in logs:
            for n, line in enumerate(read_lines(path), 1):
                if rx.search(line):
                    L.append(f"- `{path}:{n}` {line.strip()[:300]}")
        return "\n".join(L) + "\n"
    attempts = shard_logs(logs)
    loaded = []
    for p in manifests:
        with open(p, encoding="utf-8") as f:
            loaded.append((p, json.load(f)))
    if seed is not None:
        L += [f"## Seed {seed}", ""]
        for p, m in loaded:
            for key in ("seconds", "failed", "attempts", "where", "model_calls", "attempt_calls"):
                if seed in (m.get(key) or {}):
                    L.append(f"- `{p}` {key}: `{json.dumps(m[key][seed])[:600]}`")
        for a in attempts.get(seed, []):
            L += ["", f"Log `{a['file']}`, lines {a['start']}-{a['end'] or 'end'} ({a['status']}):", "", "```"]
            L += read_lines(a["file"])[a["start"] - 1:(a["end"] or 10 ** 9)]
            L += ["```"]
        return "\n".join(L) + "\n"

    pooled = merge_manifests(manifests) if manifests else None
    if pooled:
        L += ["## Totals", "", "```json", json.dumps({k: v for k, v in pooled.items() if k not in ("built", "failed")},
                                                     indent=1), "```", ""]
    # throughput
    calls = [c for tries in attempts.values() for a in tries for c in a["calls"] if c["seconds"] > 0 and c["tokens"] > 0]
    if calls:
        rates = [c["tokens"] / c["seconds"] for c in calls]
        per_shard: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        model_s = repo_s = 0
        for tries in attempts.values():
            for a in tries:
                tok = sum(c["tokens"] for c in a["calls"] if c["tokens"] > 0)
                sec = a.get("seconds") or sum(c["seconds"] for c in a["calls"])
                per_shard[a["file"]][0] += tok
                per_shard[a["file"]][1] += sec
                if a.get("seconds"):
                    model_s += sum(c["seconds"] for c in a["calls"])
                    repo_s += a["seconds"]
        machine = sum(t / s for t, s in per_shard.values() if s)
        L += ["## Throughput", "",
              f"- Per stream, per call: median {statistics.median(rates):.1f} tokens/s "
              f"(10 % {pct(rates, 0.1):.1f}, 90 % {pct(rates, 0.9):.1f}) over {len(calls)} calls.",
              f"- All {len(per_shard)} streams busy: about {machine:.0f} tokens/s for the machine "
              "(the sum of each stream's tokens over its time).",
              f"- Model calls take {100 * model_s / repo_s:.0f} % of a finished attempt's time; the rest is ISG's own "
              "work (running the code it writes) and waiting." if repo_s else "",
              f"- Prompts: largest {max((c['prompt'] or 0) for c in calls)} tokens; replies: largest "
              f"{max(c['tokens'] for c in calls)}, median {statistics.median(c['tokens'] for c in calls):.0f}.", ""]
    # repositories
    status: Dict[str, dict] = {}
    for p, m in loaded:
        shard = os.path.basename(p).replace("regeneration_manifest", "").strip(".").replace(".json", "") or "-"
        for s in m.get("seconds", {}):
            status[s] = {"status": "done", "shard": shard, "m": m}
        for s in m.get("failed", {}):
            status.setdefault(s, {"status": "failed", "shard": shard, "m": m})
    for s, tries in attempts.items():
        if tries[-1]["status"] == "in progress" and s not in status:
            status[s] = {"status": "in progress (session ended)", "shard": os.path.basename(tries[-1]["file"]), "m": {}}
    if status:
        L += ["## Repositories", "", "| Seed | Status | Shard | Attempts | Calls | Tokens | Hours | Tokens/s | "
              "Largest prompt | Largest context | Log |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for s in sorted(status, key=int):
            st, m = status[s], status[s]["m"]
            c = (m.get("model_calls") or {}).get(s, {})
            log = attempts.get(s, [None])[-1]
            seconds = (m.get("seconds") or {}).get(s) or (log or {}).get("seconds")
            tokens = c.get("tokens") or (sum(x["tokens"] for x in log["calls"] if x["tokens"] > 0) if log else None)
            n_calls = c.get("n") or (len(log["calls"]) if log else "")
            tries = (m.get("attempts") or {}).get(s) or (len(attempts.get(s, [])) or "")
            speed = f"{tokens / seconds:.1f}" if tokens and seconds else ""
            hours = f"{seconds / 3600:.2f}" if seconds else ""
            where = f"`{os.path.basename(log['file'])}:{log['start']}`" if log else ""
            L.append(f"| {s} | {st['status']} | {st['shard']} | {tries} | {n_calls} | {tokens or ''} | {hours} | {speed} | "
                     f"{c.get('max_prompt') or ''} | {c.get('max_context') or ''} | {where} |")
        L.append("")
    # failures
    failed = (pooled or {}).get("failed") or {}
    if failed:
        kinds = Counter(e.split(":")[0] for e in failed.values())
        L += ["## Failures", "", f"{len(failed)} repositories: " + ", ".join(f"{k} {v}" for k, v in kinds.most_common()),
              ""]
        for s in sorted(failed, key=int):
            first = failed[s].split("\n")[0][:200]
            L.append(f"- seed {s}: {first}")
        L.append("")
    # servers
    servers = [p for p in logs if re.match(r"(vllm|ollama)-gpu\d+\.log$", os.path.basename(p))]
    for path in servers:
        lines = read_lines(path)
        L += [f"## Server log `{path}`", ""]
        warnings = Counter()
        stats = []
        for n, line in enumerate(lines, 1):
            if any(k in line for k in SERVER_FACTS):
                L.append(f"- `:{n}` {line.strip()[:260]}")
            if "WARNING" in line:
                warnings[re.sub(r"^.*?WARNING [^\]]*\] ", "", line.strip())[:200]] += 1
            m = STATS.search(line)
            if m:
                stats.append((float(m.group(1)), int(m.group(2)), int(m.group(3)),
                              float(m.group(4)) if m.group(4) else None))
        if stats:
            busy = [s for s in stats if s[1] > 0]
            gen = [s[0] for s in busy] or [0.0]
            kv = [s[3] for s in busy if s[3] is not None]
            L += ["", f"Periodic lines: {len(stats)} ({len(busy)} with requests running). Generation "
                  f"{statistics.median(gen):.0f} tokens/s median (90 % {pct(gen, 0.9):.0f}); running "
                  f"{statistics.median([s[1] for s in busy] or [0]):.0f} median; waiting max "
                  f"{max(s[2] for s in stats)}, median {statistics.median([s[2] for s in busy] or [0]):.0f}"
                  + (f"; KV cache use median {statistics.median(kv):.0f} %, max {max(kv):.0f} %" if kv else "") + "."]
        if warnings:
            L += ["", "Warnings:"] + [f"- {k} ({v}x)" for k, v in warnings.most_common(12)]
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folders", nargs="+", help="output folders (searched recursively)")
    ap.add_argument("--seed", help="everything about one repository")
    ap.add_argument("--grep", help="every log line matching this regular expression, with file:line")
    ap.add_argument("--out", help="write here instead of printing")
    args = ap.parse_args(argv)
    text = report(args.folders, args.seed, args.grep)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
