"""Audit InfiniteScienceGym (Bentham and Srikumar, COLM 2026, arXiv:2604.13201) with the simulator's
blind baselines and quality card.

InfiniteScienceGym generates scientific data repositories from seeds and asks templated questions
about them, some unanswerable by construction (answer "not possible"). Its paper evaluates 500
questions listed in the repository's ``seeds.txt`` and reports no blind baseline. This script
measures, on the 500 items regenerated with the released code:

* **Blind baselines** (no file is opened): always "not possible"; the most common answer of the
  question's type; and a question-text policy that predicts "not possible" from what the question
  says (type, number of filter conditions, statistic, whether a number is asked for), with a
  permutation test. Each learns on a random half of the sample and answers the other half. Accuracy overall, by type and by the paper's categories, next to the paper's Table 1 and
  Figure 2, and abstention precision and recall next to its Figure 4.
* **Two graders.** The rules of the paper's Appendix J applied literally (a categorical response
  is right if it contains the answer and no other choice, in lower case; integers exactly;
  continuous answers at one significant figure fewer than requested) and an exact-match reading.
  They differ where one choice contains another: "not possible" contains "no", ".jsonl" contains
  ".json". The grading code is not released, so the script counts the items where the literal
  rule would mark a right answer wrong.
* **Reproduction.** Question types in ``seeds.txt`` against Table 7, the answerable share against
  361/139, the repositories' file types against Appendix I (76 .csv, 94 .json, 83 .jsonl, 77 .log,
  83 .txt, 87 .xlsx), and items missing from the regeneration. With ``--compare-qa``, the share of
  items whose answer is the same in two regenerations (on two machines, or two runs).

Without ``--qa`` only the ``seeds.txt`` check runs (no regeneration needed).

Regenerating (``generate``): the paper's repositories were written by Qwen3-4B-Instruct-2507 on an
H200. ``generate`` builds only the repositories the 500 items use (287 seeds, not 499), one seed
at a time, resumable, with ISG's own ``Simulator``; it loads the model in bfloat16 (what the pinned
transformers picks by default for this model) and refuses remote model code. ISG lets one model
call write up to 300,000 tokens; a call that runs on fills a 16 GB card and takes hours, so
``generate`` stops each call at 8,192 tokens (``--max-new-tokens``; 0 keeps ISG's ceiling). A cut
output fails ISG's own parser and ISG asks again, as it does for any malformed output; the
manifest counts the cut calls per repository, and a line is printed per model call.
ISG executes Python that the model writes for each dependent variable (while generating, and again
when questions are generated): run it in its own environment. Sampling on other hardware or in
another precision gives other repositories than the paper's; the blind baselines depend on the
generator's structure, not on the stories, and the reproduction checks show how far the sample is
from the paper's.

    git clone https://github.com/utahnlp/infinite-science-gym isg
    git -C isg checkout 2006f69872986e502074282ee4d23634e0cab689
    python tools/audit_isg.py audit --isg isg --out experiments_output/isg              # seeds.txt only
    python tools/audit_isg.py generate --isg isg --fs-dir isg/data/fs                   # GPU, hours
    (cd isg && python scripts/2_generate_questions.py --cache-dir ./data --n-seeded-samples 5)
    python tools/audit_isg.py audit --isg isg --qa isg/data/qa --out experiments_output/isg

Writes ``isg_audit.{json,md}`` to ``--out``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PINNED = "2006f69872986e502074282ee4d23634e0cab689"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ABSTAIN = "not possible"
TYPES = ["has_readme", "get_title", "get_abstract", "file_extension", "count_rows", "count_files_prefix",
         "count_files_conditions", "univariate_statistic_single_file", "univariate_statistic_conditions",
         "bivariate_statistic", "bivariate_hypothesis"]
# Table 7 (p. 25): category, type as the paper names it, count in the 500 evaluated questions
TABLE7 = {"has_readme": ("Repository Metadata", "Readme", 16), "get_title": ("Repository Metadata", "Title", 13),
          "get_abstract": ("Repository Metadata", "Abstract", 16), "file_extension": ("File Metadata", "Extension", 14),
          "count_rows": ("File Metadata", "Count Rows", 82), "count_files_prefix": ("Directory Traversal", "Prefix", 57),
          "count_files_conditions": ("Directory Traversal", "Condition", 53),
          "univariate_statistic_single_file": ("Univariate Statistics", "Single File", 55),
          "univariate_statistic_conditions": ("Univariate Statistics", "Condition", 61),
          "bivariate_statistic": ("Bivariate Statistics", "Statistic", 64),
          "bivariate_hypothesis": ("Bivariate Statistics", "Hypothesis", 69)}
PAPER_SPLIT = {"answerable": 361, "unanswerable": 139}  # p. 25
PAPER_EXTENSIONS = {".csv": 76, ".json": 94, ".jsonl": 83, ".log": 77, ".txt": 83, ".xlsx": 87}  # p. 25
# Table 1 (p. 5): accuracy (%) on the templated questions
TABLE1 = {"GPT-5.4": 49.2, "Claude Opus 4.6": 43.7, "GPT-OSS 20B": 29.7, "Qwen3 4B Instruct": 26.5,
          "Gemma 3 27B it": 24.2}
# Figure 2 (p. 6): accuracy (%) by category, read from the figure
FIG2 = {"Repository Metadata": {"GPT-5.4": 72.6, "Claude Opus 4.6": 67.0, "GPT-OSS 20B": 77.9, "Qwen3 4B Instruct": 70.7,
                                "Gemma 3 27B it": 67.1},
        "File Metadata": {"GPT-5.4": 51.4, "Claude Opus 4.6": 53.6, "GPT-OSS 20B": 14.7, "Qwen3 4B Instruct": 8.1,
                          "Gemma 3 27B it": 6.1},
        "Directory Traversal": {"GPT-5.4": 53.2, "Claude Opus 4.6": 44.1, "GPT-OSS 20B": 21.5,
                                "Qwen3 4B Instruct": 14.2, "Gemma 3 27B it": 11.1},
        "Univariate Statistics": {"GPT-5.4": 34.0, "Claude Opus 4.6": 24.0, "GPT-OSS 20B": 12.3,
                                  "Qwen3 4B Instruct": 9.2, "Gemma 3 27B it": 8.9},
        "Bivariate Statistics": {"GPT-5.4": 43.9, "Claude Opus 4.6": 32.8, "GPT-OSS 20B": 45.2,
                                 "Qwen3 4B Instruct": 44.2, "Gemma 3 27B it": 43.6}}
CATEGORIES = list(FIG2)
YES_NO = ["yes", "no"]
GRADERS = ("literal", "exact")


def qa_type(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def read_seeds(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def seeds_path(isg: Optional[str], seeds: Optional[str]) -> str:
    path = seeds or (os.path.join(isg, "seeds.txt") if isg else None)
    if not path or not os.path.exists(path):
        raise SystemExit("give --isg (the InfiniteScienceGym repository) or --seeds (its seeds.txt)")
    return path


# ---------------------------------------------------------------------------- items
def _norm(text: Any) -> str:
    return str(text).strip().strip("`'\"").strip().rstrip(".").strip().lower()


def _signif(x: float, digits: int) -> float:
    """ISG's ``signif`` (simulator/qa/utils.py): round to significant figures."""
    if x == 0 or not math.isfinite(x):
        return x
    if abs(x) < sys.float_info.min:
        return 0.0
    return round(x, digits - math.ceil(math.log10(abs(x))))


def load_items(keys: Sequence[str], qa_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The evaluated items from regenerated ``<fs seed>.json`` files (``2_generate_questions.py``)."""
    cache: Dict[int, Optional[Dict[str, Any]]] = {}
    items, missing = [], []
    for key in keys:
        seed_s, name = key.split(".", 1)
        seed = int(seed_s)
        if seed not in cache:
            path = os.path.join(qa_dir, f"{seed}.json")
            cache[seed] = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    cache[seed] = json.load(f)
        qa = cache[seed]
        if qa is None or name not in qa:
            missing.append(key)
            continue
        rec = qa[name]
        q = rec["question"]["question"]
        for var, value in (rec["question"].get("variables") or {}).items():
            q = q.replace(var, str(value))
        ans = rec["answer"]
        sig = re.search(r"report (\d+) significant figures", q)
        stat = (re.search(r'what is the (.+?) of the "', q) or re.search(r'what is the "(.+?)" value', q)
                or re.search(r'using "(.+?)" and a p-value', q))
        ext = (qa.get("file_extension") or {}).get("answer", {}).get("answer")
        items.append({"key": key, "fs": seed, "type": qa_type(name), "question": q,
                      "has_answer": bool(ans.get("has_answer")), "answer": ans.get("answer"),
                      "answer_type": ans.get("_type"), "sig_figs": int(sig.group(1)) if sig else None,
                      "statistic": stat.group(1) if stat else None,
                      "n_conditions": len(rec.get("path_conditions") or []) + len(rec.get("file_conditions") or []),
                      "extension": ext})
    extensions = sorted({str(i["extension"]) for i in items if i["extension"]})
    for it in items:
        if it["type"] in ("has_readme", "bivariate_hypothesis"):
            it["choices"] = YES_NO
        elif it["type"] == "file_extension":
            it["choices"] = extensions
        else:
            it["choices"] = None  # open-ended or numeric
    return items, missing


def gold_response(item: Dict[str, Any]) -> str:
    return ABSTAIN if not item["has_answer"] else _norm(item["answer"])


# ---------------------------------------------------------------------------- grading
def _contains(text: str, choice: str) -> bool:
    return _norm(choice) in text


def _to_number(text: str, integer: bool) -> Optional[float]:
    try:
        return float(int(text)) if integer else float(text)
    except ValueError:
        m = re.search(r"-?\d+" if integer else r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text)
        return float(m.group()) if m else None


def grade(item: Dict[str, Any], response: Any, grader: str = "literal") -> bool:
    """Appendix J's rules (``literal``) or exact matching (``exact``)."""
    r = _norm(response)
    choices = item.get("choices")
    if not item["has_answer"]:
        if grader == "exact":
            return r == ABSTAIN
        if choices:  # "not possible" is one more choice: it must not contain another one
            return _contains(r, ABSTAIN) and not any(_contains(r, c) for c in choices)
        return _contains(r, ABSTAIN)
    kind, gold = item["answer_type"], item["answer"]
    if kind in ("str", "bool"):
        g = _norm(gold)
        if grader == "exact":
            return r == g
        if choices:
            others = [c for c in list(choices) + [ABSTAIN] if _norm(c) != g]
            return _contains(r, g) and not any(_contains(r, o) for o in others)
        return _contains(r, g)
    if kind == "int":
        v = _to_number(r, integer=True)
        return v is not None and int(v) == int(gold)
    if kind == "float":
        v = _to_number(r, integer=False)
        if v is None or not math.isfinite(v):
            return False
        digits = max(1, (item.get("sig_figs") or 2) - 1)
        return math.isclose(_signif(v, digits), _signif(float(gold), digits), rel_tol=1e-9, abs_tol=0.0)
    return False


def grader_hazards(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Items whose right answer the literal rules mark wrong (one choice contains another)."""
    bad = [i for i in items if not grade(i, gold_response(i), "literal")]
    return {"n": len(bad), "by_type": dict(Counter(i["type"] for i in bad)),
            "examples": [{"key": i["key"], "gold": gold_response(i), "choices": i.get("choices")} for i in bad[:8]]}


# ---------------------------------------------------------------------------- blind policies
# Each policy learns on one half of the sample and answers the other half; halves are drawn many
# times and every item's hits are averaged (leave-one-out majorities are biased against balanced
# answers such as Readme's yes/no).
def _majority(counts: Counter) -> Optional[str]:
    counts = Counter({k: v for k, v in counts.items() if v > 0})
    if not counts:
        return None
    best = max(counts.values())
    return sorted(r for r, c in counts.items() if c == best)[0]


def policy_abstain(train, test, cells):
    return {i["key"]: ABSTAIN for i in test}, {i["key"]: True for i in test}


def policy_type_majority(train, test, cells):
    """The most common answer ("not possible" included) of the question's type."""
    by_type: Dict[str, Counter] = {}
    for i in train:
        by_type.setdefault(i["type"], Counter())[gold_response(i)] += 1
    resp = {i["key"]: _majority(by_type.get(i["type"], Counter())) or ABSTAIN for i in test}
    return resp, {k: v == ABSTAIN for k, v in resp.items()}


def policy_question_text(train, test, cells, min_cell: int = 3):
    """Abstain when most training items that look the same are unanswerable: same type, number of
    filter conditions, statistic and whether a number is asked for (backing off to type and number
    of conditions, then to the type, when fewer than ``min_cell`` training items share the cell);
    otherwise the type's most common answer among answerable training items."""
    levels = (lambda i: cells[i["key"]], lambda i: (i["type"], i["n_conditions"]), lambda i: i["type"])
    tallies = [Counter() for _ in levels]
    unans = [Counter() for _ in levels]
    answers: Dict[str, Counter] = {}
    for i in train:
        for lv, f in enumerate(levels):
            tallies[lv][f(i)] += 1
            unans[lv][f(i)] += not i["has_answer"]
        if i["has_answer"]:
            answers.setdefault(i["type"], Counter())[gold_response(i)] += 1
    resp, pred = {}, {}
    for i in test:
        share = 0.0
        for lv, f in enumerate(levels):
            if tallies[lv][f(i)] >= min_cell or lv == len(levels) - 1:
                share = unans[lv][f(i)] / tallies[lv][f(i)] if tallies[lv][f(i)] else 0.0
                break
        pred[i["key"]] = share > 0.5
        resp[i["key"]] = ABSTAIN if pred[i["key"]] else (_majority(answers.get(i["type"], Counter())) or ABSTAIN)
    return resp, pred


POLICIES = {"always 'not possible'": policy_abstain, "most common answer of the type": policy_type_majority,
            "question text": policy_question_text}


def _cell(it: Dict[str, Any]) -> Tuple:
    return (it["type"], it["n_conditions"], it["statistic"], it["sig_figs"] is not None)


def crossfit(items, policy, repeats: int = 20, seed: int = 0, cells: Optional[Dict[str, Tuple]] = None
             ) -> Dict[str, Any]:
    """Per item, the share of draws in which the policy's answer was right (both graders), and the
    pooled abstention counts."""
    rng = np.random.default_rng(seed)
    cells = cells or {i["key"]: _cell(i) for i in items}
    hits = {g: Counter() for g in GRADERS}
    scored, tp, fp, fn = Counter(), 0, 0, 0
    n = len(items)
    for _ in range(repeats):
        perm = rng.permutation(n)
        halves = (perm[: n // 2], perm[n // 2:])
        for a, b in (halves, halves[::-1]):
            train, test = [items[k] for k in a], [items[k] for k in b]
            resp, pred = policy(train, test, cells)
            for i in test:
                scored[i["key"]] += 1
                for g in GRADERS:
                    hits[g][i["key"]] += grade(i, resp[i["key"]], g)
                truth = not i["has_answer"]
                tp += truth and pred[i["key"]]
                fp += (not truth) and pred[i["key"]]
                fn += truth and not pred[i["key"]]
    per_item = {g: {k: hits[g][k] / scored[k] for k in scored} for g in GRADERS}
    return {"per_item": per_item, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def accuracy(items, per_item: Dict[str, float]) -> Dict[str, Any]:
    keys = [i for i in items if i["key"] in per_item]
    out: Dict[str, Any] = {"overall": float(np.mean([per_item[i["key"]] for i in keys])) if keys else None,
                           "n": len(keys)}
    out["by_type"] = {t: float(np.mean([per_item[i["key"]] for i in keys if i["type"] == t]))
                      for t in TYPES if any(i["type"] == t for i in keys)}
    out["by_category"] = {c: float(np.mean([per_item[i["key"]] for i in keys if TABLE7[i["type"]][0] == c]))
                          for c in CATEGORIES if any(TABLE7[i["type"]][0] == c for i in keys)}
    return out


def permutation_p(items, observed: Dict[str, float], n: int = 200, repeats: int = 5, seed: int = 1
                  ) -> Dict[str, float]:
    """Shuffle the question-text cells within each type: does what the question says predict?"""
    rng = np.random.default_rng(seed)
    by_type: Dict[str, List[int]] = {}
    for k, it in enumerate(items):
        by_type.setdefault(it["type"], []).append(k)
    base = [_cell(i) for i in items]
    count = Counter()
    for r in range(n):
        cells = list(base)
        for idx in by_type.values():
            for a, b in zip(idx, rng.permutation(idx)):
                cells[a] = base[b]
        res = crossfit(items, policy_question_text, repeats=repeats, seed=1000 + r,
                       cells={i["key"]: c for i, c in zip(items, cells)})
        for g in GRADERS:
            count[g] += accuracy(items, res["per_item"][g])["overall"] >= observed[g] - 1e-12
    return {g: (1 + count[g]) / (n + 1) for g in GRADERS}


# ---------------------------------------------------------------------------- reproduction checks
def seeds_check(keys: Sequence[str]) -> Dict[str, Any]:
    counts = Counter(qa_type(k.split(".", 1)[1]) for k in keys)
    rows = {t: {"paper": TABLE7[t][2], "seeds_txt": counts.get(t, 0)} for t in TYPES}
    unknown = {t: c for t, c in counts.items() if t not in TABLE7}
    return {"n_keys": len(keys), "n_repositories": len({k.split(".")[0] for k in keys}),
            "types": rows, "unknown_types": unknown,
            "types_differing": sum(1 for r in rows.values() if r["paper"] != r["seeds_txt"]),
            "abs_difference": sum(abs(r["paper"] - r["seeds_txt"]) for r in rows.values())}


def _chi2_p(observed: Sequence[float], expected: Sequence[float]) -> Optional[float]:
    from scipy import stats
    obs, exp = np.asarray(observed, float), np.asarray(expected, float)
    if obs.sum() == 0 or exp.sum() == 0:
        return None
    exp = exp * obs.sum() / exp.sum()
    return float(stats.chisquare(obs, exp).pvalue)


def regeneration_check(items: Sequence[Dict[str, Any]], missing: Sequence[str]) -> Dict[str, Any]:
    from scipy import stats
    n = len(items)
    un = sum(1 for i in items if not i["has_answer"])
    out: Dict[str, Any] = {"n_items": n, "missing": len(missing), "missing_examples": list(missing[:10]),
                           "unanswerable": un, "unanswerable_share": un / n if n else None,
                           "paper_unanswerable_share": PAPER_SPLIT["unanswerable"] / 500}
    if n:
        table = [[un, n - un], [PAPER_SPLIT["unanswerable"], PAPER_SPLIT["answerable"]]]
        out["unanswerable_vs_paper_p"] = float(stats.fisher_exact(table)[1])
    ext = Counter(str(i["extension"]) for i in items if i["extension"])
    out["extensions"] = {e: {"paper": PAPER_EXTENSIONS.get(e, 0), "regenerated": ext.get(e, 0)}
                         for e in sorted(set(PAPER_EXTENSIONS) | set(ext))}
    keys = sorted(out["extensions"])
    out["extensions_vs_paper_p"] = _chi2_p([ext.get(e, 0) for e in keys], [PAPER_EXTENSIONS.get(e, 0) + 0.5 for e in keys])
    out["unanswerable_by_type"] = {t: {"n": sum(1 for i in items if i["type"] == t),
                                       "unanswerable": sum(1 for i in items if i["type"] == t and not i["has_answer"])}
                                   for t in TYPES}
    return out


def compare_regenerations(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gb = {i["key"]: i for i in b}
    shared = [i for i in a if i["key"] in gb]
    same_q = [i for i in shared if i["question"] == gb[i["key"]]["question"]]
    same_a = [i for i in shared if gold_response(i) == gold_response(gb[i["key"]])]
    same_h = [i for i in shared if i["has_answer"] == gb[i["key"]]["has_answer"]]
    return {"shared": len(shared), "same_question": len(same_q), "same_answer": len(same_a),
            "same_answerability": len(same_h)}


def quality_card(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per type: unanswerable share, the most common answer and its share, distinct answers."""
    card = {}
    for t in TYPES:
        group = [i for i in items if i["type"] == t]
        if not group:
            continue
        golds = Counter(gold_response(i) for i in group)
        top, top_n = golds.most_common(1)[0]
        card[t] = {"n": len(group), "unanswerable_share": sum(not i["has_answer"] for i in group) / len(group),
                   "majority_answer": top if len(top) <= 40 else top[:37] + "...", "majority_share": top_n / len(group),
                   "distinct_answers": len(golds),
                   "unanswerable_by_conditions": {str(k): float(np.mean([not i["has_answer"] for i in group
                                                                         if i["n_conditions"] == k]))
                                                  for k in sorted({i["n_conditions"] for i in group})}}
    return card


# ---------------------------------------------------------------------------- report
def run_audit(args) -> Dict[str, Any]:
    keys = read_seeds(seeds_path(args.isg, args.seeds))
    report: Dict[str, Any] = {"generated": _dt.datetime.now().isoformat(timespec="seconds"),
                              "isg": _provenance(args.isg), "seeds_check": seeds_check(keys)}
    if not args.qa:
        return report
    items, missing = load_items(keys, args.qa)
    report["regeneration"] = regeneration_check(items, missing)
    if args.regeneration_manifest and os.path.exists(args.regeneration_manifest):
        with open(args.regeneration_manifest, encoding="utf-8") as f:
            report["generation_manifest"] = json.load(f)
    if not items:
        return report
    report["quality_card"] = quality_card(items)
    report["repeats"] = args.repeats
    report["grader_hazards"] = grader_hazards(items)
    report["baselines"], report["abstention"] = {}, {}
    for name, policy in POLICIES.items():
        res = crossfit(items, policy, repeats=args.repeats)
        report["baselines"][name] = {g: accuracy(items, res["per_item"][g]) for g in GRADERS}
        report["abstention"][name] = {"precision": res["precision"], "recall": res["recall"]}
    text = report["baselines"]["question text"]
    p = permutation_p(items, {g: text[g]["overall"] for g in GRADERS}, n=args.permutations)
    for g in GRADERS:
        text[g]["permutation_p"] = p[g]
    if args.compare_qa:
        other, _ = load_items(keys, args.compare_qa)
        report["compare"] = compare_regenerations(items, other)
    return report


def _provenance(isg: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"pinned": PINNED}
    if isg:
        import subprocess
        try:
            out["commit"] = subprocess.run(["git", "-C", isg, "rev-parse", "HEAD"], capture_output=True, text=True,
                                           timeout=30).stdout.strip() or None
        except Exception:
            out["commit"] = None
        out["matches_pin"] = out["commit"] == PINNED if out.get("commit") else None
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def markdown(r: Dict[str, Any]) -> str:
    L = ["# InfiniteScienceGym audited with scidatasim's blind baselines", "",
         f"Generated {r['generated']} by `tools/audit_isg.py`. Code: {r['isg'].get('commit') or 'not given'}"
         + ("" if r["isg"].get("matches_pin") in (None, True) else " (**not the pinned commit**)") + ".", ""]
    s = r["seeds_check"]
    L += ["## `seeds.txt` against Table 7", "",
          f"{s['n_keys']} items from {s['n_repositories']} repositories. {s['types_differing']} of 11 types differ "
          f"from Table 7 (total absolute difference {s['abs_difference']}).", "",
          "| Category | Type | Table 7 | seeds.txt |", "|---|---|---:|---:|"]
    for t, row in s["types"].items():
        L.append(f"| {TABLE7[t][0]} | {TABLE7[t][1]} | {row['paper']} | {row['seeds_txt']} |")
    L.append("")
    g = r.get("regeneration")
    if g:
        L += ["## Regenerated sample", "",
              f"{g['n_items']} items regenerated, {g['missing']} missing. Unanswerable: {g['unanswerable']} "
              f"({_pct(g['unanswerable_share'])} %; paper 27.8 %, Fisher p = {g.get('unanswerable_vs_paper_p', float('nan')):.3g}).",
              "", "| File type | Appendix I | Regenerated |", "|---|---:|---:|"]
        for e, row in g["extensions"].items():
            L.append(f"| {e} | {row['paper']} | {row['regenerated']} |")
        L += ["", f"File types against the paper: chi-square p = {g['extensions_vs_paper_p']:.3g}"
              if g.get("extensions_vs_paper_p") is not None else "", ""]
        if r.get("generation_manifest"):
            m = r["generation_manifest"]
            L += [f"Generated with {m.get('model')} in {m.get('dtype')} on {m.get('device')}; "
                  f"{m.get('n_done')} repositories, {m.get('n_failed')} failed.", ""]
        if r.get("compare"):
            c = r["compare"]
            L += [f"Two regenerations: {c['same_question']}/{c['shared']} identical questions, "
                  f"{c['same_answer']}/{c['shared']} identical answers, {c['same_answerability']}/{c['shared']} "
                  "the same answerability.", ""]
    b = r.get("baselines")
    if b:
        L += ["## Blind baselines", "",
              "No file is opened. Each policy learns on a random half of the sample and answers the other half "
              f"({r['repeats']} draws). Accuracy (%) under the paper's rules applied literally and under exact "
              "matching; models from Table 1 (templated questions).", "",
              "| Policy | Literal rules | Exact match |", "|---|---:|---:|"]
        for name, res in b.items():
            L.append(f"| {name} | {_pct(res['literal']['overall'])} | {_pct(res['exact']['overall'])} |")
        for m, v in TABLE1.items():
            L.append(f"| {m} (paper) | {v:.1f} | |")
        qt = b["question text"]
        L += ["", f"Question-text policy against shuffled cells: p = {qt['literal'].get('permutation_p', float('nan')):.3g} "
              f"(literal), {qt['exact'].get('permutation_p', float('nan')):.3g} (exact).", "",
              "By category (exact match; models from Figure 2, read from the figure):", "",
              "| Category | " + " | ".join(b) + " | " + " | ".join(TABLE1) + " |",
              "|---|" + "---:|" * (len(b) + len(TABLE1))]
        for c in CATEGORIES:
            L.append(f"| {c} | " + " | ".join(_pct(res["exact"]["by_category"].get(c)) for res in b.values())
                     + " | " + " | ".join(f"{FIG2[c][m]:.1f}" for m in TABLE1) + " |")
        L += ["", "By type (exact match):", "", "| Type | n | " + " | ".join(b) + " |",
              "|---|---:|" + "---:|" * len(b)]
        card = r["quality_card"]
        for t in TYPES:
            if t in card:
                L.append(f"| {TABLE7[t][1]} ({TABLE7[t][0]}) | {card[t]['n']} | "
                         + " | ".join(_pct(res["exact"]["by_type"].get(t)) for res in b.values()) + " |")
        a = r["abstention"]["question text"]
        L += ["", f"Detecting unanswerable questions from the question text alone: precision {_pct(a['precision'])} %, "
              f"recall {_pct(a['recall'])} % (paper, Figure 4: the best models at or below 83 % on both).", ""]
        h = r["grader_hazards"]
        L += ["## Grading rules", "",
              f"Under Appendix J's rules applied literally, {h['n']} right answers would be marked wrong: "
              + (", ".join(f"{TABLE7[t][1]} {n}" for t, n in h["by_type"].items()) or "none")
              + ". A correct \"not possible\" contains the choice \"no\"; \".jsonl\" contains \".json\". The grading "
              "code is not in the repository, so whether the paper's numbers are affected cannot be checked.", "",
              "## Quality card", "", "| Type | n | Unanswerable | Most common answer | Its share | Distinct |",
              "|---|---:|---:|---|---:|---:|"]
        for t, c in card.items():
            L.append(f"| {TABLE7[t][1]} | {c['n']} | {_pct(c['unanswerable_share'])} % | {c['majority_answer']} | "
                     f"{_pct(c['majority_share'])} % | {c['distinct_answers']} |")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------- generate
def run_generate(args) -> int:
    """ISG's ``1_generate_filesystem.py`` loop, for the seeds the 500 items use, resumable."""
    isg = os.path.abspath(args.isg)
    keys = read_seeds(seeds_path(args.isg, args.seeds))
    seeds = sorted({int(k.split(".")[0]) for k in keys})
    if args.limit:
        seeds = seeds[:args.limit]
    os.makedirs(args.fs_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(os.path.join(args.fs_dir, f"{s}.json"))]
    print(f"{len(seeds)} repositories needed, {len(seeds) - len(todo)} already in {args.fs_dir}", flush=True)
    if not todo:
        return 0
    import functools

    import torch
    sys.path.insert(0, isg)
    import simulator.llm as isg_llm  # ISG's code, from --isg
    from simulator import Simulator
    from simulator.config import FileConfig, FileSystemConfig, StoryConfig
    from simulator.utils import load_taxonomy
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    original = isg_llm.pipeline
    calls = {"n": 0, "tokens": 0, "cut": 0, "seconds": 0.0}

    class Capped:
        """ISG's text-generation pipeline with a ceiling on new tokens and a line per call."""

        def __init__(self, pipe):
            self._pipe = pipe

        def __getattr__(self, name):
            return getattr(self._pipe, name)

        def __call__(self, *a, **kw):
            if args.max_new_tokens:  # ISG asks for up to 300000
                kw["max_new_tokens"] = min(int(kw.get("max_new_tokens") or args.max_new_tokens), args.max_new_tokens)
            t0 = time.time()
            out = self._pipe(*a, **kw)
            seconds = time.time() - t0
            try:
                text = out[0]["generated_text"][-1]["content"]
                tokens = len(self._pipe.tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                tokens = -1
            cut = bool(args.max_new_tokens) and tokens >= args.max_new_tokens - 1
            calls["n"] += 1
            calls["tokens"] += max(tokens, 0)
            calls["cut"] += cut
            calls["seconds"] += seconds
            print(f"    model call {calls['n']}: {tokens} tokens, {seconds:.0f} s" + (" - cut at the ceiling" if cut else ""),
                  flush=True)
            return out

    def pipeline(*a, **kw):
        kw["trust_remote_code"] = False  # Qwen3 is built into transformers; no model-supplied code
        kw.setdefault("torch_dtype", dtype)
        return Capped(original(*a, **kw))
    isg_llm.pipeline = functools.wraps(original)(pipeline)
    sim = Simulator(story_config=StoryConfig(taxonomy=load_taxonomy(os.path.join(isg, "taxonomy"))),
                    fs_config=FileSystemConfig(), file_config=FileConfig(), device_map=args.device_map,
                    model_name=args.model)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    manifest_path = os.path.join(args.fs_dir, "regeneration_manifest.json")
    manifest = {"model": args.model, "dtype": args.dtype, "device": device, "isg": _provenance(isg),
                "torch": torch.__version__, "max_new_tokens": args.max_new_tokens or 300000,
                "seconds": {}, "failed": {}, "model_calls": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest.update({k: v for k, v in json.load(f).items() if k in ("seconds", "failed", "model_calls")})
    t_start = time.time()
    for n, seed in enumerate(todo, 1):
        t0 = time.time()
        calls.update(n=0, tokens=0, cut=0, seconds=0.0)
        print(f"[{n}/{len(todo)}] seed {seed}", flush=True)
        try:
            fs = sim.get_filesystem(seed)
            final = os.path.join(args.fs_dir, f"{seed}.json")
            with open(final + ".tmp", "w", encoding="utf-8") as f:  # complete or absent, even after Ctrl+C
                json.dump(fs.to_json(), f, indent=2)
            os.replace(final + ".tmp", final)
            manifest["seconds"][str(seed)] = round(time.time() - t0, 1)
            manifest["failed"].pop(str(seed), None)
        except Exception as e:  # ISG marks such seeds corrupted; the item will be missing
            manifest["failed"][str(seed)] = f"{type(e).__name__}: {e}"[:300]
        manifest["model_calls"][str(seed)] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in calls.items()}
        manifest["n_done"] = len(manifest["seconds"])
        manifest["n_failed"] = len(manifest["failed"])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        rate = (time.time() - t_start) / n
        print(f"[{n}/{len(todo)}] seed {seed} {'done' if str(seed) in manifest['seconds'] else 'FAILED'} in "
              f"{time.time() - t0:.0f} s ({calls['n']} model calls, {calls['cut']} cut); about "
              f"{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="baselines and checks on the regenerated items (or seeds.txt alone)")
    a.add_argument("--isg", help="the InfiniteScienceGym repository (for seeds.txt and the commit)")
    a.add_argument("--seeds", help="seeds.txt, if not in --isg")
    a.add_argument("--qa", help="the regenerated question files (data/qa from 2_generate_questions.py)")
    a.add_argument("--compare-qa", help="a second regeneration, for a determinism check")
    a.add_argument("--regeneration-manifest", help="regeneration_manifest.json written by 'generate'")
    a.add_argument("--permutations", type=int, default=200)
    a.add_argument("--repeats", type=int, default=20, help="random halves for the blind policies")
    a.add_argument("--out", default=os.path.join("experiments_output", "isg"))
    gsub = sub.add_parser("generate", help="regenerate the repositories the 500 items use (GPU)")
    gsub.add_argument("--isg", required=True)
    gsub.add_argument("--seeds")
    gsub.add_argument("--fs-dir", required=True, help="where <seed>.json go (ISG's data/fs)")
    gsub.add_argument("--model", default=MODEL)
    gsub.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                      help="bfloat16 is what transformers 4.57's default ('auto') picks for Qwen3-4B")
    gsub.add_argument("--max-new-tokens", type=int, default=8192,
                      help="ceiling per model call (ISG asks for 300000; 0 keeps that)")
    gsub.add_argument("--device-map", default="auto")
    gsub.add_argument("--limit", type=int, default=0, help="only the first N repositories (to time a run)")
    args = p.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    if args.qa and not args.regeneration_manifest:
        guess = os.path.join(os.path.dirname(os.path.abspath(args.qa)), "fs", "regeneration_manifest.json")
        args.regeneration_manifest = guess if os.path.exists(guess) else None
    report = run_audit(args)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "isg_audit.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(args.out, "isg_audit.md"), "w", encoding="utf-8") as f:
        f.write(markdown(report))
    print(f"wrote {os.path.join(args.out, 'isg_audit.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
