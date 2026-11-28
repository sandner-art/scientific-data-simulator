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
manifest counts the cut calls per repository, and a line is printed per model call. The model goes
entirely on the GPU (``--device-map cuda:0``): ISG's ``auto`` moves the layers that do not fit
into free GPU memory to the CPU without saying so, and generation then runs at CPU speed.
``generate`` prints where the model sits and stops if part of it is off the GPU (``--allow-cpu``).

``--backend ollama`` lets a local Ollama server write instead of transformers (on Windows the
transformers path managed 5 tokens/s on an RTX A4000). The model is the same,
Qwen3-4B-Instruct-2507 (``qwen3:4b-instruct-2507-fp16``, F16 instead of bfloat16); every prompt is
rendered by the model's own chat template through its tokenizer, exactly as ISG's pipeline renders
it; ISG's sampling parameters (temperature 0.7, top-p 0.8, top-k 20) are passed, with a seed per
repository and call; replies come back in the pipeline's shape. Sampling by another engine gives
other text than transformers would, as other hardware does too.

Several repositories at once (``--shard I/N``): the seeds are dealt into N interleaved shards and a
process builds shard I, so several processes against one Ollama server (with
``OLLAMA_NUM_PARALLEL`` slots), or processes on several machines, split the work without overlap.
Each shard keeps its own manifest (``regeneration_manifest.shard-I-of-N.json``), and every manifest
records per repository the machine (``--machine``; default: the GPUs ``nvidia-smi`` lists) and
the engine that built it; ``audit`` pools the manifests it finds next to the questions.
``--stop-after H`` starts no repository after H hours (a Kaggle session ends at 12 h);
``--ollama-digest`` refuses a model whose digest differs, so every machine writes with the same
weights. Each model call's line gives the prompt and reply lengths, and the manifest the largest
context per repository, which sets how many slots fit in GPU memory. Parallel decoding gives each
call the same seed as before, but batches change the floating-point sums, so two runs can differ.
``docs/reproduce/isg_regeneration.md`` gives the steps, locally and on Kaggle.

``--backend openai`` sends the same rendered prompts to an OpenAI-compatible server
(``/v1/completions``) such as vLLM, which batches the calls of many processes on one GPU far
better than Ollama's slots (``kaggle/isg_generate.ipynb`` runs it on two T4s). The tool refuses a
server that does not serve ``--served-model`` or holds another context length than ``--num-ctx``,
and records ``--weights-revision`` and ``--note`` (how the server was started).

A generation cut short by a time budget finishes quick repositories first, and these may hold
easier questions; ``audit --complete-prefix`` therefore analyses only the longest run of seeds,
in ascending order, that were all attempted, a sample fixed by seed order rather than by speed.

ISG executes Python that the model writes for each dependent variable (while generating, and again
when questions are generated): run it in its own environment. Sampling on other hardware or in
another precision gives other repositories than the paper's; the blind baselines depend on the
generator's structure, not on the stories, and the reproduction checks show how far the sample is
from the paper's.

    git clone https://github.com/utahnlp/infinite-science-gym isg
    git -C isg checkout 2006f69872986e502074282ee4d23634e0cab689
    python tools/audit_isg.py audit --isg isg --out experiments_output/isg              # seeds.txt only
    python tools/audit_isg.py generate --isg isg --fs-dir isg/data/fs                   # GPU, hours
    python tools/audit_isg.py generate --isg isg --fs-dir isg/data/fs --backend ollama --shard 1/2  # and 2/2
    (cd isg && python scripts/2_generate_questions.py --cache-dir ./data --n-seeded-samples 5)
    python tools/audit_isg.py audit --isg isg --qa isg/data/qa --out experiments_output/isg

Writes ``isg_audit.{json,md}`` to ``--out``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
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
    paths = manifest_paths(args.regeneration_manifest)
    if paths:
        report["generation_manifest"] = merge_manifests(paths)
    if getattr(args, "complete_prefix", False):
        if not paths:
            raise SystemExit("--complete-prefix needs the generation manifests (--regeneration-manifest)")
        keys, report["sample"] = complete_prefix(keys, report["generation_manifest"])
    items, missing = load_items(keys, args.qa)
    report["regeneration"] = regeneration_check(items, missing)
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


def complete_prefix(keys: Sequence[str], pooled: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """The items of the longest run of seeds, in ascending order, that were all attempted.

    A run stopped by a time budget finishes quick repositories first, and quick repositories may
    hold easier questions. Cutting at the first seed not yet attempted keeps the sample a fixed
    rule of seed order (the seeds are random keys) instead of a rule of speed. Seeds that failed
    inside the prefix stay in it; their items count as missing."""
    seeds = sorted({int(k.split(".")[0]) for k in keys})
    attempted = set(pooled.get("built", [])) | set(pooled.get("failed", {}))
    prefix: List[int] = []
    for s in seeds:
        if str(s) not in attempted:
            break
        prefix.append(s)
    chosen = set(prefix)
    kept = [k for k in keys if int(k.split(".")[0]) in chosen]
    built_outside = sorted(int(s) for s in pooled.get("built", []) if int(s) not in chosen and int(s) in set(seeds))
    return kept, {"rule": "longest prefix of the needed seeds (ascending) that were all attempted",
                  "repositories": len(prefix), "of_repositories": len(seeds), "last_seed": prefix[-1] if prefix else None,
                  "items": len(kept), "of_items": len(keys), "failed_inside": sorted(
                      int(s) for s in pooled.get("failed", {}) if int(s) in chosen),
                  "built_but_left_out": len(built_outside)}


def manifest_paths(path: Optional[str]) -> List[str]:
    """A manifest file, or every ``regeneration_manifest*.json`` in a directory (one per shard)."""
    if not path:
        return []
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "regeneration_manifest*.json")))
    return [path] if os.path.exists(path) else []


def _engine(m: Dict[str, Any]) -> str:
    """Who wrote, from a manifest's top-level fields (manifests before 0.8 record no per-seed machine)."""
    b = m.get("backend") or {}
    if b.get("name") == "openai":
        rev = str(b.get("weights_revision") or "")[:12]
        engine = "vLLM" if b.get("engine") == "vllm" else "OpenAI-compatible server"
        return (f"{engine} {b.get('version') or ''}".strip() + f", {b.get('model')}" + (f" ({rev})" if rev else ""))
    if b.get("name") == "ollama":
        return f"Ollama {b.get('version')}, {b.get('model')} ({str(b.get('digest') or '')[:12]})"
    return f"transformers, {m.get('model')} in {m.get('dtype')}"


def merge_manifests(paths: Sequence[str]) -> Dict[str, Any]:
    """Pool the manifests of shards and machines: a repository built anywhere counts as built."""
    seconds: Dict[str, float] = {}
    failed: Dict[str, str] = {}
    calls: Dict[str, Dict[str, Any]] = {}
    where: Dict[str, str] = {}
    attempts: Dict[str, int] = {}
    spent = {"tokens": 0, "seconds": 0.0}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        built = m.get("seconds") or {}
        seconds.update(built)
        calls.update(m.get("model_calls") or {})
        for s, e in (m.get("failed") or {}).items():
            failed.setdefault(s, e)
        for s, k in (m.get("attempts") or {}).items():
            attempts[s] = max(attempts.get(s, 0), int(k))
        for tries in (m.get("attempt_calls") or {}).values():
            spent["tokens"] += sum(t.get("tokens", 0) for t in tries)
            spent["seconds"] += sum(t.get("seconds_total", 0) for t in tries)
        default = f"{_engine(m)} on {m.get('device')}"
        recorded = m.get("where") or {}
        for s in built:
            where[s] = recorded.get(s, default)
    for s in seconds:
        failed.pop(s, None)
    done = [calls[s] for s in seconds if s in calls]
    contexts = [c["max_context"] for c in done if c.get("max_context")]
    prompts = [c["max_prompt"] for c in done if c.get("max_prompt")]
    return {"files": [os.path.basename(p) for p in paths], "n_done": len(seconds), "n_failed": len(failed),
            "built": sorted(seconds, key=int),
            "failed": failed, "built_with": dict(Counter(where.values()).most_common()),
            "model_calls": sum(c.get("n", 0) for c in done), "tokens": sum(c.get("tokens", 0) for c in done),
            "cut": sum(c.get("cut", 0) for c in done), "largest_context": max(contexts) if contexts else None,
            "largest_prompt": max(prompts) if prompts else None,
            "repository_hours": round(sum(seconds.values()) / 3600, 1),
            "attempts": sum(attempts.values()) if attempts else None,
            "built_after_retry": sum(1 for s in seconds if attempts.get(s, 1) > 1),
            "tokens_all_attempts": spent["tokens"] or None,
            "hours_all_attempts": round(spent["seconds"] / 3600, 1) if spent["seconds"] else None}


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
            L += [f"Generation ({len(m['files'])} manifest{'s' if len(m['files']) != 1 else ''}): {m['n_done']} "
                  f"repositories built, {m['n_failed']} failed; {m['model_calls']} model calls, {m['cut']} cut (ceiling or "
                  f"full context); {m['repository_hours']} h summed over repositories"
                  + (f"; largest context {m['largest_context']} tokens" if m.get("largest_context") else "")
                  + (f" (largest prompt {m['largest_prompt']})" if m.get("largest_prompt") else "") + ".",
                  "", "| Built with | Repositories |", "|---|---:|"]
            L += [f"| {k} | {v} |" for k, v in m["built_with"].items()] + [""]
        if r.get("sample"):
            sm = r["sample"]
            L += [f"Sample: the {sm['repositories']} of {sm['of_repositories']} repositories up to seed {sm['last_seed']} "
                  f"(ascending seed order, all attempted), {sm['items']} of {sm['of_items']} items; "
                  f"{len(sm['failed_inside'])} failed inside it; {sm['built_but_left_out']} built later in the order "
                  "left out, so speed does not choose the sample.", ""]
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
OLLAMA_MODEL = "qwen3:4b-instruct-2507-fp16"  # the same Qwen3-4B-Instruct-2507, F16 GGUF
QWEN_STOP = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]


def load_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


class Ollama:
    """A minimal client for a local Ollama server (standard library HTTP)."""

    def __init__(self, url: str, model: str, num_ctx: int, timeout: float = 7200):
        self.url, self.model, self.num_ctx, self.timeout = url.rstrip("/"), model, num_ctx, timeout

    def _request(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama {path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
        except urllib.error.URLError as e:
            raise ConnectionError(f"Ollama is not reachable at {self.url} ({e.reason})") from None

    def version(self) -> Optional[str]:
        return self._request("/api/version").get("version")

    def load(self) -> None:
        self._request("/api/generate", {"model": self.model, "prompt": "", "keep_alive": "60m",
                                        "options": {"num_ctx": self.num_ctx}})

    def placement(self) -> Dict[str, Any]:
        for m in self._request("/api/ps").get("models", []):
            if self.model in (m.get("name"), m.get("model")):
                return {"size_gb": round(m.get("size", 0) / 2 ** 30, 2),
                        "vram_gb": round(m.get("size_vram", 0) / 2 ** 30, 2), "digest": m.get("digest"),
                        "quantization": (m.get("details") or {}).get("quantization_level")}
        return {}

    def generate(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("/api/generate", {"model": self.model, "prompt": prompt, "raw": True, "stream": False,
                                               "keep_alive": "60m", "options": options})


class _TemplatePipe:
    """Stands in for ISG's transformers text-generation pipeline: the chat is rendered as the
    pipeline renders it (the model's own chat template, through its tokenizer), a server writes
    the reply with ISG's sampling parameters, and the result has the pipeline's shape (a prefilled
    final message is continued, as with ``continue_final_message``)."""

    def __init__(self, client, tokenizer, seed_of_call, num_ctx: int):
        self.client, self.tokenizer, self.seed_of_call, self.num_ctx = client, tokenizer, seed_of_call, num_ctx
        self.last: Optional[Dict[str, Any]] = None

    def _render(self, messages, kw):
        cont = bool(kw.get("continue_final_message"))
        prompt = self.tokenizer.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=not cont,
                                                    continue_final_message=cont)
        n_prompt = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        room = self.num_ctx - n_prompt - 16
        if room < 64:
            raise RuntimeError(f"a prompt of {n_prompt} tokens does not fit the context of {self.num_ctx}")
        sample = kw.get("do_sample", True)
        sampling = {"limit": min(int(kw.get("max_new_tokens") or room), room),
                    "temperature": float(kw.get("temperature", 1.0)) if sample else 0.0,
                    "top_p": float(kw.get("top_p", 1.0)), "top_k": int(kw.get("top_k", 0) or 0)}
        return cont, prompt, n_prompt, sampling

    @staticmethod
    def _shape(messages, cont: bool, text: str):
        if cont:
            out = list(messages)[:-1] + [{"role": messages[-1]["role"], "content": messages[-1]["content"] + text}]
        else:
            out = list(messages) + [{"role": "assistant", "content": text}]
        return [{"generated_text": out}]


class OllamaPipe(_TemplatePipe):
    """ISG's pipeline, written by Ollama (``/api/generate`` with the rendered prompt, raw)."""

    def __init__(self, client: Ollama, tokenizer, seed_of_call):
        super().__init__(client, tokenizer, seed_of_call, client.num_ctx)

    def __call__(self, messages, **kw):
        cont, prompt, n_prompt, s = self._render(messages, kw)
        options = {"num_ctx": self.num_ctx, "num_predict": s["limit"], "temperature": s["temperature"],
                   "top_p": s["top_p"], "top_k": s["top_k"], "min_p": 0.0, "repeat_penalty": 1.0,
                   "seed": self.seed_of_call(), "stop": QWEN_STOP}
        r = self.client.generate(prompt, options)
        self.last = {"tokens": r.get("eval_count"), "prompt_tokens": r.get("prompt_eval_count", n_prompt),
                     "generation_seconds": (r.get("eval_duration") or 0) / 1e9, "limit": s["limit"],
                     "done_reason": r.get("done_reason")}
        return self._shape(messages, cont, r.get("response", ""))


class OpenAIPipe(_TemplatePipe):
    """ISG's pipeline, written by an OpenAI-compatible server such as vLLM (``/v1/completions``
    with the rendered prompt; vLLM takes top-k, min-p and the repetition penalty in the request)."""

    def __call__(self, messages, **kw):
        cont, prompt, n_prompt, s = self._render(messages, kw)
        text = self.client.complete(prompt, max_tokens=s["limit"], temperature=s["temperature"], top_p=s["top_p"],
                                    top_k=s["top_k"] or -1, min_p=0.0, repetition_penalty=1.0,
                                    seed=self.seed_of_call(), stop=QWEN_STOP)
        r = self.client.last or {}
        self.last = {"tokens": r.get("tokens"), "prompt_tokens": r.get("prompt_tokens", n_prompt),
                     "generation_seconds": None, "limit": s["limit"], "done_reason": r.get("stopped")}
        return self._shape(messages, cont, text)


def _shard(text: str) -> Tuple[int, int]:
    from scidatasim.ai.local import parse_shard
    try:
        return parse_shard(text or "")
    except ValueError:
        raise argparse.ArgumentTypeError(f"I/N with 1 <= I <= N, not {text!r}") from None


def _digest(text: Any) -> str:
    return str(text or "").lower().split(":")[-1]


def machine_label(given: Optional[str], torch_mod) -> str:
    """``--machine``, else the GPUs nvidia-smi lists (2 x Tesla T4), else torch's first GPU."""
    if given:
        return given
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout
        names = [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:  # no NVIDIA driver here, or nvidia-smi not on PATH
        names = []
    if not names and torch_mod.cuda.is_available():
        names = [torch_mod.cuda.get_device_name(0)]
    if not names:
        return "no GPU found"
    return ", ".join(f"{n} x {name}" if n > 1 else name for name, n in Counter(names).items())


def model_placement(sim) -> Dict[str, int]:
    """Where the loaded model sits: device -> number of modules (``hf_device_map``), or
    parameters when the model has no device map."""
    model = getattr(getattr(getattr(sim, "llm", None), "pipe", None), "model", None)
    if model is None:
        return {}
    where: Counter = Counter()
    layout = getattr(model, "hf_device_map", None)
    if layout:
        for v in layout.values():
            where[f"cuda:{v}" if isinstance(v, int) else str(v)] += 1
    else:
        try:
            for param in model.parameters():
                where[str(param.device)] += 1
        except Exception:  # not a torch model: nothing to report
            return {}
    return dict(where)


def run_generate(args) -> int:
    """ISG's ``1_generate_filesystem.py`` loop, for the seeds the 500 items use, resumable."""
    t_begin = time.time()
    isg = os.path.abspath(args.isg)
    keys = read_seeds(seeds_path(args.isg, args.seeds))
    seeds = sorted({int(k.split(".")[0]) for k in keys})
    if args.limit:
        seeds = seeds[:args.limit]
    shard, n_shards = args.shard
    seeds = seeds[shard - 1::n_shards]  # interleaved: every shard gets early and late seeds alike
    os.makedirs(args.fs_dir, exist_ok=True)
    todo = [s for s in seeds if not os.path.exists(os.path.join(args.fs_dir, f"{s}.json"))]
    part = f" in shard {shard} of {n_shards}" if n_shards > 1 else ""
    print(f"{len(seeds)} repositories needed{part}, {len(seeds) - len(todo)} already in {args.fs_dir}", flush=True)
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
    fresh = {"n": 0, "tokens": 0, "cut": 0, "seconds": 0.0, "prompt_tokens": None, "max_prompt": None,
             "max_context": None}
    calls = dict(fresh)
    run_context = {"max": 0, "prompt": 0}

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
            last = getattr(self._pipe, "last", None)
            try:
                if last and last.get("tokens") is not None:
                    tokens = int(last["tokens"])
                else:
                    text = out[0]["generated_text"][-1]["content"]
                    tokens = len(self._pipe.tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception:
                tokens = -1
            # Ollama says why it stopped; its limit is lower than the ceiling when the context is short
            ceiling = (last or {}).get("limit") or args.max_new_tokens
            if last and last.get("done_reason"):
                cut = last["done_reason"] == "length"
            else:
                cut = bool(ceiling) and tokens >= ceiling - 1
            calls["n"] += 1
            calls["tokens"] += max(tokens, 0)
            calls["cut"] += cut
            calls["seconds"] += seconds
            prompt = last.get("prompt_tokens") if last else None  # known when the backend reports it
            if prompt is not None:
                context = int(prompt) + max(tokens, 0)
                calls["prompt_tokens"] = (calls["prompt_tokens"] or 0) + int(prompt)
                calls["max_prompt"] = max(calls["max_prompt"] or 0, int(prompt))
                calls["max_context"] = max(calls["max_context"] or 0, context)
                run_context["max"] = max(run_context["max"], context)
                run_context["prompt"] = max(run_context["prompt"], int(prompt))
            writing = last.get("generation_seconds") if last else None
            rate = f", {tokens / (writing or seconds):.0f} tokens/s" if (writing or seconds) > 0 and tokens > 0 else ""
            print(f"    model call {calls['n']}: " + (f"prompt {prompt} + " if prompt is not None else "")
                  + f"{tokens} tokens, {seconds:.0f} s{rate}" + (" - cut at the ceiling" if cut else ""), flush=True)
            return out

    current = {"seed": 0, "attempt": 0}
    backend: Dict[str, Any] = {"name": args.backend}

    def seed_of_call():  # reproducible per repository, attempt and call; attempt 0 as before
        return (current["seed"] * 100003 + current["attempt"] * 7777777 + calls["n"]) % 2 ** 31
    if args.backend == "ollama":
        client = Ollama(args.ollama_url, args.ollama_model, args.num_ctx)
        try:
            backend["version"] = client.version()
            client.load()
            backend.update(client.placement())
        except (ConnectionError, RuntimeError) as e:
            raise SystemExit(f"{e}. Start Ollama (ollama serve) and fetch the model (ollama pull {args.ollama_model}).")
        backend.update(model=args.ollama_model, num_ctx=args.num_ctx, url=args.ollama_url)
        print(f"Ollama {backend.get('version')}: {args.ollama_model} ({_digest(backend.get('digest'))[:12]}), "
              f"{backend.get('vram_gb', 0):.1f} of {backend.get('size_gb', 0):.1f} GB on the GPU, context "
              f"{args.num_ctx} tokens per slot", flush=True)
        if args.ollama_digest and not _digest(backend.get("digest")).startswith(_digest(args.ollama_digest)):
            raise SystemExit(f"Ollama serves {args.ollama_model} with digest {_digest(backend.get('digest'))[:12]}, not "
                             f"{args.ollama_digest}: other weights than the other machines'. Pull the model again or "
                             "check the name.")
        if backend.get("size_gb") and backend.get("vram_gb", 0) < backend["size_gb"] - 0.01 and not args.allow_cpu:
            raise SystemExit("part of the model is on the CPU, so it would write at CPU speed. Ollama reserves "
                             "--num-ctx tokens of context for each of its OLLAMA_NUM_PARALLEL slots: lower one of "
                             "them, or close programs that use GPU memory; --allow-cpu runs anyway.")
        tokenizer = load_tokenizer(args.tokenizer or args.model)

        def pipeline(*a, **kw):
            return Capped(OllamaPipe(client, tokenizer, seed_of_call))
    elif args.backend == "openai":
        from scidatasim.ai.local import OpenAICompatibleClient
        served = args.served_model or args.model
        client = OpenAICompatibleClient(served, args.openai_url, api_key_env=args.api_key_env, timeout=7200)
        try:
            backend.update(client.describe())
        except (ConnectionError, RuntimeError) as e:
            raise SystemExit(f"{e}. Start the server first (e.g. vllm serve ... --max-model-len {args.num_ctx}).")
        ids = {m.get("id"): m for m in backend.get("served") or []}
        if backend.get("served") is not None and served not in ids:
            raise SystemExit(f"the server at {args.openai_url} serves {sorted(ids)}, not {served}: "
                             "--served-model must name one of them")
        length = (ids.get(served) or {}).get("max_model_len")
        if length and int(length) != args.num_ctx:
            raise SystemExit(f"the server holds {length} tokens per sequence, --num-ctx says {args.num_ctx}: make "
                             "them the same on every machine (vllm serve --max-model-len)")
        backend.update(model=served, num_ctx=args.num_ctx, url=args.openai_url, weights_revision=args.weights_revision,
                       note=args.note)
        print(f"{_engine({'backend': backend})} at {args.openai_url}, context {args.num_ctx} tokens per sequence",
              flush=True)
        tokenizer = load_tokenizer(args.tokenizer or args.model)

        def pipeline(*a, **kw):
            return Capped(OpenAIPipe(client, tokenizer, seed_of_call, args.num_ctx))
    else:
        def pipeline(*a, **kw):
            kw["trust_remote_code"] = False  # Qwen3 is built into transformers; no model-supplied code
            kw.setdefault("torch_dtype", dtype)
            return Capped(original(*a, **kw))
    isg_llm.pipeline = functools.wraps(original)(pipeline)
    cuda = torch.cuda.is_available() and args.backend == "transformers"
    device_map = args.device_map or ("cuda:0" if cuda else "cpu")
    free = None
    if cuda:
        free, total = torch.cuda.mem_get_info()
        print(f"{torch.cuda.get_device_name(0)}: {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB free before loading",
              flush=True)
    sim = Simulator(story_config=StoryConfig(taxonomy=load_taxonomy(os.path.join(isg, "taxonomy"))),
                    fs_config=FileSystemConfig(), file_config=FileConfig(), device_map=device_map,
                    model_name=args.model)
    placement = model_placement(sim)
    if placement:
        print("model placed on: " + ", ".join(f"{d} ({n} modules)" for d, n in placement.items()), flush=True)
    elsewhere = sorted(d for d in placement if not d.startswith("cuda"))
    if cuda and elsewhere and not args.allow_cpu:
        raise SystemExit(f"part of the model is on {', '.join(elsewhere)}, so it would write at CPU speed. Close "
                         "programs that use GPU memory and run again with --device-map cuda:0 (the default), or "
                         "pass --allow-cpu to run anyway.")
    machine = machine_label(args.machine, torch)
    device = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
              else "cpu" if args.backend == "transformers" else machine)
    if args.backend == "ollama":
        dtype_used = backend.get("quantization") or "as Ollama serves it"
        engine = f"{_engine({'backend': backend})}, context {args.num_ctx}"
    elif args.backend == "openai":
        dtype_used = "as the server loads it (see backend.note)"
        engine = f"{_engine({'backend': backend})}, context {args.num_ctx}"
    else:
        dtype_used = args.dtype
        engine = f"transformers, {args.model} in {args.dtype}, device map {device_map}"
    name = "regeneration_manifest" + (f".shard-{shard}-of-{n_shards}" if n_shards > 1 else "") + ".json"
    manifest_path = os.path.join(args.fs_dir, name)
    manifest = {"model": args.model, "dtype": dtype_used, "device": device, "machine": machine,
                "isg": _provenance(isg), "torch": torch.__version__, "max_new_tokens": args.max_new_tokens or 300000,
                "backend": backend, "device_map": device_map, "placement": placement, "shard": [shard, n_shards],
                "limit": args.limit or None,
                "gpu_free_gb_before_loading": round(free / 2 ** 30, 2) if free is not None else None,
                "seconds": {}, "failed": {}, "model_calls": {}, "where": {}, "attempts": {}, "attempt_calls": {},
                "max_attempts": args.max_attempts}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest.update({k: v for k, v in json.load(f).items()
                             if k in ("seconds", "failed", "model_calls", "where", "attempts", "attempt_calls")})
    for s in manifest["failed"]:  # manifests before attempts were counted: one failed attempt each
        manifest["attempts"].setdefault(s, 1)
    given_up = [s for s in todo if manifest["attempts"].get(str(s), 0) >= args.max_attempts]
    if given_up:
        print(f"{len(given_up)} failed {args.max_attempts} times and are not tried again: {given_up}", flush=True)
        todo = [s for s in todo if s not in given_up]
    t_start = time.time()
    for n, seed in enumerate(todo, 1):
        if args.stop_after and time.time() - t_begin > args.stop_after * 3600:
            print(f"stopping: {args.stop_after} h have passed (--stop-after); {len(todo) - n + 1} repositories "
                  "left for the next run", flush=True)
            break
        t0 = time.time()
        calls.clear()
        calls.update(fresh)
        current["seed"] = seed
        current["attempt"] = manifest["attempts"].get(str(seed), 0)
        again = f" (attempt {current['attempt'] + 1}, new sampling seeds)" if current["attempt"] else ""
        print(f"[{n}/{len(todo)}] seed {seed}{again}", flush=True)
        try:
            fs = sim.get_filesystem(seed)
            final = os.path.join(args.fs_dir, f"{seed}.json")
            with open(final + ".tmp", "w", encoding="utf-8") as f:  # complete or absent, even after Ctrl+C
                json.dump(fs.to_json(), f, indent=2)
            os.replace(final + ".tmp", final)
            manifest["seconds"][str(seed)] = round(time.time() - t0, 1)
            manifest["where"][str(seed)] = f"{engine} on {machine}"
            manifest["failed"].pop(str(seed), None)
        except Exception as e:  # ISG marks such seeds corrupted; the item will be missing
            manifest["failed"][str(seed)] = f"{type(e).__name__}: {e}"[:300]
        record = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in calls.items()}
        manifest["model_calls"][str(seed)] = record
        manifest["attempts"][str(seed)] = current["attempt"] + 1
        manifest["attempt_calls"].setdefault(str(seed), []).append(
            {**record, "ok": str(seed) in manifest["seconds"] and str(seed) not in manifest["failed"],
             "seconds_total": round(time.time() - t0, 1)})
        manifest["n_done"] = len(manifest["seconds"])
        manifest["n_failed"] = len(manifest["failed"])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        rate = (time.time() - t_start) / n
        largest = f", largest context {calls['max_context']}" if calls["max_context"] else ""
        print(f"[{n}/{len(todo)}] seed {seed} {'done' if str(seed) in manifest['seconds'] else 'FAILED'} in "
              f"{time.time() - t0:.0f} s ({calls['n']} model calls, {calls['cut']} cut{largest}); about "
              f"{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)
    if run_context["max"]:
        print(f"largest context in this run: {run_context['max']} of {args.num_ctx} tokens (largest prompt "
              f"{run_context['prompt']}); a smaller --num-ctx changes no reply while it is at least the largest "
              f"prompt + {args.max_new_tokens + 16}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="baselines and checks on the regenerated items (or seeds.txt alone)")
    a.add_argument("--isg", help="the InfiniteScienceGym repository (for seeds.txt and the commit)")
    a.add_argument("--seeds", help="seeds.txt, if not in --isg")
    a.add_argument("--qa", help="the regenerated question files (data/qa from 2_generate_questions.py)")
    a.add_argument("--compare-qa", help="a second regeneration, for a determinism check")
    a.add_argument("--regeneration-manifest",
                   help="a manifest written by 'generate', or a directory whose regeneration_manifest*.json are "
                        "pooled (default: the fs directory next to --qa)")
    a.add_argument("--complete-prefix", action="store_true",
                   help="analyse only the items of the longest run of seeds (ascending) that were all attempted: "
                        "a time-limited generation then gives a sample fixed by seed order, not by speed")
    a.add_argument("--permutations", type=int, default=200)
    a.add_argument("--repeats", type=int, default=20, help="random halves for the blind policies")
    a.add_argument("--out", default=os.path.join("experiments_output", "isg"))
    gsub = sub.add_parser("generate", help="regenerate the repositories the 500 items use (GPU)")
    gsub.add_argument("--isg", required=True)
    gsub.add_argument("--seeds")
    gsub.add_argument("--fs-dir", required=True, help="where <seed>.json go (ISG's data/fs)")
    gsub.add_argument("--model", default=MODEL)
    gsub.add_argument("--tokenizer", help="where to load the model's tokenizer (chat template) from, e.g. a local copy "
                                          "of the weights (default: --model)")
    gsub.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                      help="bfloat16 is what transformers 4.57's default ('auto') picks for Qwen3-4B")
    gsub.add_argument("--max-new-tokens", type=int, default=8192,
                      help="ceiling per model call (ISG asks for 300000; 0 keeps that)")
    gsub.add_argument("--device-map", default=None,
                      help="default: cuda:0 when a GPU is present (all of the model on it), else cpu; 'auto' lets "
                           "transformers move layers that do not fit to the CPU, as ISG does")
    gsub.add_argument("--allow-cpu", action="store_true", help="run even if part of the model is on the CPU")
    gsub.add_argument("--backend", default="transformers", choices=["transformers", "ollama", "openai"],
                      help="who writes: ISG's transformers pipeline, a local Ollama server (much faster on Windows), "
                           "or an OpenAI-compatible server such as vLLM (many repositories at once)")
    gsub.add_argument("--ollama-model", default=OLLAMA_MODEL)
    gsub.add_argument("--ollama-url", default="http://localhost:11434")
    gsub.add_argument("--openai-url", default="http://127.0.0.1:8000/v1", help="the server, up to /v1")
    gsub.add_argument("--served-model", help="the name the server serves the model under (default: --model)")
    gsub.add_argument("--weights-revision", help="recorded: the Hugging Face revision of the weights it serves")
    gsub.add_argument("--note", help="recorded: how the server was started (e.g. its command line)")
    gsub.add_argument("--api-key-env", help="environment variable holding a key, if the server wants one")
    gsub.add_argument("--num-ctx", type=int, default=16384,
                      help="context (prompt + reply) in tokens: per Ollama slot, or vLLM's --max-model-len; "
                           "the same on every process")
    gsub.add_argument("--ollama-digest", help="refuse unless Ollama's model digest starts with this (all machines "
                                              "write with the same weights)")
    gsub.add_argument("--limit", type=int, default=0, help="only the first N repositories (to time a run)")
    gsub.add_argument("--shard", type=_shard, default=(1, 1), metavar="I/N",
                      help="build shard I of N (seeds dealt round-robin after --limit); keep N the same everywhere")
    gsub.add_argument("--max-attempts", type=int, default=3,
                      help="a seed that failed this often is not tried again; each retry samples with new seeds")
    gsub.add_argument("--stop-after", type=float, default=0, metavar="HOURS",
                      help="start no repository after this many hours (a Kaggle session ends at 12)")
    gsub.add_argument("--machine", help="recorded per repository (default: the GPUs nvidia-smi lists)")
    args = p.parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    if args.qa and not args.regeneration_manifest:  # ISG's layout: data/qa next to data/fs
        guess = os.path.join(os.path.dirname(os.path.abspath(args.qa)), "fs")
        args.regeneration_manifest = guess if os.path.isdir(guess) else None
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
