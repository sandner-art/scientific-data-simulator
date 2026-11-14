# Example 11: the simulator as a testbed

The simulator knows every hidden value and can regenerate any run, so it can test benchmarks
and agents, not only feed them. This example runs the law-discovery suite of example 10 through
the diagnostics in `scidatasim.diagnostics` and writes `report.md` and `report.json`.

```bash
python -m examples.example_11_testbed.run_experiment            # about 3 minutes
python -m examples.example_11_testbed.run_experiment --quick    # about 1 minute
```

## What it measures

1. **Quality card and shortcuts.** Answer balance, spread, duplicates and correct-position bias
   per question template, and two kinds of "answers without the data":
   - a blind baseline, which answers each question from the answers to the other questions;
   - a nearest-neighbour predictor on public cues (figure file size, caption length), with a
     permutation test.
2. **Scorer loophole.** A polynomial fitted to each run's observations is submitted as the
   formula. Under the old rule (1 % RMS error on the data range) it passed 0.19 of the formula
   questions in the full run; under the current rule (the data range and half of it beyond, with
   a tolerance set by what the true law reaches under the run's noise) it passed none, while the
   curve-fitting baseline went from 0.86 to 0.92. On the data range alone a smooth curve can be
   matched without finding the law; power laws stay partly within a polynomial's reach, because
   their exponents are often close to 1 or 2. Those questions carry the private label
   `polynomial_reachable`, and the section gives the law fitter's accuracy without them.
3. **Leak audit** of the Hugging Face release, with a canary string added.
4. **Contamination.** A fresh twin of the suite has the same design and new seeds, so its hidden
   laws have never been released. A solver that has memorised part of a leaked answer key is
   scored on the original and on the twin, and a paired permutation test compares the two.
5. **Canary probe.** A model that has seen the release and one that has not are both asked to
   complete the canary.
6. **Agent fallacies.** 24 episodes per agent, counterfactual laws, 2 % tolerance. Agents may
   measure on [0.5, 5]; formulas are scored on [0.5, 7.25].

| agent | accuracy | flags raised (rate) |
|---|---|---|
| fits over the whole measurable range | 1.00 | none |
| snaps to the textbook constant | 0.42 | matches_textbook 1.00 (11 applicable), contradicts_own_data 0.33, premature_commitment 0.58 |
| answers without experiments | 0.00 | answer_without_evidence 1.00, premature_commitment 1.00 |
| measures one fifth of the measurable range | 0.54 | narrow_sampling 1.00, premature_commitment 0.46 |
| repeats one experiment three times | 1.00 | repeated_experiment 1.00 |
| ignores its measurement | 0.00 | contradicts_own_data 1.00, premature_commitment 1.00 |

`matches_textbook` applies only where the textbook form of the law differs from the hidden law by
more than the tolerance on the scoring range: 11 of the 24 episodes. The textbook form is fitted
over the range the agent measured, as a solver recalling it would fit it. In the others, either the
family has no textbook constant or the change does not show on that range. Rates are over the
applicable episodes. `contradicts_own_data` is raised only for wrong answers, and its noise
floor is the tolerance, so near-misses on clean data are not flagged. A repeated experiment can
be deliberate averaging, so that flag is informational.

Accuracy alone cannot tell these agents apart. The narrow sampler is often right by luck, and
the repeater scores as well as the good fitter. The process flags separate all six. Every
wrong answer of the textbook agent that can be checked matches the textbook form rather than the
counterfactual law.

Numbers vary with `--suite-seed` and the episode seeds; the pattern was the same in the runs made
so far.
