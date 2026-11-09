# Example 11: the simulator as a testbed

The simulator knows every hidden value and can regenerate any run, so it can test benchmarks
and agents, not only feed them. This example runs the law-discovery suite of example 10 through
the diagnostics in `simulator.diagnostics` and writes `report.md` and `report.json`.

```bash
python -m examples.example_11_testbed.run_experiment            # about 2.5 minutes
python -m examples.example_11_testbed.run_experiment --quick    # about 1 minute
```

## What it measures

1. **Quality card and shortcuts.** Answer balance, spread, duplicates and correct-position bias
   per question template, and two kinds of "answers without the data":
   - a blind baseline, which answers each question from the answers to the other questions;
   - a nearest-neighbour predictor on public cues (figure file size, caption length), with a
     permutation test.
2. **Leak audit** of the Hugging Face release, with a canary string added.
3. **Contamination.** A fresh twin of the suite has the same design and new seeds, so its hidden
   laws have never been released. A solver that has memorised part of a leaked answer key is
   scored on the original and on the twin, and a paired permutation test compares the two.
4. **Canary probe.** A model that has seen the release and one that has not are both asked to
   complete the canary.
5. **Agent fallacies.** 24 episodes per agent, counterfactual laws, 2 % tolerance.

| agent | accuracy | flags raised (rate) |
|---|---|---|
| fits over the whole domain | 1.00 | none |
| snaps to the textbook constant | 0.42 | matches_textbook 1.00 (11 applicable), contradicts_own_data 0.33, premature_commitment 0.58 |
| answers without experiments | 0.00 | answer_without_evidence 1.00, premature_commitment 1.00 |
| measures one fifth of the domain | 0.62 | narrow_sampling 1.00, premature_commitment 0.38 |
| repeats one experiment three times | 1.00 | repeated_experiment 1.00 |
| ignores its measurement | 0.00 | contradicts_own_data 1.00, premature_commitment 1.00 |

`matches_textbook` applies only where the textbook form of the law differs from the hidden law by
more than the tolerance on the scoring range: 11 of the 24 episodes. In the others, either the
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
