# Example 12: language models as attacker, question writer and agent

One logged client, three roles. Every model call is appended to `logs/<role>.jsonl` with the role,
model name, settings, prompt and output, so results can be cited and replayed.

```bash
# offline, with a scripted stand-in (not a language model: shows the pipeline only)
python -m examples.example_12_model_roles.run_experiment

# with your model: a function ask(prompt) -> text, and the model name to record
python -m examples.example_12_model_roles.run_experiment --model mymodels:ask --model-name provider/model-2026-05
```

A minimal `mymodels.py` for the Anthropic SDK (any provider works the same way):

```python
import anthropic
_client = anthropic.Anthropic()

def ask(prompt):
    msg = _client.messages.create(model="<model id>", max_tokens=2000, temperature=0,
                                  messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text
```

## Steps

1. **Release.** A law-discovery suite (18 runs) is exported to Hugging Face format and passes the
   leak audit.
2. **Attacker** (`simulator.ai.llm_attack`). The model gets the release with the evidence removed:
   no figures, no measurement columns, no data tables. It is asked to answer every question. Per
   question template, its accuracy is tested against a null rate: the best of the blind baseline
   (majority answer or majority position), chance, and a floor of 0.5/n. The test is an exact
   binomial, Bonferroni-corrected over templates. The verdict is one of three:
   - PASS;
   - FLAGGED: answerable without the evidence. This is a leak or a legitimate route, such as a
     caption that states the plotted range or a strong prior, so inspect the answers;
   - INCONCLUSIVE: fewer than 90 % of the questions were answered, or the view had to be truncated.
3. **Writer** (`simulator.ai.paraphrase_tasks`). The model rephrases each question. A rewrite is
   kept only if it:
   - has the same numbers;
   - names every answer option;
   - introduces no answer from the dataset vocabulary, such as a law family or the formula;
   - adds no negation, number words, emphasis or extra lines, and keeps the options in order.

   These checks catch common leaks but not synonyms or subtle changes of meaning, so read a sample
   of the accepted rewrites.

   The rewritten public set is then leak-audited. A nearest-neighbour test checks whether question
   lengths predict the answers, which would be a wording cue the writer added.
4. **Agent** (`simulator.ai.LLMAgent`). The model plays the interactive environment from example
   10, and each episode is diagnosed (`simulator.diagnostics.episodes`).
5. **Provenance and replay.** `llm_provenance.json` summarises the calls per role and model, with
   SHA-256 hashes of the logs and no prompt text. The agent episodes are replayed from the log
   without the model and must come out identical. The report lists the roles each model played;
   anything evaluated on the rewritten questions should use a model other than the writer.

## Rules the roles follow

- Answers always come from the simulator; a model never supplies ground truth.
- Model-written text goes through the same leak audit as everything else.
- The model that writes questions should not be the one evaluated on them.
