# Regenerating InfiniteScienceGym's items on more than one GPU

`tools/audit_isg.py generate` rebuilds the 287 data repositories that InfiniteScienceGym's 500
evaluated questions use (Bentham and Srikumar, arXiv:2604.13201). It uses ISG's own code and lets
a local Ollama server write with the paper's model. Run one repository at a time, the job is
limited by how fast one GPU writes a single stream. On an RTX A4000 that is about 24 tokens/s once
the card is warm, or about 26 minutes per repository and five days for all 287.

The fix is to generate several repositories at once:

- **Slots.** One Ollama server with several slots writes several replies in the same pass over the
  weights.
- **Shards.** Several processes split the seeds without overlap, on one machine or on several
  (here: your PC and a Kaggle notebook with two T4 GPUs).

This page gives the steps. The tool's docstring describes the method.

## What stays fixed, and what differs from the paper

| | |
|---|---|
| ISG code | `https://github.com/utahnlp/infinite-science-gym`, commit `2006f69872986e502074282ee4d23634e0cab689`; its `requirements.txt` "Simulator" versions (numpy 2.4.0, pandas 2.3.3, scipy 1.17.0, transformers 4.57.3, XlsxWriter 3.2.9, torch 2.9.1) |
| Model | Qwen3-4B-Instruct-2507 as Ollama's `qwen3:4b-instruct-2507-fp16` (F16; the paper ran bfloat16 on an H200). Every machine must serve the same build: `--ollama-digest 259cbeb0f33b` refuses any other |
| Ollama | the same version on every machine (`ollama --version`; 0.40.13 at home on 24 November 2026) |
| Prompts | rendered by the model's own chat template through its tokenizer, as ISG's pipeline renders them |
| Sampling | ISG's: temperature 0.7, top-p 0.8, top-k 20; a seed per repository and model call |
| Ceiling | 8,192 new tokens per model call (ISG allows 300,000); a cut reply fails ISG's parser and ISG asks again |
| Context | `--num-ctx 16384` per slot, the same everywhere (see below) |

**Variation between runs.** Parallel decoding gives each call the same seed as before. But
batching changes the order of floating-point sums, so a repository built in a batch can differ
from the same seed built alone. Other GPUs do the same. Either way the sample is a regeneration,
not the paper's items.

**What the manifests record.** For every repository: the machine, the engine and how it went (model
calls, tokens, calls cut, the largest prompt and context). The audit pools all the manifests and
lists them in its report.

## Size the run first

The three test repositories took 26 to 37 model calls and 29,000 to 46,000 tokens each, so all 287
come to about 10 million written tokens. How many slots fit is set by the context:

- **Memory per slot.** Ollama reserves `--num-ctx` tokens of context per slot. For Qwen3-4B in F16
  that is 144 KiB per token (36 layers x 8 key-value heads x 128 dimensions x key and value x 2
  bytes), or 2.25 GiB per slot at 16,384.
- **What fits on a 16 GB card.** The weights take about 7.5 GiB, so two slots fit at 16,384, three
  at about 12,288, and four at about 8,192. The tool stops if Ollama puts part of the model on the
  CPU (`--allow-cpu` overrides).
- **When a smaller context is safe.** Lowering `--num-ctx` changes no reply as long as it stays at
  least the largest prompt + 8,208 tokens. Below that, long prompts leave less room than the
  8,192-token ceiling.
- **Where to read the largest prompt.** Each model call's line now shows `prompt P + T tokens`.
  The end of a run prints the largest prompt and context, and the manifests keep them per
  repository.
- **What counts as cut.** A reply that stops because the room ran out is now counted as cut, like
  one that hits the ceiling (Ollama reports why it stopped). Before this version only the ceiling
  counted, so a cut caused by a long prompt went unnoticed.

Measure before committing days: run each machine for a short while (`--limit`, or `--stop-after
0.5`). Then compare tokens per second per stream with one stream, and read the largest prompt.

## Shards

`--shard I/N` deals the seeds round-robin into N shards (after `--limit`), and the process builds
shard I. Rules:

- **Keep N the same on every machine and for the whole job.** With `--limit`, shard I of the first
  100 seeds is the beginning of shard I of all 287, so removing `--limit` later is safe.
- **One process per slot.** Give every process its own I.
- **Manifests.** Each shard writes `regeneration_manifest.shard-I-of-N.json` next to the
  repositories. A run without shards writes `regeneration_manifest.json`, as before.
- **Interruptions.** A repository is written only when complete. Stopping and starting again loses
  at most the repositories in progress, and a failed seed is retried on the next run.
- **Moving a shard.** To move a shard to another machine, copy its finished repositories and its
  manifest there, then run the same `--shard`.

An example split, N = 6: shards 1-2 at home (one GPU, two slots), shards 3-6 on Kaggle (two GPUs,
two slots each).

## At home (Windows, one GPU)

Quit the Ollama tray app (it runs its own server), then start a server with two slots:

```bat
set OLLAMA_NUM_PARALLEL=2
ollama serve
```

In a second window, with `isg_venv` active:

```bat
set PYTHONUTF8=1
cd /d C:\Users\quant\Documents\GitHub\scientific-data-simulator
set GEN=python tools\audit_isg.py generate --isg C:\opendata\isg --fs-dir C:\opendata\isg\data\fs --backend ollama --ollama-digest 259cbeb0f33b --machine "home RTX A4000"
start "shard 1" cmd /k %GEN% --shard 1/6
start "shard 2" cmd /k %GEN% --shard 2/6
```

Each window prints its own progress. They write into the same folder without collisions.

## On Kaggle (two T4 GPUs)

Kaggle's free tier gives a verified account (phone number) GPU time each week, with sessions of up
to 12 hours. The GPU T4 x2 option has two 16 GB T4s. The T4 has no bfloat16, but the F16 model
needs none. Kaggle's own pages state your current quota.

Keep the notebook and every dataset **private**. The repositories contain the answers to the
questions, and posting them makes them available for training. That would undo the protection
against contamination that procedural generation exists to give.

### One-time setup

1. **Create the notebook.** New notebook; in Settings choose Accelerator *GPU T4 x2* and turn
   Internet on.
2. **Upload what is already built.** Upload the repositories built before sharding (at home: `1.json`,
   `2.json`, `4.json` from `C:\opendata\isg\data\fs`) as a private dataset, and add it as an input.
   With N = 6, seed 4 falls in shard 3, which Kaggle builds, so without the upload it would be built
   a second time.
3. **Paste the four cells below.**

### Every session

1. **Add the previous output.** From the second session on, add the previous version's output as
   an input (Add Input, Your Work, this notebook). The restore cell carries the finished
   repositories and Kaggle's own shard manifests forward, so each version's output holds
   everything so far.
2. **Run it.** *Save Version*, *Save & Run All (Commit)*. The session runs without the browser
   open.
3. **Mind the deadline.** `--stop-after 10` starts no repository after 10 hours, leaving room for
   the last ones to finish before the 12-hour limit. Kaggle may not keep the output of a run
   stopped at the limit.

**Cell 1, configuration (Python):**

```python
import os
os.environ.update(
    OLLAMA_VERSION="0.40.13",       # the same as at home: ollama --version
    SDS_COMMIT="<commit with patch 0063>",
    ISG_COMMIT="2006f69872986e502074282ee4d23634e0cab689",
    SHARDS="6", FIRST="3",          # this machine runs shards 3, 4, 5, 6 of 6
    SLOTS="2", HOURS="10", NUM_CTX="16384", DIGEST="259cbeb0f33b")
```

**Cell 2, software (bash):**

```bash
%%bash
set -euo pipefail
curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=$OLLAMA_VERSION sh > /tmp/ollama_install.log 2>&1
git clone -q https://github.com/utahnlp/infinite-science-gym /tmp/isg
git -C /tmp/isg checkout -q $ISG_COMMIT
git clone -q https://github.com/sandner-art/scientific-data-simulator /tmp/sds
git -C /tmp/sds checkout -q $SDS_COMMIT
pip install -q uv
uv venv -q --python 3.12 /tmp/isg_venv
uv pip install -q --python /tmp/isg_venv/bin/python torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q --python /tmp/isg_venv/bin/python numpy==2.4.0 pandas==2.3.3 scipy==1.17.0 \
    transformers==4.57.3 XlsxWriter==3.2.9
```

(torch is only imported here: Ollama does the writing, so the CPU build suffices.)

**Cell 3, restore finished work (Python):**

```python
import glob, json, os, shutil
fs = "/kaggle/working/fs"
os.makedirs(fs, exist_ok=True)
os.makedirs("/kaggle/working/logs", exist_ok=True)
first, n, slots = int(os.environ["FIRST"]), int(os.environ["SHARDS"]), int(os.environ["SLOTS"])
mine = {f"regeneration_manifest.shard-{i}-of-{n}.json" for i in range(first, first + 2 * slots)}

def entries(path):
    with open(path) as f:
        m = json.load(f)
    return len(m.get("seconds", {})) + len(m.get("failed", {}))

for path in glob.glob("/kaggle/input/**/*.json", recursive=True):
    name, dest = os.path.basename(path), os.path.join(fs, os.path.basename(path))
    if name in mine and (not os.path.exists(dest) or entries(path) > entries(dest)):
        shutil.copy(path, dest)                      # this machine's manifests, newest wins
    elif name.split(".")[0].isdigit() and not os.path.exists(dest):
        shutil.copy(path, dest)                      # finished repositories
print(len(glob.glob(fs + "/[0-9]*.json")), "repositories already built")
```

**Cell 4, generate (bash):**

```bash
%%bash
cd /kaggle/working
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES=$gpu OLLAMA_HOST=127.0.0.1:$((11434 + gpu)) OLLAMA_NUM_PARALLEL=$SLOTS \
    nohup ollama serve > logs/ollama-gpu$gpu.log 2>&1 &
done
sleep 15
OLLAMA_HOST=127.0.0.1:11434 ollama pull qwen3:4b-instruct-2507-fp16 > logs/pull.log 2>&1
i=$FIRST
for gpu in 0 1; do
  for slot in $(seq $SLOTS); do
    /tmp/isg_venv/bin/python -I -X utf8 /tmp/sds/tools/audit_isg.py generate --isg /tmp/isg --fs-dir fs \
      --backend ollama --ollama-url http://127.0.0.1:$((11434 + gpu)) --ollama-digest $DIGEST \
      --num-ctx $NUM_CTX --shard $i/$SHARDS --stop-after $HOURS --machine "Kaggle, Tesla T4 (GPU $gpu)" \
      > logs/shard-$i.log 2>&1 &
    i=$((i + 1))
  done
done
wait
tail -n 3 logs/shard-*.log
/tmp/isg_venv/bin/python -c "import sys, json; sys.path.insert(0, '/tmp/sds/tools'); import audit_isg as a; \
print(json.dumps(a.merge_manifests(a.manifest_paths('fs')), indent=1))"
```

Both servers read the same model store, so one pull serves both. If the digest check fails, the
registry now serves another build under the same name: stop, and pull again on every machine
before mixing builds.

The Kaggle steps were written against Kaggle's documented setup and Ollama's install script, but
they have not yet been run on Kaggle. Make the first session a short one (`HOURS="0.5"`), and
check the logs before a long one.

## Bringing the parts together

1. **Download Kaggle's output.** It holds the `fs` and `logs` folders.
2. **Copy everything from `fs` into `C:\opendata\isg\data\fs`.** Kaggle's manifest names
   (`shard-3` to `shard-6`) never exist at home. The few home repositories it carried along are
   identical copies.
3. **Then, at home, as before:**

```bat
cd /d C:\opendata\isg
python scripts\2_generate_questions.py --cache-dir .\data --n-seeded-samples 5
cd /d C:\Users\quant\Documents\GitHub\scientific-data-simulator
python tools\audit_isg.py audit --isg C:\opendata\isg --qa C:\opendata\isg\data\qa --out C:\opendata\isg_out
```

The audit pools every `regeneration_manifest*.json` in `data\fs`. Its report lists the
repositories built, failed and cut, and which machine and engine built how many.
