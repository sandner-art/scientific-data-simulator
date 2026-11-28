"""Writes kaggle/isg_prepare.ipynb and kaggle/isg_generate.ipynb: python tools/make_kaggle_notebooks.py kaggle"""
import json
import os
import sys

OUT = sys.argv[1]


def nb(cells, gpu):
    kaggle = {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isGpuEnabled": gpu, "isInternetEnabled": True,
              "language": "python", "sourceType": "notebook", "dataSources": []}
    def cell(k, t, src):
        extra = {"outputs": [], "execution_count": None} if t == "code" else {}
        return {"cell_type": t, "id": f"cell-{k}", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True),
                **extra}
    return {"cells": [cell(k, t, src) for k, (t, src) in enumerate(cells)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "kaggle": kaggle},
            "nbformat": 4, "nbformat_minor": 5}


COMMON = r'''
import glob, json, os, shutil, subprocess, sys, time

def sh(cmd, **kw):
    """Run a shell command, show it, stop the notebook if it fails."""
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)

def isg_venv(packages, torch_cpu, path="/tmp/isg_venv"):
    """ISG's own versions in their own environment (ISG runs model-written code)."""
    py = f"{path}/bin/python"
    if not os.path.exists(py):
        sh(f"{sys.executable} -m pip install -q uv")
        sh(f"{sys.executable} -m uv venv -q --python 3.12 {path}")
        sh(f"{sys.executable} -m uv pip install -q --python {py} {torch_cpu} --index-url https://download.pytorch.org/whl/cpu")
        sh(f"{sys.executable} -m uv pip install -q --python {py} " + " ".join(f"'{p}'" for p in packages))
    return py

def entries(path):
    with open(path) as f:
        m = json.load(f)
    return len(m.get("seconds", {})) + len(m.get("failed", {}))

def gather(sources, fs, manifests=lambda name: True):
    """Finished repositories (<seed>.json) and manifests from attached outputs into fs;
    of two copies of a manifest, the one with more entries wins."""
    os.makedirs(fs, exist_ok=True)
    for path in sorted(sources):
        name, dest = os.path.basename(path), os.path.join(fs, os.path.basename(path))
        if name.startswith("regeneration_manifest") and manifests(name):
            if not os.path.exists(dest) or entries(path) > entries(dest):
                shutil.copy(path, dest)
        elif name.split(".")[0].isdigit() and name.endswith(".json") and not os.path.exists(dest):
            shutil.copy(path, dest)
    return len(glob.glob(os.path.join(fs, "[0-9]*.json")))
'''

PREPARE = [
("markdown", r'''
# InfiniteScienceGym regeneration, part 1 of 2: prepare (and finish) on CPU

Runs on a **CPU** session (no GPU quota), with *Internet* on. Keep this notebook and its output
**private**: once repositories and questions are in it, it holds answers.

* `MODE = "prepare"`: pinned code (ISG and scidatasim), the model's weights at a recorded revision,
  a check that ISG's code, the tokenizer and the seed list work, and `plan.json` for the GPU
  notebook (`isg_generate.ipynb`). Save the version; add its output as an input of the GPU notebook.
* `MODE = "finish"`: add the GPU notebook's output(s) as inputs. Generates the questions (ISG's
  `2_generate_questions.py`, CPU) and runs the audit on the longest complete prefix of seeds
  (`audit_isg.py audit --complete-prefix`). Results in `results/`.

Steps and background: `docs/reproduce/isg_regeneration.md` in scidatasim.
'''),
("code", r'''
MODE = "prepare"          # "prepare" or "finish"
SDS_COMMIT = "main"       # scidatasim; pin the commit with patch 0064 for the paper
ISG_COMMIT = "2006f69872986e502074282ee4d23634e0cab689"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = ""       # empty: today's revision (recorded in plan.json)
VLLM_VERSION = ""         # empty: the newest (recorded in plan.json); the GPU notebook installs it
WEIGHTS_IN_OUTPUT = False # True: also keep the 8 GB of weights in this output (a fallback for the GPU notebook)
ISG_PACKAGES = ["numpy==2.4.0", "pandas==2.3.3", "scipy==1.17.0", "transformers==4.57.3", "XlsxWriter==3.2.9",
                "huggingface_hub<1.0"]
TORCH_CPU = "torch==2.9.1"
OUT = "/kaggle/working"
'''),
("code", COMMON),
("code", r'''
code = f"{OUT}/code"
for name, url, commit in [("isg", "https://github.com/utahnlp/infinite-science-gym", ISG_COMMIT),
                          ("sds", "https://github.com/sandner-art/scientific-data-simulator", SDS_COMMIT)]:
    if not os.path.isdir(f"{code}/{name}"):
        sh(f"git clone -q {url} {code}/{name} && git -C {code}/{name} checkout -q {commit}")
commits = {n: subprocess.run(["git", "-C", f"{code}/{n}", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip() for n in ("isg", "sds")}
print(commits)
py = isg_venv(ISG_PACKAGES, TORCH_CPU)
'''),
("code", r'''
if MODE == "prepare":
    from huggingface_hub import HfApi, snapshot_download
    revision = HfApi().model_info(MODEL, revision=MODEL_REVISION or None).sha
    # the tokenizer and configs for the checks; the GPU notebook downloads the weights at this revision
    # itself (about a minute, against 15 minutes reading 8 GB from an input). WEIGHTS_IN_OUTPUT = True
    # also stores the weights here, as a fallback should the Hub be unreachable.
    snapshot_download(MODEL, revision=revision, local_dir=f"{OUT}/model",
                      allow_patterns=None if WEIGHTS_IN_OUTPUT else ["*.json", "*.txt"])
    if not VLLM_VERSION:  # the newest release, so every session installs the same one
        found = subprocess.run(f"{sys.executable} -m pip index versions vllm", shell=True, capture_output=True,
                               text=True).stdout
        VLLM_VERSION = found.split("(")[1].split(")")[0] if "(" in found else ""
    print("weights", MODEL, revision, "| vLLM", VLLM_VERSION or "(newest at install)")
'''),
("code", r'''
if MODE == "prepare":  # the checks that would otherwise cost GPU time
    tools = f"{code}/sds/tools/audit_isg.py"
    check = (f"import sys; sys.path[:0] = ['{code}/sds/tools', '{code}/isg']; import audit_isg as a; "
             f"import simulator; tok = a.load_tokenizer('{OUT}/model'); "
             "p = tok.apply_chat_template([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'PRE:'}], "
             "tokenize=False, add_generation_prompt=False, continue_final_message=True); "
             "assert p.rstrip().endswith('PRE:'), p; print('ISG imports, tokenizer and template: fine')")
    sh(f"cd /tmp && {py} -c \"{check}\"")
    sh(f"{py} {tools} audit --isg {code}/isg --out {OUT}/results/seeds_only")   # seeds.txt check, no data
    keys = open(f"{code}/isg/seeds.txt").read().split()
    seeds = sorted({int(k.split(".")[0]) for k in keys})
    plan = {"model": MODEL, "revision": revision, "vllm": VLLM_VERSION, "commits": commits,
            "isg_packages": ISG_PACKAGES, "torch_cpu": TORCH_CPU, "repositories": len(seeds), "items": len(keys),
            "prepared": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    json.dump(plan, open(f"{OUT}/plan.json", "w"), indent=1)
    print(json.dumps(plan, indent=1))
'''),
("code", r'''
if MODE == "finish":
    data = f"{OUT}/isgdata"
    n = gather(glob.glob("/kaggle/input/**/fs/*.json", recursive=True), f"{data}/fs")
    print(n, "repositories from the attached generation outputs")
    sh(f"cd {code}/isg && {py} scripts/2_generate_questions.py --cache-dir {data} --n-seeded-samples 5 "
       f"> {OUT}/questions.log 2>&1")
    sh(f"{py} {code}/sds/tools/audit_isg.py audit --isg {code}/isg --qa {data}/qa --complete-prefix "
       f"--out {OUT}/results")
    print(open(f"{OUT}/results/isg_audit.md").read())
'''),
]

GENERATE = [
("markdown", r'''
# InfiniteScienceGym regeneration, part 2 of 2: generate on two T4 GPUs

Settings: Accelerator **GPU T4 x2**, *Internet* on. Inputs: the output of `isg_prepare.ipynb`
(`MODE = "prepare"`), and from the second session on, this notebook's previous output. Keep it
**private**: the repositories hold the answers.

Run with *Save Version*, *Save & Run All (Commit)*; the browser need not stay open.

* One vLLM server per GPU writes for `PER_GPU` processes at once (it batches their calls); each
  process builds one shard of the seeds (`--shard I/N`, N = 2 x `PER_GPU`; keep `PER_GPU` the same
  for the whole job).
* No repository is started after `HOURS` hours; those in progress finish. A session ends at 12 hours,
  so leave room for the slowest repository: `HOURS = 1` for a first, measuring session, `9.5` after.
* The progress lines give repositories done, tokens per second and CPU load; the last cell writes
  the final summary, `report.md` (from `tools/isg_run_report.py`), to the output. Questions and the
  audit run on CPU: `isg_prepare.ipynb` with `MODE = "finish"`.
* `ENGINE = "ollama"` is the fallback if vLLM does not start on the T4 (slower: two slots per GPU).
  The engine is recorded per repository; do not mix engines in one sample without saying so.
'''),
("code", r'''
ENGINE = "vllm"          # "vllm", or "ollama" if vLLM fails on the T4
HOURS = 1.0              # start no repository after this; 9.5 for a full session
PER_GPU = 8              # processes per GPU; N = 2 * PER_GPU shards, the same for the whole job
NUM_CTX = 16384          # tokens per sequence (vLLM --max-model-len, Ollama num_ctx), the same everywhere
VLLM_VERSION = ""        # empty: the one in plan.json
GPU_MEMORY = 0.90        # vLLM's share of each T4
KV_CACHE_GIB = None      # e.g. 5.0: vLLM's KV cache in GiB instead of GPU_MEMORY (its log suggests a value)
MAX_ATTEMPTS = 3         # a seed that failed this often is not tried again; retries sample with new seeds
OLLAMA_VERSION, OLLAMA_MODEL, OLLAMA_DIGEST = "0.40.13", "qwen3:4b-instruct-2507-fp16", "259cbeb0f33b"
OUT = "/kaggle/working"
'''),
("code", COMMON),
("code", r'''
plans = glob.glob("/kaggle/input/**/plan.json", recursive=True)
assert plans, "add the output of isg_prepare.ipynb (MODE = 'prepare') as an input"
prep = os.path.dirname(plans[0])
plan = json.load(open(plans[0]))
VLLM_VERSION = VLLM_VERSION or plan["vllm"]
N = 2 * PER_GPU
for path in glob.glob("/kaggle/input/**/run.json", recursive=True):  # an earlier session of this job
    assert json.load(open(path))["shards"] == N, f"{path} used another number of shards: keep PER_GPU as it was"
fs = f"{OUT}/fs"
os.makedirs(f"{OUT}/logs", exist_ok=True)
mine = {f"regeneration_manifest.shard-{i}-of-{N}.json" for i in range(1, N + 1)}
done = gather([p for p in glob.glob("/kaggle/input/**/fs/*.json", recursive=True)], fs, lambda name: name in mine)
json.dump({"shards": N, "engine": ENGINE, "plan": plan}, open(f"{OUT}/run.json", "w"), indent=1)
print(f"{done} of {plan['repositories']} repositories already built; {N} shards")
shutil.copytree(f"{prep}/code", "/tmp/code", dirs_exist_ok=True)   # ISG may write next to its code
py = isg_venv(plan["isg_packages"], plan["torch_cpu"])
'''),
("code", r'''
import shlex
import urllib.request
servers, urls, logs = [], [], []
model_dir = "/tmp/model"                 # local disk: loading from an input took 15 minutes
if not os.path.exists(f"{model_dir}/config.json") or not glob.glob(f"{model_dir}/*.safetensors"):
    t = time.time()
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(plan["model"], revision=plan["revision"], local_dir=model_dir)
    except Exception as e:                # the Hub unreachable: the weights kept by the prepare notebook, if any
        print("download failed:", e)
        assert glob.glob(f"{prep}/model/*.safetensors"), "no weights: rerun prepare with WEIGHTS_IN_OUTPUT = True"
        shutil.copytree(f"{prep}/model", model_dir, dirs_exist_ok=True)
    print(f"weights {plan['model']} at {plan['revision'][:12]} in {model_dir} ({time.time() - t:.0f} s)")
if ENGINE == "vllm":
    if not os.path.exists("/tmp/vllm_venv/bin/vllm"):
        sh(f"{sys.executable} -m uv venv -q --python 3.12 /tmp/vllm_venv")
        sh(f"{sys.executable} -m uv pip install -q --python /tmp/vllm_venv/bin/python 'vllm=={VLLM_VERSION}'"
           if VLLM_VERSION else f"{sys.executable} -m uv pip install -q --python /tmp/vllm_venv/bin/python vllm")
    version = subprocess.run(["/tmp/vllm_venv/bin/python", "-c", "import vllm; print(vllm.__version__)"],
                             capture_output=True, text=True).stdout.strip()
    memory = (f"--kv-cache-memory {int(KV_CACHE_GIB * 2 ** 30)}" if KV_CACHE_GIB
              else f"--gpu-memory-utilization {GPU_MEMORY}")
    serve = (f"vllm serve {model_dir} --served-model-name {plan['model']} --dtype half --max-model-len {NUM_CTX} "
             f"{memory} --host 127.0.0.1")
    for gpu in (0, 1):
        logs.append(f"{OUT}/logs/vllm-gpu{gpu}.log")
        servers.append(subprocess.Popen(shlex.split(f"/tmp/vllm_venv/bin/{serve} --port {8000 + gpu}"),
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                        stdout=open(logs[-1], "w"), stderr=subprocess.STDOUT))
        urls.append(f"http://127.0.0.1:{8000 + gpu}/v1")
    note = f"vLLM {version}: {serve} (one server per T4)"
else:
    sh(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh > {OUT}/logs/ollama_install.log 2>&1")
    for gpu in (0, 1):
        logs.append(f"{OUT}/logs/ollama-gpu{gpu}.log")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OLLAMA_HOST": f"127.0.0.1:{11434 + gpu}",
               "OLLAMA_NUM_PARALLEL": "2"}
        servers.append(subprocess.Popen(["ollama", "serve"], env=env, stdout=open(logs[-1], "w"),
                                        stderr=subprocess.STDOUT))
        urls.append(f"http://127.0.0.1:{11434 + gpu}")
    time.sleep(15)
    sh(f"OLLAMA_HOST=127.0.0.1:11434 ollama pull {OLLAMA_MODEL} > {OUT}/logs/pull.log 2>&1")
    note = f"Ollama {OLLAMA_VERSION}, OLLAMA_NUM_PARALLEL=2 (one server per T4)"

probe = [u + "/models" if ENGINE == "vllm" else u + "/api/version" for u in urls]
for url, server, log in zip(probe, servers, logs):
    for _ in range(180):                     # loading the model and capturing CUDA graphs take minutes
        try:
            urllib.request.urlopen(url, timeout=10)
            break
        except Exception:
            if server.poll() is not None:
                raise SystemExit(f"the server stopped. The end of {log}:\n" + open(log).read()[-3000:])
            time.sleep(10)
    else:
        raise SystemExit(f"no answer from {url} after 30 minutes; see {log}")
print("servers up:", urls, "|", note)
'''),
("code", r'''
tool = "/tmp/code/sds/tools/audit_isg.py"
procs = []
for gpu, url in enumerate(urls):
    for k in range(PER_GPU):
        i = gpu * PER_GPU + k + 1
        engine = (["--backend", "openai", "--openai-url", url, "--weights-revision", plan["revision"]]
                  if ENGINE == "vllm" else ["--backend", "ollama", "--ollama-url", url, "--ollama-model", OLLAMA_MODEL,
                                            "--ollama-digest", OLLAMA_DIGEST])
        cmd = [py, "-I", "-X", "utf8", tool, "generate", "--isg", "/tmp/code/isg", "--fs-dir", fs, "--model",
               plan["model"], "--tokenizer", model_dir, *engine, "--num-ctx", str(NUM_CTX),
               "--shard", f"{i}/{N}", "--stop-after", str(HOURS), "--max-attempts", str(MAX_ATTEMPTS),
               "--machine", f"Kaggle, Tesla T4 (GPU {gpu})", "--note", note]
        procs.append(subprocess.Popen(cmd, stdout=open(f"{OUT}/logs/shard-{i}.log", "w"), stderr=subprocess.STDOUT))

import re
CALL = re.compile(r"model call \d+: (?:prompt \d+ \+ )?(\d+) tokens")

def written():                           # every reply so far, from the shard logs (also unfinished repositories)
    total = 0
    for log in glob.glob(f"{OUT}/logs/shard-*.log"):
        with open(log, errors="replace") as f:
            total += sum(int(m.group(1)) for m in CALL.finditer(f.read()))
    return total

print(f"{os.cpu_count()} CPUs for 2 servers and {len(procs)} processes")

t0, start, tokens0, last = time.time(), done, written(), 0.0
while any(p.poll() is None for p in procs):
    time.sleep(30)
    if time.time() - last < 600 and any(p.poll() is None for p in procs):
        continue
    last = time.time()
    now, hours = len(glob.glob(f"{fs}/[0-9]*.json")), (time.time() - t0) / 3600
    print(f"{hours:.1f} h: {now} of {plan['repositories']} repositories ({now - start} this session), "
          f"{sum(p.poll() is None for p in procs)} processes running, {(written() - tokens0) / max(hours * 3600, 1):.0f} "
          f"tokens/s written this session, CPU load {os.getloadavg()[0]:.1f}", flush=True)
for server in servers:
    server.terminate()
print("exit codes:", [p.returncode for p in procs], "(non-zero: see logs/shard-*.log)")
'''),
("code", r'''
# The final summary: report.md in the output (totals, throughput, every repository, failures, servers).
# The same report at home, from the downloaded output: python tools/isg_run_report.py <folder>
sh(f"{py} /tmp/code/sds/tools/isg_run_report.py {OUT} --out {OUT}/report.md")
text = open(f"{OUT}/report.md").read()
print(text.split("## Repositories")[0])
'''),
]

os.makedirs(OUT, exist_ok=True)
for name, cells, gpu in [("isg_prepare.ipynb", PREPARE, False), ("isg_generate.ipynb", GENERATE, True)]:
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(nb(cells, gpu), f, indent=1)
        f.write("\n")
