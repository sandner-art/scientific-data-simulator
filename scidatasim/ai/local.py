# scidatasim/ai/local.py
"""Models on your own GPU, or on any OpenAI-compatible server, and work split over processes.

The clients are ``prompt -> text`` functions, so they plug into everything in
:mod:`scidatasim.ai` (wrap them in :class:`~scidatasim.ai.clients.LoggedClient` to keep the
record) and into :class:`scidatasim.llm.CallableLLMClient`. They use the standard library only::

    from scidatasim.ai import LLMAgent, LoggedClient
    from scidatasim.ai.local import OllamaClient
    from scidatasim.diagnostics import run_episodes

    model = OllamaClient("qwen3:4b-instruct-2507-fp16", temperature=0.7, seed=0)
    agent = LLMAgent(LoggedClient(model, "runs/agent.jsonl", role="agent", model=model.model_id,
                                  settings=model.settings))
    logs = run_episodes(env, agent, seeds=range(200), out_dir="runs/episodes", shard="1/4")

* :class:`OllamaClient` talks to a local `Ollama <https://ollama.com>`_ server (``/api/chat``).
* :class:`OpenAICompatibleClient` talks to anything that serves the OpenAI chat API: vLLM,
  llama.cpp's server, LM Studio, Ollama's ``/v1``, or a hosted provider. Its :meth:`complete`
  sends a prompt already rendered with the model's chat template (``/completions``), for
  pipelines that must control the template themselves.
* :func:`shard` deals work round-robin into N parts (``"I/N"``), so several processes - against
  one server with several slots, or on several machines - split it without overlap. Long runs
  then write one file per item and skip what is done (``run_episodes(..., out_dir=...)``).

Seeds: with ``seed`` set, call k gets ``seed + k``. The same server, model, settings and batch
give the same text; another GPU, engine version or batch (parallel requests change the order of
floating-point sums) can give other text. Log the calls rather than rely on regeneration.

Where the prompts go matters for evaluation data: a hosted provider whose terms let it train on
inputs receives your held-out items, and items in someone's training set are no longer held out.
A local server keeps them on your machine.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

T = TypeVar("T")
ShardSpec = Union[None, str, Tuple[int, int], Sequence[int]]


def parse_shard(spec: ShardSpec) -> Tuple[int, int]:
    """``"I/N"`` or ``(I, N)`` -> ``(I, N)`` with ``1 <= I <= N``; ``None`` -> ``(1, 1)``."""
    if spec is None:
        return 1, 1
    if isinstance(spec, str):
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec)
        if not m:
            raise ValueError(f"a shard is I/N, not {spec!r}")
        i, n = int(m.group(1)), int(m.group(2))
    else:
        i, n = (int(v) for v in spec)
    if not 1 <= i <= n:
        raise ValueError(f"a shard needs 1 <= I <= N, not {i}/{n}")
    return i, n


def shard(items: Sequence[T], spec: ShardSpec) -> List[T]:
    """Part I of N: items I-1, I-1+N, I-1+2N, ... (round-robin, so every part gets early and late
    items alike). With the same N everywhere the parts never overlap, and part I of the first k
    items is the beginning of part I of all of them."""
    i, n = parse_shard(spec)
    return list(items)[i - 1::n]


def _request(url: str, body: Optional[Mapping[str, Any]] = None, timeout: float = 3600,
             headers: Optional[Mapping[str, str]] = None, what: str = "server") -> Dict[str, Any]:
    import urllib.error
    import urllib.request
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST",
                                 headers={"Content-Type": "application/json", **dict(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{what} {url}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    except urllib.error.URLError as e:
        raise ConnectionError(f"{what} is not reachable at {url} ({e.reason})") from None
    except TimeoutError:
        raise TimeoutError(f"{what} at {url} gave no answer within {timeout:g} s") from None


def _drop_none(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


class OllamaClient:
    """``prompt -> text`` through a local Ollama server's chat endpoint (its chat template).

    Args:
        model: Ollama's name for it, e.g. ``"qwen3:4b-instruct-2507-fp16"``.
        url: the server (``OLLAMA_HOST``'s address).
        temperature, top_p, top_k, max_tokens, num_ctx: sampling and size; ``None`` keeps the
            model's defaults. ``num_ctx`` is per slot; keep it the same on every process that
            shares a server, or Ollama reloads the model.
        seed: call k gets ``seed + k``; ``None`` leaves it to the server.
        system: a system message sent with every prompt.
        options: further Ollama options, passed as they are.

    After each call, :attr:`last` holds the server's counts (prompt and reply tokens, why it
    stopped, seconds); :meth:`describe` the version and the model's digest, for the record.
    """

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", *, temperature: Optional[float] = None,
                 top_p: Optional[float] = None, top_k: Optional[int] = None, max_tokens: Optional[int] = None,
                 num_ctx: Optional[int] = None, seed: Optional[int] = None, system: Optional[str] = None,
                 options: Optional[Mapping[str, Any]] = None, keep_alive: str = "30m", timeout: float = 3600):
        self.model, self.url, self.seed, self.system = model, url.rstrip("/"), seed, system
        self.keep_alive, self.timeout = keep_alive, timeout
        self.options = _drop_none({"temperature": temperature, "top_p": top_p, "top_k": top_k,
                                   "num_predict": max_tokens, "num_ctx": num_ctx, **dict(options or {})})
        self.calls = 0
        self.last: Optional[Dict[str, Any]] = None

    @property
    def model_id(self) -> str:
        return f"ollama/{self.model}"

    @property
    def settings(self) -> Dict[str, Any]:
        return {**self.options, "seed": self.seed, "system": bool(self.system)}

    def __call__(self, prompt: str) -> str:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + \
            [{"role": "user", "content": str(prompt)}]
        options = dict(self.options)
        if self.seed is not None:
            options["seed"] = int(self.seed) + self.calls
        t0 = time.time()
        r = _request(self.url + "/api/chat", {"model": self.model, "messages": messages, "stream": False,
                                              "keep_alive": self.keep_alive, "options": options},
                     timeout=self.timeout, what="Ollama")
        self.calls += 1
        self.last = {"prompt_tokens": r.get("prompt_eval_count"), "tokens": r.get("eval_count"),
                     "stopped": r.get("done_reason"), "seconds": round(time.time() - t0, 3)}
        return (r.get("message") or {}).get("content", "")

    def describe(self) -> Dict[str, Any]:
        """Engine version, model digest and quantisation (``ollama list`` shows the digest's start)."""
        out: Dict[str, Any] = {"engine": "ollama", "url": self.url, "model": self.model,
                               "version": _request(self.url + "/api/version", timeout=60, what="Ollama").get("version")}
        for m in _request(self.url + "/api/tags", timeout=60, what="Ollama").get("models", []):
            if self.model in (m.get("name"), m.get("model")):
                out.update(digest=m.get("digest"), quantization=(m.get("details") or {}).get("quantization_level"))
        return out


class OpenAICompatibleClient:
    """``prompt -> text`` through an OpenAI-compatible server (vLLM, llama.cpp, LM Studio, ...).

    Args:
        model: the name the server serves the model under.
        base_url: up to and including ``/v1``.
        api_key: sent as a bearer token; or name an environment variable with ``api_key_env``.
            A local server usually needs none. The key is never recorded.
        temperature, top_p, max_tokens, seed, system: as for :class:`OllamaClient`.
        extra: further request fields, passed as they are (vLLM takes ``top_k``, ``min_p``,
            ``repetition_penalty`` here).
    """

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:8000/v1", *, api_key: Optional[str] = None,
                 api_key_env: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None,
                 max_tokens: Optional[int] = None, seed: Optional[int] = None, system: Optional[str] = None,
                 extra: Optional[Mapping[str, Any]] = None, timeout: float = 3600):
        self.model, self.base_url, self.seed, self.system, self.timeout = model, base_url.rstrip("/"), seed, system, timeout
        self._key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        self.params = _drop_none({"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                                  **dict(extra or {})})
        self.calls = 0
        self.last: Optional[Dict[str, Any]] = None

    @property
    def model_id(self) -> str:
        return f"openai-compatible/{self.model}"

    @property
    def settings(self) -> Dict[str, Any]:
        return {**self.params, "seed": self.seed, "system": bool(self.system)}

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        return _request(self.base_url + path, body, timeout=self.timeout, headers=headers, what="server")

    def _next_seed(self) -> Optional[int]:
        return None if self.seed is None else int(self.seed) + self.calls

    def _record(self, r: Dict[str, Any], t0: float) -> Dict[str, Any]:
        self.calls += 1
        usage = r.get("usage") or {}
        choice = (r.get("choices") or [{}])[0]
        self.last = {"prompt_tokens": usage.get("prompt_tokens"), "tokens": usage.get("completion_tokens"),
                     "stopped": choice.get("finish_reason"), "seconds": round(time.time() - t0, 3)}
        return choice

    def __call__(self, prompt: str) -> str:
        messages = ([{"role": "system", "content": self.system}] if self.system else []) + \
            [{"role": "user", "content": str(prompt)}]
        t0 = time.time()
        r = self._post("/chat/completions", _drop_none({"model": self.model, "messages": messages,
                                                        "seed": self._next_seed(), **self.params}))
        return ((self._record(r, t0).get("message") or {}).get("content")) or ""

    def complete(self, text: str, **params: Any) -> str:
        """Continue ``text`` as it is (``/completions``): the caller has rendered the chat template.
        ``params`` override the client's for this call (e.g. ``max_tokens``, ``stop``)."""
        t0 = time.time()
        body = _drop_none({"model": self.model, "prompt": text, "seed": self._next_seed(), **self.params, **params})
        return self._record(self._post("/completions", body), t0).get("text") or ""

    def describe(self) -> Dict[str, Any]:
        """The served models and, for vLLM, the engine version."""
        out: Dict[str, Any] = {"engine": "openai-compatible", "url": self.base_url, "model": self.model}
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            models = _request(self.base_url + "/models", timeout=60, headers=headers).get("data", [])
            out["served"] = [_drop_none({"id": m.get("id"), "root": m.get("root"), "max_model_len": m.get("max_model_len")})
                             for m in models]
        except (ConnectionError, RuntimeError):
            out["served"] = None
        try:  # vLLM answers /version next to /v1
            out["version"] = _request(re.sub(r"/v1$", "", self.base_url) + "/version", timeout=60).get("version")
            out["engine"] = "vllm" if out["version"] else out["engine"]
        except (ConnectionError, RuntimeError, ValueError):
            pass
        return out
