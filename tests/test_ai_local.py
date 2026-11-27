# tests/test_ai_local.py
"""Local-model clients against stand-in servers, sharding, and resumable episodes."""
import json
import os

import pytest

from scidatasim.ai import LLMAgent, LoggedClient
from scidatasim.ai.local import OllamaClient, OpenAICompatibleClient, parse_shard, shard
from scidatasim.diagnostics import diagnose_episode, run_episodes
from scidatasim.envs import ExperimentEnv

SPEC = {"base_config": {"experiment_type": "scidatasim.experiments.law_discovery.logic.HiddenLawExperiment",
                        "n_points": 30, "counterfactual": True, "x_min": 0.5, "x_max": 5.0,
                        "observation": {"operators": [{"type": "gaussian", "snr_db": 40}]}},
        "hidden": {"law": {"choices": ["power_law"]}, "law_seed": [0, 1.0e12]},
        "controls": {"x_min": [0.1, 9.0], "x_max": [0.5, 10.0]},
        "target": "law_expression", "target_type": "expression", "domain": [0.5, 5.0], "budget": 3,
        "tolerance": 0.02}


class _Server:
    """A local HTTP server answering from a dict ``path -> function(body, headers) -> reply``."""

    def __init__(self, routes):
        import http.server
        import threading
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self, body):
                outer.requests.append((self.path, body, dict(self.headers)))
                route = routes.get(self.path)
                if route is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(route(body, self.headers)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._answer(None)

            def do_POST(self):
                self._answer(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def no_proxy(monkeypatch):  # a proxy set for the machine must not catch 127.0.0.1
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def test_shards_split_without_overlap_and_grow_with_the_list():
    assert parse_shard(None) == (1, 1) and parse_shard("2/3") == (2, 3) and parse_shard((3, 3)) == (3, 3)
    for bad in ("0/3", "4/3", "x", "1/0"):
        with pytest.raises(ValueError):
            parse_shard(bad)
    items = list(range(10))
    parts = [shard(items, f"{i}/3") for i in (1, 2, 3)]
    assert parts[0] == [0, 3, 6, 9] and sorted(sum(parts, [])) == items
    assert shard(items[:7], "2/3") == shard(items, "2/3")[:len(shard(items[:7], "2/3"))]  # --limit, then all


def test_ollama_client_records_counts_and_seeds(tmp_path, no_proxy):
    server = _Server({
        "/api/chat": lambda b, h: {"message": {"role": "assistant", "content": f"echo {b['messages'][-1]['content']}"},
                                   "prompt_eval_count": 11, "eval_count": 3, "done_reason": "stop"},
        "/api/version": lambda b, h: {"version": "9.9"},
        "/api/tags": lambda b, h: {"models": [{"name": "m:fp16", "digest": "abc", "details": {"quantization_level": "F16"}}]},
    })
    try:
        model = OllamaClient("m:fp16", server.url, temperature=0.7, top_k=20, num_ctx=8192, seed=5, system="Be brief.")
        logged = LoggedClient(model, str(tmp_path / "calls.jsonl"), role="agent", model=model.model_id,
                              settings=model.settings)
        assert logged("hi") == "echo hi" and logged("again") == "echo again"
        bodies = [b for p, b, _ in server.requests if p == "/api/chat"]
        assert [b["options"]["seed"] for b in bodies] == [5, 6]
        assert bodies[0]["messages"][0] == {"role": "system", "content": "Be brief."}
        assert bodies[0]["options"] == {"temperature": 0.7, "top_k": 20, "num_ctx": 8192, "seed": 5}
        assert model.last["prompt_tokens"] == 11 and model.last["stopped"] == "stop"
        assert model.describe() == {"engine": "ollama", "url": server.url, "model": "m:fp16", "version": "9.9",
                                    "digest": "abc", "quantization": "F16"}
        entry = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
        assert entry["model"] == "ollama/m:fp16" and entry["settings"]["seed"] == 5
    finally:
        server.close()
    with pytest.raises(ConnectionError, match="not reachable"):
        OllamaClient("m", server.url, timeout=5)("hi")


def test_openai_compatible_client_chat_raw_and_key(no_proxy, monkeypatch):
    def chat(b, h):
        return {"choices": [{"message": {"content": "chat reply"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    def raw(b, h):
        return {"choices": [{"text": " continued", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": b["max_tokens"]}}
    server = _Server({"/v1/chat/completions": chat, "/v1/completions": raw,
                      "/v1/models": lambda b, h: {"data": [{"id": "qwen", "root": "/models/qwen"}]},
                      "/version": lambda b, h: {"version": "0.99"}})
    monkeypatch.setenv("TEST_KEY", "secret-123")
    try:
        model = OpenAICompatibleClient("qwen", server.url + "/v1", api_key_env="TEST_KEY", temperature=0.7, seed=1,
                                       extra={"top_k": 20, "min_p": 0.0})
        assert model("hello") == "chat reply"
        assert model.complete("<|im_start|>user\nhi<|im_end|>\n", max_tokens=3, stop=["<|im_end|>"]) == " continued"
        (_, chat_body, chat_headers), (_, raw_body, _) = server.requests[:2]
        assert chat_body["seed"] == 1 and raw_body["seed"] == 2 and raw_body["top_k"] == 20
        assert raw_body["prompt"].startswith("<|im_start|>") and raw_body["stop"] == ["<|im_end|>"]
        assert chat_headers["Authorization"] == "Bearer secret-123"
        assert "secret" not in json.dumps(model.settings)
        assert model.last == {"prompt_tokens": 4, "tokens": 3, "stopped": "length", "seconds": model.last["seconds"]}
        d = model.describe()
        assert d["engine"] == "vllm" and d["version"] == "0.99" and d["served"] == [{"id": "qwen", "root": "/models/qwen"}]
    finally:
        server.close()


def test_run_episodes_resumes_and_shards(tmp_path):
    out = str(tmp_path / "episodes")
    answer = LLMAgent(lambda p: '{"type": "answer", "value": "2*x"}')
    first = run_episodes(ExperimentEnv(SPEC), answer, seeds=range(5), out_dir=out, shard="1/2")
    assert [log.seed for log in first] == [0, 2, 4]
    assert sorted(os.listdir(out)) == ["episode-0.json", "episode-2.json", "episode-4.json"]

    def must_not_be_called(prompt):
        raise AssertionError("a finished episode was played again")
    again = run_episodes(ExperimentEnv(SPEC), LLMAgent(must_not_be_called), seeds=range(5), out_dir=out, shard="1/2")
    assert [diagnose_episode(a) for a in again] == [diagnose_episode(f) for f in first]
    other = run_episodes(ExperimentEnv(SPEC), answer, seeds=range(5), out_dir=out, shard="2/2")
    assert [log.seed for log in other] == [1, 3] and len(os.listdir(out)) == 5
