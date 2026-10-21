"""Select a model backend without changing research decisions or evaluation."""
from copy import deepcopy
import json
import urllib.parse
import urllib.request

from rsi_researcher.runtime.client import (ModelTransport, StrictDeepSeek,
    RANKING_PROTOCOL, MEMORY_CONTROL_PROTOCOL, object_schema, TEXT_SCHEMA, validate_schema_result)


VLLM_PARAMETERS = {"model": "qwen3-8b", "temperature": .7, "top_p": .8,
                   "top_k": 20, "seed": 42, "max_tokens": 2048}


def model_description(config=None):
    options = (config or {}).get("llm", {})
    provider = options.get("provider", "deepseek")
    if provider == "deepseek":
        return {"provider": provider, **deepcopy(StrictDeepSeek.metadata)}
    if provider != "vllm":
        raise ValueError("llm.provider must be deepseek or vllm")
    base_url = options.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("base_url must be an HTTP API base without credentials or query parameters")
    parameters = {key: options.get(key, value) for key, value in VLLM_PARAMETERS.items()}
    sampler=({"designer_temperature":options["designer_temperature"],
              "designer_top_p":options.get("designer_top_p",.95)}
             if "designer_temperature" in options else None)
    if sampler is not None and (
            not isinstance(sampler["designer_temperature"],(float,int))
            or not 0<=sampler["designer_temperature"]<=1.5
            or not 0<sampler["designer_top_p"]<=1):
        raise ValueError("invalid role-specific designer sampler")
    return {"provider": provider, "base_url": base_url, "model_parameters": parameters,
            "designer_sampler":sampler,
            "enable_thinking": options.get("enable_thinking", False),
            "request_timeout": options.get("timeout_seconds", 180),
            "api_key_env": options.get("api_key_env"),
            "api_protocol": "vllm_json_schema_v1", "ranking_protocol": RANKING_PROTOCOL,
            "memory_control_protocol": MEMORY_CONTROL_PROTOCOL,
            "model_revision": options.get("model_revision", "see server model manifest"),
            "proxy": "direct connection; ignores client proxy environment"}


class VLLMModel(ModelTransport):
    """vLLM schema-constrained JSON, using the same local checks and quotas."""
    api_protocol = "vllm_json_schema_v1"
    finish_reason = "stop"
    retry_invalid_response = True

    def __init__(self, directory, max_requests=100, limits=None, config=None):
        self.metadata = model_description(config)
        self.endpoint = self.metadata["base_url"] + "/chat/completions"
        self.key_env = self.metadata["api_key_env"]
        self.request_timeout = self.metadata["request_timeout"]
        self.model_parameters = self.metadata["model_parameters"]
        super().__init__(directory, max_requests, limits)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                        urllib.request.HTTPSHandler(context=self.ssl_context))

    def build_payload(self, prompt, schema):
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
        payload = super().build_payload(prompt, schema)
        payload["messages"][0]["content"] = (
            "Return exactly one JSON object matching the provided schema. "
            "Follow the task's language and content constraints, be concise, and do not emit thinking text. "
            "The schema describes an answer, not permission to execute a shell command.\n" + json.dumps(schema))
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "research_result", "strict": True, "schema": schema}}
        payload["chat_template_kwargs"] = {"enable_thinking": self.metadata["enable_thinking"]}
        # Keep Qwen ranking at fixed temperature / model seed for paired
        # comparisons; only the scientific DESIGN role explores varied
        # proposals. Stable episode-derived seeds make runs reproducible.
        sampler=self.metadata.get("designer_sampler")
        if sampler is not None and (self.scope or {}).get("scientific_role")=="design":
            import hashlib
            episode=(self.scope or {}).get("episode","")
            episode_hash=int.from_bytes(hashlib.sha256(episode.encode()).digest()[:4],"big")
            payload["temperature"]=sampler["designer_temperature"]
            payload["top_p"]=sampler["designer_top_p"]
            payload["seed"]=(self.model_parameters["seed"]+episode_hash)%2147483647
        return payload

    def response_content(self, choice):
        return choice["message"]["content"]

    def validate_result(self, result, schema):
        validate_schema_result(result, schema)

    def open_response(self, request, timeout):
        return self.opener.open(request, timeout=timeout)

    def health(self):
        base = self.metadata["base_url"]
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        with self.opener.open(urllib.request.Request(base.removesuffix("/v1") + "/health", headers=headers), timeout=10) as response:
            healthy = response.status == 200
        with self.opener.open(urllib.request.Request(base + "/models", headers=headers), timeout=10) as response:
            models = json.load(response)
        served = [entry["id"] for entry in models["data"]]
        if self.model_parameters["model"] not in served:
            raise ValueError("configured model is not listed by the server")
        return {"healthy": healthy, "served_models": served}


def create_model(directory, max_requests=100, limits=None, config=None):
    description = model_description(config)
    if description["provider"] == "deepseek":
        return StrictDeepSeek(directory, max_requests, limits)
    return VLLMModel(directory, max_requests, limits, config)


def check_local_model(directory, config):
    """One bounded representative structured request; never selects DeepSeek."""
    from pathlib import Path
    from rsi_researcher.runtime.client import atomic_json, usage_summary
    from rsi_researcher.runtime.limits import RunLimits
    if model_description(config)["provider"] != "vllm":
        raise ValueError("check-model is for a self-hosted vLLM backend")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    limits = RunLimits(directory, max_requests=3, max_tokens=30000, max_seconds=600)
    model = create_model(directory, limits.max_requests, limits, config)
    atomic_json(directory / "model_config.json", model.metadata)
    health = model.health()
    answer = model.ask("这是接口接入检查。当前开发证据：质量0.50，热门倾斜0.10。请提出一个最小试验；不要声称已运行或已改善。", object_schema({
        "hypothesis": TEXT_SCHEMA, "alternative_explanation": TEXT_SCHEMA,
        "operator": {"type": "string", "enum": ["retain", "popularity_penalty", "support_preserving"]},
        "strength": {"type": "number", "enum": [0, .01, .02, .05]},
        "prediction": TEXT_SCHEMA, "falsifier": TEXT_SCHEMA}), {"stage": "backend_smoke"})
    result = {"status": "completed", "health": health, "answer": answer,
              "scope": "backend compatibility only; not research effectiveness"}
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "usage.json", usage_summary(directory))
    return result
