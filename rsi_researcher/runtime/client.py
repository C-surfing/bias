"""Shared model HTTP transport and checked response protocols."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import ssl
import time
import urllib.error
import urllib.request
import certifi
from rsi_researcher.paths import ROOT

AUTHOR_COMMIT = "71ff3b5bce69432556884c49fe7f19528b6756ec"
RANKING_PROTOCOL = "permutation_then_missing_insertions_once_v2"
JSON_API_PROTOCOL = "at_most_two_retries_including_empty_json_v2"
API_PROTOCOL = "strict_local_schema_checked_two_retries_v4"
MEMORY_CONTROL_PROTOCOL = "feedback_visible_selective_writes_v1"
MODEL_PARAMETERS = {"model": "deepseek-flash", "temperature": 0,
                    "thinking": {"type": "disabled"}, "max_tokens": 1200}
# Some grammar backends match the entire string; keep JSON Schema's intended
# "contains a non-whitespace character" meaning in either interpretation.
TEXT_SCHEMA = {"type": "string", "pattern": r"[\s\S]*\S[\s\S]*"}


class InvalidOutput(ValueError):
    pass


class SchemaViolation(InvalidOutput):
    pass


class BudgetExceeded(RuntimeError):
    pass


class ModelFailure(RuntimeError):
    def __init__(self, message, http_status=None):
        super().__init__(message)
        self.http_status = http_status


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidOutput("duplicate JSON field")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=unique)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidOutput("invalid JSON response") from exc


def fields(result, names):
    if not isinstance(result, dict) or set(result) != set(names):
        raise InvalidOutput("unexpected response fields")


def nonempty_text(value):
    if not isinstance(value, str) or not value.strip():
        raise InvalidOutput("empty or non-text memory/reason")
    return value


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def submission_content(calls):
    if not isinstance(calls, list) or len(calls) != 1:
        raise InvalidOutput("expected exactly one result submission")
    call = calls[0]
    if call["type"] != "function" or call["function"]["name"] != "submit_result":
        raise InvalidOutput("unexpected result function")
    return call["function"]["arguments"]


def logged_response_content(row):
    return submission_content(row["tool_calls"]) if row.get("tool_calls") is not None else row["content"]


def validate_ranking(result, candidates):
    fields(result, ("ranking",))
    ranking = result["ranking"]
    if not isinstance(ranking, list) or any(not isinstance(i, str) for i in ranking) or len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise InvalidOutput("ranking is not a candidate permutation")
    return ranking


def repair_ranking_membership(kernel, state, query, previous):
    candidates = query["candidates"]
    fields(previous, ("ranking",))
    old = previous["ranking"]
    if not isinstance(old, list) or any(not isinstance(i, str) or i not in candidates for i in old):
        raise InvalidOutput("insertion repair requires known IDs in the previous ranking")
    retained = list(dict.fromkeys(old))
    missing = [i for i in candidates if i not in retained]
    if not missing:
        return validate_ranking({"ranking": retained}, candidates)
    values = {"user_description": state["users"][query["user"]], "candidate_num": len(candidates),
              "example_list_of_item_description": kernel.descriptions(state, candidates)}
    prompt = kernel.render("system_prompt_template_evaluation_basic", values)
    prompt += ("\nDo not repeat the full ranking. Decide only where to insert each omitted item according to "
               "the stated preferences. Use an existing ranked ID as before_id, or null to append at the end. "
               "Include every omitted item once. Insertions sharing the same anchor follow your listed order. "
               "The existing order will be preserved by the program.\nExisting order: " + json.dumps(retained)
               + "\nOmitted IDs: " + json.dumps(missing))
    anchor_schema = {"anyOf": [{"type": "string", "enum": sorted(retained)}, {"type": "null"}]} if retained else {"type": "null"}
    schema = object_schema({"insertions": {"type": "array", "items": object_schema({
        "item_id": {"type": "string", "enum": sorted(missing)}, "before_id": anchor_schema})}})
    result = kernel.model.ask(prompt, schema,
                              {"stage": "rank_missing_insertions", "query": query["query_id"]})
    fields(result, ("insertions",))
    insertions = result["insertions"]
    if not isinstance(insertions, list) or len(insertions) != len(missing):
        raise InvalidOutput("insertion list does not cover missing items")
    remaining, ranking = set(missing), retained.copy()
    for insertion in insertions:
        fields(insertion, ("item_id", "before_id"))
        item, anchor = insertion["item_id"], insertion["before_id"]
        if not isinstance(item, str) or item not in remaining or (anchor is not None and (not isinstance(anchor, str) or anchor not in retained)):
            raise InvalidOutput("invalid insertion or anchor")
        remaining.remove(item)
        ranking.insert(len(ranking) if anchor is None else ranking.index(anchor), item)
    return validate_ranking({"ranking": ranking}, candidates)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class ModelTransport:
    endpoint = None
    api_protocol = JSON_API_PROTOCOL
    finish_reason = "stop"
    scope = None
    retry_invalid_response = False
    key_env = None
    request_timeout = 45
    model_parameters = {}

    def __init__(self, directory, max_requests=100, limits=None):
        self.key = os.environ.get(self.key_env) if self.key_env else None
        if self.key_env and not self.key:
            raise ModelFailure(self.key_env + " is missing")
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.log = Path(directory) / "calls.jsonl"
        self.limits = limits
        if (Path(directory) / "limits.json").exists() and limits is None:
            raise ValueError("resuming this run requires its original limits")
        if limits is not None and limits.max_requests != max_requests:
            raise ValueError("request limit disagrees with frozen limits")
        self.max_requests = max_requests
        previous = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []
        self.requests = sum(row["status"] == "started" for row in previous)

    def record(self, value):
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def build_payload(self, prompt, schema):
        schema = json.dumps(schema) if isinstance(schema, dict) else schema
        return {
            **deepcopy(self.model_parameters),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "Return only a JSON object. This JSON format replaces all output-format instructions in the user template: " + schema + ". Use the supplied item IDs, never titles, for ID fields."},
                {"role": "user", "content": prompt},
            ],
        }

    def response_content(self, choice):
        return choice["message"]["content"]

    def validate_result(self, result, schema):
        pass

    def open_response(self, request, timeout):
        return urllib.request.urlopen(request, timeout=timeout, context=self.ssl_context)

    def ask(self, prompt, schema, context):
        context = (self.scope or {}) | context
        payload = self.build_payload(prompt, schema)
        for attempt in range(3):
            if self.requests >= self.max_requests:
                raise BudgetExceeded("request budget exhausted")
            timeout = min(self.request_timeout, self.limits.remaining_seconds()) if self.limits else self.request_timeout
            reservation = self.limits.reserve(payload) if self.limits else None
            self.requests += 1
            call_id = self.requests
            start = time.monotonic()
            self.record({"status": "started", "call_id": call_id, "attempt": attempt + 1,
                         "context": context, "time_utc": datetime.now(timezone.utc).isoformat(),
                         "api_protocol": self.api_protocol, "endpoint": self.endpoint,
                         "request": payload, **({"reserved_tokens": reservation} if self.limits else {})})
            try:
                request = urllib.request.Request(
                    self.endpoint, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + self.key} if self.key else {})},
                )
                with self.open_response(request, timeout) as response:
                    body = response.read().decode("utf-8")
            except (urllib.error.URLError, OSError, socket.timeout) as exc:
                code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                self.record({"status": "failed", "call_id": call_id, "http_status": code,
                             "error_type": type(exc).__name__, "usage_known": False,
                             "seconds": time.monotonic() - start})
                if attempt == 2 or (code is not None and code != 429 and code < 500):
                    raise ModelFailure("model request failed; see sanitized call log", http_status=code) from None
                time.sleep(min(attempt + 1, self.limits.remaining_seconds()) if self.limits else attempt + 1)
                continue
            content, choice = None, None
            try:
                data = parse_json(body)
                choice = data["choices"][0]
                content = choice["message"].get("content")
                self.record({"status": "received", "call_id": call_id,
                             "response_model": data.get("model"), "response_id": data.get("id"),
                             "finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                             "content": content, "tool_calls": choice["message"].get("tool_calls"),
                             "seconds": time.monotonic() - start})
                if self.limits:
                    self.limits.settle(reservation, data.get("usage"))
                if choice.get("finish_reason") != self.finish_reason:
                    raise InvalidOutput("incomplete model response")
                content = self.response_content(choice)
                result = parse_json(content)
                self.validate_result(result, schema)
                return result
            except (KeyError, IndexError, TypeError, InvalidOutput) as exc:
                empty = isinstance(content, str) and not content.strip() and isinstance(choice, dict) and choice.get("finish_reason") == self.finish_reason
                self.record({"status": "invalid_response", "call_id": call_id,
                             "reason": "schema_violation" if isinstance(exc, SchemaViolation) else "empty_content" if empty else "invalid_or_incomplete"})
                if (empty or self.retry_invalid_response) and attempt < 2:
                    if self.retry_invalid_response:
                        payload["messages"][0]["content"] += (
                            " The previous submission failed local format validation. Return one complete result, "
                            "without duplicated fields. Required top-level fields: " + ", ".join(schema["required"]) + ".")
                        if isinstance(exc, SchemaViolation):
                            payload["messages"][0]["content"] += " Validation detail: " + str(exc)
                    time.sleep(min(attempt + 1, self.limits.remaining_seconds()) if self.limits else attempt + 1)
                    continue
                raise InvalidOutput("invalid or incomplete model response") from None


class DeepSeek(ModelTransport):
    endpoint = "https://api.deepseek.com/chat/completions"
    key_env = "DEEPSEEK_API_KEY"
    model_parameters = MODEL_PARAMETERS


def validate_schema_result(result, schema):
    from jsonschema import Draft202012Validator, ValidationError
    try:
        Draft202012Validator(schema).validate(result)
    except ValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        detail = "allowed values: " + json.dumps(exc.validator_value, ensure_ascii=False) if exc.validator == "enum" else "constraint: " + str(exc.validator)
        raise SchemaViolation("response field " + path + " violates schema; " + detail) from None


class StrictDeepSeek(DeepSeek):
    endpoint = "https://api.deepseek.com/beta/chat/completions"
    api_protocol = API_PROTOCOL
    finish_reason = "tool_calls"
    retry_invalid_response = True
    metadata = {"api_protocol": API_PROTOCOL, "ranking_protocol": RANKING_PROTOCOL,
                "memory_control_protocol": MEMORY_CONTROL_PROTOCOL, "model_parameters": MODEL_PARAMETERS}

    def build_payload(self, prompt, schema):
        from jsonschema import Draft202012Validator
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("strict output needs an explicit object schema")
        Draft202012Validator.check_schema(schema)
        payload = super().build_payload(prompt, json.dumps(schema))
        del payload["response_format"]
        payload["messages"][0]["content"] = (
            "Submit exactly one result using submit_result. Its JSON schema replaces all output-format "
            "instructions in the user template. Supply one answer, not multiple alternatives. "
            "Use the supplied item IDs for ID fields. This submission does not execute any action.")
        payload["tools"] = [{"type": "function", "function": {"name": "submit_result", "strict": True,
                             "description": "Submit the requested structured answer without executing an action.",
                             "parameters": schema}}]
        payload["tool_choice"] = {"type": "function", "function": {"name": "submit_result"}}
        return payload

    def response_content(self, choice):
        return submission_content(choice["message"]["tool_calls"])

    def validate_result(self, result, schema):
        validate_schema_result(result, schema)


def usage_summary(directory):
    rows = [json.loads(line) for line in (Path(directory) / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    received = {r["call_id"]: r for r in rows if r["status"] == "received"}
    started = [r for r in rows if r["status"] == "started"]
    prompt_tokens = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in received.values())
    completion_tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in received.values())
    self_hosted = bool(started) and all(r.get("api_protocol") == "vllm_json_schema_v1" for r in started)
    return {"requests": len(started), "received": len(received),
            "requests_with_unknown_usage": sum(r["call_id"] not in received or received[r["call_id"]].get("usage") is None for r in started),
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "usd_upper_estimate_for_reported_usage": None if self_hosted else (prompt_tokens * 0.3 + completion_tokens * 1.2) / 1_000_000,
            "price_assumption": "Self-hosted GPU cost not estimated; no DeepSeek fee applies" if self_hosted else "2026-10-13 Flash peak: input $0.3/M, output $1.2/M; ignores cache savings; unknown usage excluded",
            "price_source": None if self_hosted else "https://api-docs.deepseek.com/quick_start/pricing/"}


