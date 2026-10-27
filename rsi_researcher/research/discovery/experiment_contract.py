"""Bind model-chosen scientific questions to verified development columns/arms.

Qwen chooses comparison semantics. Python constructs mechanical SQL/analysis fields.
The contract does not claim a randomized experiment or identify causal bias.
"""
from __future__ import annotations

import re
from jsonschema import Draft202012Validator

_NUMERIC = ("BOOLEAN", "INTEGER", "INT", "BIGINT", "SMALLINT", "TINYINT",
            "DOUBLE", "FLOAT", "REAL", "DECIMAL", "HUGEINT", "UBIGINT")


def experiment_intent_schema(tools, selected_contract=None):
    records = tools.feature_records
    choices = [name for name in records]
    if selected_contract:
        chosen = selected_contract["group_expression"]
        choices = [chosen] if chosen in records else choices
    tables = {r["source_table"] for r in records.values()}
    outcomes = sorted({
        c["column"] for table in tables for c in tools.schema()[table]
        if c["type"].upper().startswith(_NUMERIC)
        and c["column"] not in ("uid", "vid", "event_day", "moment_ms")})
    values = sorted({str(row[r["name"]]) for r in records.values()
                     for row in r["evidence"]["rows"]
                     if r["name"] in row and row[r["name"]] is not None})
    props = {
        "question": {"type": "string", "minLength": 8},
        "group_feature": {"type": "string", "enum": choices},
        "control_value": {"type": "string", "enum": values},
        "treatment_value": {"type": "string", "enum": values},
        "outcome": {"type": "string", "enum": outcomes},
        "population_filter": {"type": "string"},
        "analysis": {"type": "string", "enum": ["paired_user", "cluster_weighted"]},
        "prediction": {"type": "string", "minLength": 5},
        "falsifier": {"type": "string", "minLength": 5},
        "meaningful_margin": {"type": "number", "minimum": 0, "maximum": 1},
    }
    if selected_contract:
        for name, key in [("control_value","control_value"),
                          ("treatment_value","treatment_value"),
                          ("outcome","outcome_expression"),
                          ("analysis","analysis")]:
            props[name]["enum"] = [selected_contract[key]]
    return {"type": "object", "properties": props,
            "required": list(props), "additionalProperties": False}


def compile_experiment_intent(intent, tools, selected_contract=None):
    Draft202012Validator(
        experiment_intent_schema(tools, selected_contract)).validate(intent)
    name = intent["group_feature"]
    record = tools.feature_records[name]
    levels = {str(row[name]) for row in record["evidence"]["rows"]
              if name in row and row[name] is not None}
    if intent["control_value"] == intent["treatment_value"]:
        raise ValueError("The two comparison groups must differ")
    if not {intent["control_value"], intent["treatment_value"]} <= levels:
        raise ValueError("Both comparison arms must be actual values of ONE registered feature")
    dtype = next((c["type"] for c in tools.schema()[record["source_table"]]
                  if c["column"] == intent["outcome"]), "")
    if not dtype.upper().startswith(_NUMERIC):
        raise ValueError("Outcome must be one numeric per-event column")
    if selected_contract and selected_contract["source_table"] != record["source_table"]:
        raise ValueError("Selected contrast source does not match the registered feature")
    return {
        "question": intent["question"],
        "source_table": record["source_table"],
        "source_query": "",
        "population_filter": intent["population_filter"],
        "unit_expression": "uid",
        "group_expression": name,
        "control_value": intent["control_value"],
        "treatment_value": intent["treatment_value"],
        "outcome_expression": intent["outcome"],
        "strata_expression": "",
        "analysis": intent["analysis"],
        "placebo_source_query": "",
        "placebo_group_expression": "",
        "prediction": intent["prediction"],
        "falsifier": intent["falsifier"],
        "meaningful_margin": intent["meaningful_margin"],
    }
