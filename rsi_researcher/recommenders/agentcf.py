"""AgentCF preference learning and ranking; research policy lives elsewhere."""
from copy import deepcopy
from string import Template
from rsi_researcher.runtime.client import (BudgetExceeded, InvalidOutput, ModelFailure, RANKING_PROTOCOL, TEXT_SCHEMA, fields, nonempty_text, object_schema, repair_ranking_membership, validate_ranking)

class AgentCF:
    def __init__(self, model, prompts):
        self.model, self.prompts = model, prompts

    def render(self, key, values):
        return Template(self.prompts[key]).substitute(values)

    def descriptions(self, state, candidates):
        if len(set(candidates)) != len(candidates) or not candidates or not set(candidates) <= state["items"].keys():
            raise ValueError("illegal candidate panel")
        return "\n".join(f"{n + 1}. ID={i}; {state['items'][i]['memory']}" for n, i in enumerate(candidates))

    def event(self, state, event, arm):
        if arm not in ("original_update", "frozen", "user_only", "item_only"):
            raise ValueError("unknown arm")
        if event["event_id"] in state["completed"]:
            raise ValueError("event already committed")
        user, positive, negative = event["user"], event["positive"], event["negative"]
        if set(event["candidates"]) != {positive, negative} or positive == negative:
            raise ValueError("event candidates do not match feedback")
        working = deepcopy(state)
        for round_id in range(2):
            values = {"user_description": working["users"][user],
                      "list_of_item_description": self.descriptions(working, event["candidates"]),
                      "pos_item_title": working["items"][positive]["title"],
                      "neg_item_title": working["items"][negative]["title"]}
            context = {"arm": arm, "event": event["event_id"], "round": round_id}
            selected = self.model.ask(self.render("system_prompt_template", values),
                                      object_schema({"selected_id": {"type": "string", "enum": sorted(event["candidates"])},
                                                     "reason": TEXT_SCHEMA}), context | {"stage": "select"})
            fields(selected, ("selected_id", "reason"))
            if selected["selected_id"] not in event["candidates"]:
                raise InvalidOutput("selected ID outside candidates")
            values["system_reason"] = nonempty_text(selected["reason"])
            correct = selected["selected_id"] == positive
            if not correct or round_id == 0:
                template = "user_prompt_template_true" if correct else "user_prompt_template"
                prompt = self.render("user_prompt_system_role", values) + "\n" + self.render(template, values)
                updated = self.model.ask(prompt, object_schema({"memory": TEXT_SCHEMA}), context | {"stage": "user_update"})
                fields(updated, ("memory",))
                memory = nonempty_text(updated["memory"])
                if arm in ("original_update", "user_only"):
                    values["user_description"] = memory
                    working["users"][user] = memory
            template = "item_prompt_template_true" if correct else "item_prompt_template"
            updated = self.model.ask(self.render(template, values),
                                     object_schema({"first_memory": TEXT_SCHEMA, "second_memory": TEXT_SCHEMA}),
                                     context | {"stage": "item_update"})
            fields(updated, ("first_memory", "second_memory"))
            for field in ("first_memory", "second_memory"):
                nonempty_text(updated[field])
            # Only the observed positive item is written, including when input order is reversed.
            positive_field = ("first_memory", "second_memory")[event["candidates"].index(positive)]
            if arm in ("original_update", "item_only"):
                working["items"][positive]["memory"] = updated[positive_field]
        working["completed"].append(event["event_id"])
        return working

    def rank(self, state, query):
        candidates = query["candidates"]
        values = {"user_description": state["users"][query["user"]], "candidate_num": len(candidates),
                  "example_list_of_item_description": self.descriptions(state, candidates)}
        result = self.model.ask(self.render("system_prompt_template_evaluation_basic", values),
                                object_schema({"ranking": {"type": "array", "items": {"type": "string", "enum": sorted(candidates)}}}),
                                {"stage": "rank", "query": query["query_id"]})
        try:
            return validate_ranking(result, candidates)
        except InvalidOutput:
            record = getattr(self.model, "record", None)
            context = (getattr(self.model, "scope", None) or {}) | {"query": query["query_id"]}
            if record:
                record({"status": "ranking_membership_invalid", "context": context, "protocol": RANKING_PROTOCOL})
            try:
                ranking = repair_ranking_membership(self, state, query, result)
            except (InvalidOutput, ModelFailure, BudgetExceeded):
                if record:
                    record({"status": "ranking_correction_failed", "context": context})
                raise
            if record:
                record({"status": "ranking_correction_succeeded", "context": context})
            return ranking


