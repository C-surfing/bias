"""Model-free bias probes on a committed author run (zero additional model calls).

Every probe reads only artifacts the AgentCF author run has already written —
``original_update/state.json``, ``original_update/train-*.json``,
``original_update/ranking.json``, ``input.json``, ``data_profile.json`` and
``calls.jsonl``.  None of them call the model, so they can run while another
run is in progress or after it finished, and they never mutate any run state.

The probes are descriptive, not causal.  A lexicon hit is an observation about
the text, not proof that the recommender is biased.
"""
from collections import Counter
import json
import re
from pathlib import Path

from rsi_researcher.datasets.author import popularity_group

# --- Lexicons (deterministic, lower-case substring matches) -----------------

NEGATIVE_SENTIMENT = (
    "dislike", "don't like", "do not like", "don't enjoy", "do not enjoy",
    "doesn't enjoy", "does not enjoy", "not a fan", "avoid", "too complex",
    "overly complex", "over-complex", "intricate", "unappealing",
    "unpleasant", "boring", "noisy", "chaotic", "inaccessible",
    "not my taste", "off-putting", "tedious", "repetitive", "dated",
    "inferior", "lacks", "doesn't appeal", "does not appeal", "cold",
    "distant", "overwrought", "disjointed", "jarring", "unpleasant",
)

POPULARITY_LEXICON = (
    "popular", "popularity", "classic", "mainstream", "widely acclaimed",
    "acclaimed", "hit", "best-selling", "bestselling", "iconic", "timeless",
    "chart-topping", "chart topper", "fan favorite", "fan favourite",
    "universally loved", "well-known", "well known", "renowned", "famous",
    "legendary", "masterpiece", "blockbuster", "record-breaking",
)

MUSIC_ATTRIBUTES = (
    "rock", "pop", "jazz", "country", "blues", "classical", "electronic",
    "electronica", "folk", "soul", "r&b", "rhythm and blues", "hip-hop",
    "hip hop", "metal", "progressive", "aor", "album-oriented rock",
    "indie", "punk", "disco", "funk", "gospel", "reggae", "ambient",
    "house", "techno", "dance", "guitar", "piano", "drums", "vocals",
    "vocal", "strings", "synthesizer", "synth", "bass", "saxophone",
    "horns", "horn", "acoustic", "electric", "melodic", "melody",
    "upbeat", "mellow", "energetic", "soothing", "rhythm", "rhythmic",
    "harmony", "harmonies", "arrangement", "arrangements", "instrumental",
    "lyrics", "lyrical", "uptempo", "downtempo", "ballad", "anthem",
    "60s", "70s", "80s", "90s", "sixties", "seventies", "eighties",
    "nineties",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+|[。！？；]")


def _sentences(text):
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _has_any(text, terms):
    low = text.lower()
    return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", low) for term in terms)


def _count_any(text, terms):
    low = text.lower()
    return sum(1 for term in terms if term in low)


def _mean(values):
    return sum(values) / len(values) if values else None


def _load_run(run_dir):
    """Read committed author-run artifacts without writing or model calls."""
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "original_update" / "state.json").read_text(encoding="utf-8"))
    completed = state["completed"]
    records = [json.loads((run_dir / "original_update" / (eid + ".json")).read_text(encoding="utf-8"))
               for eid in completed]
    inputs = json.loads((run_dir / "input.json").read_text(encoding="utf-8"))
    profile = json.loads((run_dir / "data_profile.json").read_text(encoding="utf-8"))
    ranking_rows = json.loads((run_dir / "original_update" / "ranking.json").read_text(encoding="utf-8"))
    calls_path = run_dir / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()] if calls_path.exists() else []
    return state, records, inputs, profile, ranking_rows, calls


def kendall_tau(order_a, order_b):
    """Rank correlation between two permutations of the same items.

    Returns 1.0 when orders are identical, -1.0 when reversed, and 0.0 for an
    average random permutation.  Requires the same set of items.
    """
    a, b = list(order_a), list(order_b)
    if set(a) != set(b) or len(set(a)) != len(a):
        raise ValueError("kendall_tau requires two permutations of the same items")
    pos = {item: index for index, item in enumerate(a)}
    pairs = concordant = 0
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            x, y = b[i], b[j]
            if (pos[x] < pos[y]) == (i < j):
                concordant += 1
            pairs += 1
    return (2 * concordant - pairs) / pairs if pairs else 0.0


# --- Probe 1: exposure bias (fabricated dislike of unseen comparison item) ---

def _raw_dislike_hit(after_memory, title):
    """Legacy broad lexical screen, retained for comparing old reports."""
    title_key = title.lower().strip()
    if not title_key or len(title_key) < 3 or title_key.isdigit():
        return False
    for sentence in _sentences(after_memory):
        low = sentence.lower()
        if title_key in low and any(term in low for term in NEGATIVE_SENTIMENT):
            return True
    return False


_DISLIKE_ASSERTION = re.compile(
    r"\b(?:dislikes?|disliked|hates?|avoids?|not a fan|not my taste|"
    r"(?:do not|don't|does not|doesn't) (?:like|enjoy|appeal))\b", re.I)
_NEGATED_ASSERTION = re.compile(
    r"\b(?:no|not|never|without|don't|doesn't|cannot|can't)\s+(?:[\w'-]+\s+){0,5}$", re.I)


def _false_dislike(after_memory, title):
    """Explicit unnegated dislike cue plus comparison title in one sentence.

    Still a text proxy, not a semantic error label: misses implicit aversion,
    short/numeric titles and may confuse the subject of a complex sentence.
    """
    title_key = title.lower().strip()
    if len(title_key) < 3 or title_key.isdigit():
        return False
    for sentence in _sentences(after_memory):
        low = sentence.lower()
        titles = list(re.finditer(re.escape(title_key), low))
        if not titles:
            continue
        for match in _DISLIKE_ASSERTION.finditer(sentence):
            if any(t.start() <= match.start() < t.end() for t in titles):
                continue  # the title itself is not a preference assertion
            if not _NEGATED_ASSERTION.search(sentence[:match.start()]):
                return True
    return False


def probe_exposure_bias(state, records, inputs, case_limit=3):
    """Fraction of events whose user reflection fabricates dislike of an unseen item."""
    initial_items = inputs["initial"]["items"]
    cases = []
    raw_hits = 0
    for record in records:
        event = record["event"]
        user, negative = event["user"], event["negative"]
        title = initial_items[negative]["title"]
        before = record["before"]["users"][user]
        after = record["after"]["users"][user]
        raw_hits += _raw_dislike_hit(after, title)
        if _false_dislike(after, title):
            cases.append({"event_id": event["event_id"], "user": user, "negative_id": negative,
                          "negative_title": title, "user_memory_before": before,
                          "user_memory_after": after})
    total = len(records)
    return {"events_total": total,
            "false_dislike_events": len(cases),
            "false_dislike_memory_rate": len(cases) / total if total else None,
            "legacy_raw_lexicon_rate": raw_hits / total if total else None,
            "representative_cases": cases[:case_limit],
            "lexicon_size": len(NEGATIVE_SENTIMENT),
            "scope": "Explicit unnegated dislike cue and unseen comparison title in one sentence. Screening only, not a true error rate; manual subject/negation and evidence review required."}


# --- Probe 2: conformity bias (popularity / mainstream prior) ---------------

def probe_conformity_bias(state, calls):
    # `received` rows carry no context; recover the stage from the matching
    # `started` row via call_id.
    select_calls = {c["call_id"] for c in calls
                    if c.get("status") == "started" and (c.get("context") or {}).get("stage") == "select"}
    select_texts = []
    for call in calls:
        if call.get("status") == "received" and call.get("call_id") in select_calls:
            try:
                reason = json.loads(call.get("content", "")).get("reason")
            except (ValueError, AttributeError):
                continue  # malformed transport replies are not selection reasons
            if isinstance(reason, str):
                select_texts.append(reason)
    item_memories = [item["memory"] for item in state["items"].values()]
    user_memories = list(state["users"].values())

    def rate(texts):
        return _mean([1.0 if _has_any(text, POPULARITY_LEXICON) else 0.0 for text in texts])

    return {"select_reason_popularity_rate": rate(select_texts),
            "item_memory_popularity_rate": rate(item_memories),
            "user_memory_popularity_rate": rate(user_memories),
            "popularity_prior_lexicon_rate": rate(select_texts),
            "select_reason_texts": len(select_texts),
            "lexicon_size": len(POPULARITY_LEXICON),
            "scope": "Whole-word popularity/mainstream vocabulary in parsed selection reasons and memories; not evidence of users' social conformity."}


# --- Probe 3: selection bias / PAAC information gap -------------------------

def probe_selection_paac(state, records, inputs, profile):
    counts = profile["training_item_counts"]
    head_min, tail_max = profile["head_min_count"], profile["tail_max_count"]
    initial_items = inputs["initial"]["items"]
    update_counts = Counter()
    for record in records:
        update_counts[record["event"]["positive"]] += 1
    groups = {"head": [], "mid": [], "tail": [], "unseen": []}
    for item_id, item in state["items"].items():
        group = popularity_group(item_id, counts, head_min, tail_max)
        final_memory = item["memory"]
        initial_memory = initial_items.get(item_id, {}).get("memory", final_memory)
        groups[group].append({
            "item": item_id,
            "update_count": update_counts.get(item_id, 0),
            "final_chars": len(final_memory),
            "initial_chars": len(initial_memory),
            "growth_ratio": len(final_memory) / max(1, len(initial_memory)),
            "attribute_terms": _count_any(final_memory, MUSIC_ATTRIBUTES),
        })
    result = {}
    for group, rows in groups.items():
        result[group] = {
            "count": len(rows),
            "items_with_any_update": sum(r["update_count"] > 0 for r in rows),
            "mean_update_count": _mean([r["update_count"] for r in rows]),
            "mean_memory_chars": _mean([r["final_chars"] for r in rows]),
            "mean_growth_ratio": _mean([r["growth_ratio"] for r in rows]),
            "mean_attribute_terms": _mean([r["attribute_terms"] for r in rows]),
            "items_with_attribute_terms": sum(r["attribute_terms"] > 0 for r in rows),
        }
    return {"groups": result,
            "replayed_events": sum("source_event_id" in r["event"] for r in records),
            "unique_source_events": len({r["event"].get("source_event_id", r["event"]["event_id"]) for r in records}),
            "group_definition": profile.get("group_definition"),
            "attribute_lexicon_size": len(MUSIC_ATTRIBUTES),
            "scope": "Learning-opportunity proxy; update counts are partly determined by the replay policy. Text length/lexicon coverage do not establish memory quality or causal selection debiasing; this is not PAAC reproduction."}


# --- Probe 4: position bias (ranking mirrors presentation order) ------------

def probe_position_bias(ranking_rows):
    per_query = []
    for row in ranking_rows:
        query = row["query"]
        valid = row.get("valid", False)
        ranking = row.get("ranking") or []
        tau = None
        if valid and ranking:
            try:
                tau = kendall_tau(query["candidates"], ranking)
            except ValueError:
                tau = None
        per_query.append({"query_id": query["query_id"], "valid": valid,
                          "input_order_following_tau": tau,
                          "top5_head_share": row.get("top5_head_share"),
                          "candidate_head_share": row.get("candidate_head_share")})
    taus = [r["input_order_following_tau"] for r in per_query if r["input_order_following_tau"] is not None]
    return {"input_order_following_kendall_tau": _mean(taus),
            "queries": per_query,
            "interpretation": "Correlation between prompt candidate order and output ranking; a high value "
                              "means the ranking mirrors presentation order; a negative value may indicate reverse-order following. "
                              "A value near zero does not establish preference-driven ordering. Repeated identical orders and paired permutations are needed. Descriptive, not causal."}


def candidate_order_flip_sensitivity(forward_rows, flipped_rows):
    """Compare two saved rankings produced from reversed candidate orders.

    Zero model calls here: it only compares artifacts already written when a
    flip re-rank was executed (Stage 2.3).  Both inputs are ``ranking.json``
    row lists keyed by the same ``query_id``.
    """
    by_id = {row["query"]["query_id"]: row for row in forward_rows}
    flips = {row["query"]["query_id"]: row for row in flipped_rows}
    common = sorted(set(by_id) & set(flips))
    rows = []
    for qid in common:
        f, g = by_id[qid], flips[qid]
        f_rank, g_rank = f.get("ranking") or [], g.get("ranking") or []
        tau = kendall_tau(f_rank, g_rank) if f_rank and g_rank else None
        rows.append({"query_id": qid,
                     "top1_agrees": (f_rank[:1] == g_rank[:1]) if f_rank and g_rank else None,
                     "ranking_tau": tau,
                     "ndcg_at_5_delta": (g.get("ndcg_at_5") or 0) - (f.get("ndcg_at_5") or 0),
                     "top5_head_share_delta": (g.get("top5_head_share") or 0) - (f.get("top5_head_share") or 0)})
    taus = [r["ranking_tau"] for r in rows if r["ranking_tau"] is not None]
    top1 = [r["top1_agrees"] for r in rows if r["top1_agrees"] is not None]
    return {"queries": len(rows),
            "mean_ranking_tau": _mean(taus),
            "top1_agreement_rate": _mean([1.0 if v else 0.0 for v in top1]),
            "mean_abs_ndcg_at_5_delta": _mean([abs(r["ndcg_at_5_delta"]) for r in rows]),
            "mean_abs_top5_head_share_delta": _mean([abs(r["top5_head_share_delta"]) for r in rows]),
            "per_query": rows,
            "interpretation": "How much the ranking changes when the candidate presentation order is reversed."}


# --- Aggregation ------------------------------------------------------------

def run_author_probes(run_dir):
    state, records, inputs, profile, ranking_rows, calls = _load_run(run_dir)
    return {"protocol": "author_descriptive_probes_v3",
            "scope": "model-free diagnostics on committed author-run artifacts; zero model calls",
            "dataset": profile.get("dataset"),
            "committed_events": len(state["completed"]),
            "ranking_queries": len(ranking_rows),
            "exposure_bias": probe_exposure_bias(state, records, inputs),
            "conformity_bias": probe_conformity_bias(state, calls),
            "selection_paac": probe_selection_paac(state, records, inputs, profile),
            "position_bias": probe_position_bias(ranking_rows)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="a committed author run directory (contains original_update/)")
    args = parser.parse_args()
    print(json.dumps(run_author_probes(args.run_dir), ensure_ascii=False, indent=2))
