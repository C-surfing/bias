"""Fresh data evidence must not be blocked by an unfinished earlier hypothesis."""
from rsi_researcher.research.discovery.open_research import should_form_concepts


def test_open_discovery_accepts_fresh_relational_evidence_with_active_branch():
    ongoing = [{"id": "H001", "status": "investigating"}]
    assert should_form_concepts(
        open_phenomena=True, new_relational_evidence=True,
        legacy_formation=False, concepts=ongoing)
    assert not should_form_concepts(
        open_phenomena=True, new_relational_evidence=False,
        legacy_formation=False, concepts=ongoing)


def test_legacy_mode_keeps_prior_rejection_rule():
    current = [{"status": "needs_revision"}]
    assert not should_form_concepts(
        open_phenomena=False, new_relational_evidence=True,
        legacy_formation=True, concepts=current)
    assert should_form_concepts(
        open_phenomena=False, new_relational_evidence=False,
        legacy_formation=True, concepts=[{"status": "rejected_development"}])
