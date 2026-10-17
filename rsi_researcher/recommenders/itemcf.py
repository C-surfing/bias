"""CPU item-to-item collaborative filtering and train-only popularity controls.

Rows describe *observed* feedback.  A missing interaction is an absent row;
rating zero is valid observed feedback (for example an exposed, unclicked
video).  Dataset readers must remove a dataset's missing-value placeholders
before constructing this recommender.
"""

from collections import Counter, defaultdict
import math


POLICIES = {"retain", "popularity_penalty", "user_calibrated", "support_preserving"}


class ItemCF:
    """Positive-feedback cosine co-occurrence, with top neighbours per item.

    Ranking reads candidate IDs, never their evaluation ratings.  Scores and
    popularity penalties are scaled against the fixed training catalogue, so
    evaluating a different panel cannot change a pair's relative order.
    """

    def __init__(self, train, item_ids, positive_threshold=4, neighbors=40):
        self.item_ids = tuple(sorted(str(item) for item in item_ids))
        if not self.item_ids or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("training catalogue must contain distinct item IDs")
        if type(neighbors) is not int or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if (type(positive_threshold) not in (int, float)
                or not math.isfinite(positive_threshold) or positive_threshold <= 0):
            raise ValueError("positive threshold must be finite and positive")
        self._catalogue = set(self.item_ids)
        self._positive = defaultdict(set)
        self._seen = defaultdict(set)
        self.train_counts = dict.fromkeys(self.item_ids, 0)
        item_sums = dict.fromkeys(self.item_ids, 0.0)
        observation_count = 0
        rating_sum = 0.0
        for row in train:
            user, item, rating = str(row["user"]), str(row["item"]), row["rating"]
            if item not in self._catalogue:
                raise ValueError("observed item is absent from the training catalogue")
            if type(rating) not in (int, float) or not math.isfinite(rating) or rating < 0:
                raise ValueError("observed rating must be finite and nonnegative")
            # All supplied rows are observations, including binary zero feedback.
            self._seen[user].add(item)
            self.train_counts[item] += 1
            item_sums[item] += rating
            observation_count += 1
            rating_sum += rating
            if rating >= positive_threshold:
                self._positive[user].add(item)

        head_count = max(1, math.ceil(len(self.item_ids) * .2))
        self.head_items = frozenset(sorted(self.item_ids,
            key=lambda item: (-self.train_counts[item], item))[:head_count])
        maximum_count = max(self.train_counts.values())
        log_scale = math.log1p(maximum_count)
        self._popularity = {item: math.log1p(count) / log_scale if log_scale else 0.0
                            for item, count in self.train_counts.items()}
        overall_mean = rating_sum / observation_count if observation_count else 0.0
        self._fallback = {item: item_sums[item] / count if count else overall_mean
                          for item, count in self.train_counts.items()}
        positive_count = sum(len(items) for items in self._positive.values())
        head_positive_count = sum(len(items & self.head_items) for items in self._positive.values())
        self._global_head_ratio = head_positive_count / positive_count if positive_count else 0.0

        item_users = defaultdict(set)
        for user, items in self._positive.items():
            for item in items:
                item_users[item].add(user)
        self._neighbors = {}
        for item in self.item_ids:
            # One counter at a time avoids storing a dense item-item matrix.
            overlaps = Counter()
            for user in sorted(item_users[item]):
                overlaps.update(self._positive[user])
            overlaps.pop(item, None)
            similarities = [(other, count / math.sqrt(len(item_users[item]) * len(item_users[other])))
                            for other, count in overlaps.items()]
            self._neighbors[item] = tuple(sorted(similarities,
                key=lambda pair: (-pair[1], pair[0]))[:neighbors])
        self._score_cache = {}
        self.metadata = {
            "algorithm": "positive_item_cosine_top_neighbors_v1",
            "positive_threshold": positive_threshold, "neighbors": neighbors,
            "catalogue_items": len(self.item_ids), "training_observations": observation_count,
            "training_users": len(self._seen), "head_fraction": .2, "head_items": head_count,
            "popularity_source": "observed_training_rows_including_zero_feedback",
            "missing_feedback": "absent_row_not_rating_zero",
            "score_normalization": "divide_by_maximum_over_fixed_training_catalogue",
            "fallback": "training_item_mean_then_global_mean; all_empty_training_scores_are_zero",
            "personalization": "1_minus_training_positive_head_share; empty_history_uses_training_global_share",
            "interpretation": "Historical feedback is an exposure-affected proxy, not preference truth.",
        }

    def _scores(self, user):
        """Sparse train-only scores and their full-catalogue normalization scale."""
        if user not in self._score_cache:
            scores = defaultdict(float)
            for item in sorted(self._positive.get(user, ())):
                for other, similarity in self._neighbors[item]:
                    scores[other] += similarity
            if not scores:
                scores = self._fallback
            self._score_cache[user] = (scores, max(scores.values(), default=0.0))
        return self._score_cache[user]

    def _policy(self, user, policy):
        policy = policy or {"operator": "retain", "strength": 0.0}
        operator, strength = policy.get("operator", "retain"), policy.get("strength", 0.0)
        if operator not in POLICIES:
            raise ValueError("unknown output policy")
        if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("output penalty strength must be between zero and one")
        if operator == "retain":
            return 0.0
        if operator == "user_calibrated":
            positives = self._positive.get(user, set())
            head_share = len(positives & self.head_items) / len(positives) if positives else self._global_head_ratio
            strength *= 1.0 - head_share
        return strength

    def rank(self, user, items, policy=None):
        """Return the complete candidate permutation without reading labels."""
        user = str(user)
        candidates = [str(item) for item in items]
        if len(set(candidates)) != len(candidates) or any(item not in self._catalogue for item in candidates):
            raise ValueError("candidate IDs must be distinct members of the training catalogue")
        scores, scale = self._scores(user)
        strength = self._policy(user, policy)
        if policy and policy.get("operator") == "support_preserving":
            # A proportional penalty cannot push positive evidence below an
            # unscored item when strength < 1. It still may harm relevance.
            return sorted(candidates, key=lambda item:
                (-(scores.get(item, 0.0) / scale if scale else 0.0) *
                  (1.0 - strength * self._popularity[item]), item))
        return sorted(candidates, key=lambda item:
            (-(scores.get(item, 0.0) / scale if scale else 0.0) + strength * self._popularity[item], item))

    def diagnose_panels(self, panel_rows):
        """Describe train-score support before intervention; never reads labels."""
        panels = defaultdict(set)
        for row in panel_rows:
            panels[str(row["user"])].add(str(row["item"]))
        users = []
        for user, items in sorted(panels.items()):
            scores, scale = self._scores(user)
            values = [scores.get(item, 0.0) / scale if scale else 0.0 for item in items]
            users.append({"user": user, "candidates": len(items),
                          "positive_score_candidates": sum(value > 0 for value in values),
                          "zero_score_candidates": sum(value == 0 for value in values),
                          "distinct_scores": len(set(values)),
                          "training_positive_items": len(self._positive.get(user, ()))})
        count = sum(row["candidates"] for row in users)
        return {"users": len(users), "candidate_pairs": count,
                "zero_score_fraction": sum(row["zero_score_candidates"] for row in users) / count if count else None,
                "users_without_positive_panel_scores": sum(row["positive_score_candidates"] == 0 for row in users),
                "users_without_training_positives": sum(row["training_positive_items"] == 0 for row in users),
                "scope": "training-derived score support on current development candidate IDs; not bias causality",
                "per_user": users}

    def rank_panels(self, panel_rows, policy=None):
        """Group user/item IDs; ignore all other fields, including test ratings."""
        panels = defaultdict(set)
        for row in panel_rows:
            panels[str(row["user"])].add(str(row["item"]))
        return {user: self.rank(user, sorted(items), policy) for user, items in sorted(panels.items())}

    def rank_catalog(self, users, policy=None, k=10):
        """Rank the fixed catalogue, excluding every observed training item."""
        if type(k) is not int or k < 1:
            raise ValueError("catalogue ranking length must be a positive integer")
        return {user: self.rank(user, (item for item in self.item_ids
                    if item not in self._seen.get(user, set())), policy)[:k]
                for user in sorted({str(value) for value in users})}
