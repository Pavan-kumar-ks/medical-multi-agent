"""LLM Evaluation Harness for MedOrchestrator.

Scores each agent response on four dimensions:
  1. relevance          — Does the response address the query?
  2. factual_consistency — Are claims internally consistent and grounded?
  3. task_completion     — Did the agent do what was expected?
  4. hallucination_free  — No invented facts (phone numbers, drug doses, URLs)

Scoring strategy:
  • Rule-based checks: JSON structure, required fields, confidence ranges (fast, free)
  • LLM-as-judge:      Groq evaluates quality on a 0–1 scale (deeper, costs tokens)

Usage:
  harness = EvaluationHarness(use_llm_judge=True)
  result  = harness.score_response(query_dict, response_text)
  report  = harness.run_batch(TEST_QUERIES[:10], run_fn)
"""
from __future__ import annotations

import re
import json
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from app.observability.logger import get_logger
    _log = get_logger("medorchestrator.eval")
except Exception:
    _log = logging.getLogger("medorchestrator.eval")

try:
    from app.recovery.hallucination_guard import detect_hallucination
    _GUARD_OK = True
except Exception:
    _GUARD_OK = False


# ── Dimension weights ─────────────────────────────────────────────────────────

WEIGHTS = {
    "relevance":           0.30,
    "factual_consistency": 0.25,
    "task_completion":     0.30,
    "hallucination_free":  0.15,
}


# ── Rule-based scorers ────────────────────────────────────────────────────────

def _score_relevance_rule(response: str, query: Dict) -> float:
    """Check that the response mentions content related to the query.

    Supports two modes (set via expected._must_contain_mode):
      "all"  (default) — every keyword must appear
      "any"            — at least ONE keyword must appear (OR logic)
    """
    expected     = query.get("expected", {})
    must_contain = [k.lower() for k in expected.get("must_contain", [])]
    must_not     = [k.lower() for k in expected.get("must_not_contain", [])]
    mode         = expected.get("_must_contain_mode", "all")
    resp_lower   = response.lower()

    # must_not penalties (apply regardless of mode)
    misses = sum(1 for kw in must_not if kw in resp_lower)
    penalty = 0.25 * misses

    if not must_contain:
        # No positive constraints — score by absence of must_not content only
        return max(0.0, 0.80 - penalty)

    if mode == "any":
        # Pass as long as at least one keyword appears
        any_hit = any(kw in resp_lower for kw in must_contain)
        score   = 1.0 if any_hit else 0.0
    else:
        # Default ALL mode — fraction of keywords present
        hits  = sum(1 for kw in must_contain if kw in resp_lower)
        score = hits / len(must_contain)

    return max(0.0, min(1.0, score - penalty))


def _score_task_completion_rule(response: str, query: Dict) -> float:
    """Check that the task type was correctly handled."""
    expected = query.get("expected", {})
    task_type = expected.get("type", "diagnosis")
    resp_lower = response.lower()

    if task_type == "non_medical":
        # Ideal: system politely declines with a reference to medical scope
        decline_signals = [
            "medical", "designed", "only", "medical query", "cannot help",
            "not able", "medical assistant", "clinical", "health",
        ]
        # Negative signals: acted like it was a real diagnosis
        acted_as_diagnosis = bool(re.search(
            r"confidence\s*:\s*\d|diagnosis|most likely|emergency contacts|panel review",
            resp_lower,
        ))
        if acted_as_diagnosis:
            return 0.15   # Definitely wrong behaviour
        return 0.85 if any(s in resp_lower for s in decline_signals) else 0.55

    if task_type == "emergency":
        # Primary strong signals — any one is sufficient for 0.75 base
        strong_signals = ["emergency", "urgent", "ambulance", "immediate"]
        # Supporting signals — add 0.10 each
        support_signals = ["hospital", "seek", "call", "911", "112", "doctor", "surgery"]
        has_strong  = any(s in resp_lower for s in strong_signals)
        support_hits = sum(1 for s in support_signals if s in resp_lower)
        base  = 0.75 if has_strong else 0.30
        bonus = min(0.25, support_hits * 0.10)
        return min(1.0, base + bonus)

    if task_type == "followup":
        # Should be a follow-up answer, NOT a new diagnosis
        diagnosis_re_run = ["top diagnosis", "diagnoses:", "confidence:", "panel review"]
        if any(s in resp_lower for s in diagnosis_re_run):
            return 0.3  # Re-ran diagnosis instead of answering
        answer_signals = ["advice", "recommend", "diet", "remedy", "medication", "warning", "prognosis"]
        hits = sum(1 for s in answer_signals if s in resp_lower)
        return min(1.0, 0.4 + hits * 0.2)

    if task_type == "diagnosis":
        # Should have disease names and confidence scores
        has_disease     = bool(re.search(r"diagnosis|disease|condition", resp_lower))
        has_confidence  = bool(re.search(r"confidence|%|likely|probability", resp_lower))
        has_hospitals   = "hospital" in resp_lower
        has_tests       = bool(re.search(r"test|recommend|blood", resp_lower))
        return (
            (0.3 if has_disease    else 0.0)
            + (0.2 if has_confidence else 0.0)
            + (0.25 if has_hospitals  else 0.0)
            + (0.25 if has_tests      else 0.0)
        )

    return 0.5


def _score_factual_consistency_rule(response: str, query: Dict) -> float:
    """Check internal consistency of the response."""
    expected = query.get("expected", {})
    top_diseases = [d.lower() for d in (expected.get("top_disease") or [])]
    min_conf = expected.get("min_confidence")
    resp_lower = response.lower()

    score = 1.0

    # Check if any expected disease is mentioned
    if top_diseases:
        mentioned = any(d in resp_lower for d in top_diseases)
        if not mentioned:
            score -= 0.35   # Expected disease not in response

    # Check confidence range
    if min_conf is not None:
        conf_matches = re.findall(r"(\d{1,3})\s*%", response)
        if conf_matches:
            confs = [float(c) / 100 for c in conf_matches]
            if max(confs) < min_conf:
                score -= 0.2

    # Penalise contradictory phrases
    contradictions = [
        ("emergency", "not urgent"),
        ("not a medical", "diagnosis"),
    ]
    for pos, neg in contradictions:
        if pos in resp_lower and neg in resp_lower:
            score -= 0.2

    return max(0.0, score)


def _score_hallucination_free(response: str, query: Dict) -> float:
    """Return 1.0 if no hallucination flags, lower if flags detected."""
    if not _GUARD_OK:
        return 0.8  # Conservative default when guard unavailable

    flagged, flags = detect_hallucination(response, query.get("query", ""))
    if not flagged:
        return 1.0
    # Deduct per flag, capped at 0
    return max(0.0, 1.0 - 0.15 * len(flags))


# ── LLM-as-judge scorer ───────────────────────────────────────────────────────

_LLM_JUDGE_TEMPLATE = """You are an evaluation judge for a medical AI assistant.

Query: {query}
Expected type: {expected_type}

Response to evaluate:
---
{response}
---

Score this response on each dimension from 0.00 to 1.00.
Be strict: 1.0 = perfect, 0.5 = mediocre, 0.0 = completely wrong.

Return ONLY valid JSON:
{{
  "relevance":           0.00,
  "factual_consistency": 0.00,
  "task_completion":     0.00,
  "reasoning":           "1-2 sentence explanation"
}}
"""


def _llm_judge(response: str, query: Dict) -> Optional[Dict[str, float]]:
    """Ask the LLM to score the response. Returns None if LLM unavailable."""
    try:
        from app.config import llm_call
    except ImportError:
        return None

    prompt = _LLM_JUDGE_TEMPLATE.format(
        query         = query.get("query", ""),
        expected_type = query.get("expected", {}).get("type", "unknown"),
        response      = response[:2000],   # truncate to save tokens
    )
    try:
        raw = llm_call(prompt, agent="eval_judge")
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw[start:end])
            return {
                "relevance":           float(data.get("relevance", 0.5)),
                "factual_consistency": float(data.get("factual_consistency", 0.5)),
                "task_completion":     float(data.get("task_completion", 0.5)),
                "reasoning":           str(data.get("reasoning", "")),
            }
    except Exception as e:
        _log.warning(f"LLM judge failed: {e}")
    return None


# ── Harness ───────────────────────────────────────────────────────────────────

class EvaluationHarness:
    """Score individual or batches of agent responses.

    Parameters
    ----------
    use_llm_judge : bool
        If True, use the LLM as an additional judge (costs tokens).
        Rule-based scores are always computed regardless.
    llm_judge_weight : float
        Weight of LLM-judge scores vs rule-based (0 = rules only, 1 = LLM only).
    """

    def __init__(self, use_llm_judge: bool = False, llm_judge_weight: float = 0.4):
        self.use_llm_judge    = use_llm_judge
        self.llm_judge_weight = llm_judge_weight

    def score_response(self, query: Dict, response: str) -> Dict[str, Any]:
        """Score a single response against its query spec.

        Returns
        -------
        dict with keys:
          query_id, category, scores (per dimension), weighted_total,
          passed (bool), flags, llm_reasoning
        """
        t0 = time.perf_counter()

        # ── Rule-based scores ─────────────────────────────────────────────
        rule_scores = {
            "relevance":           _score_relevance_rule(response, query),
            "factual_consistency": _score_factual_consistency_rule(response, query),
            "task_completion":     _score_task_completion_rule(response, query),
            "hallucination_free":  _score_hallucination_free(response, query),
        }

        final_scores = dict(rule_scores)
        llm_reasoning = ""

        # ── LLM-as-judge (optional) ───────────────────────────────────────
        if self.use_llm_judge:
            llm_data = _llm_judge(response, query)
            if llm_data:
                w = self.llm_judge_weight
                for dim in ("relevance", "factual_consistency", "task_completion"):
                    if dim in llm_data:
                        final_scores[dim] = (
                            (1 - w) * rule_scores[dim] + w * llm_data[dim]
                        )
                llm_reasoning = llm_data.get("reasoning", "")

        # ── Weighted total ────────────────────────────────────────────────
        total = sum(final_scores[dim] * WEIGHTS[dim] for dim in WEIGHTS)

        # ── Hallucination flags ───────────────────────────────────────────
        hal_flagged = False
        hal_flags: List[str] = []
        if _GUARD_OK:
            hal_flagged, hal_flags = detect_hallucination(response, query.get("query", ""))

        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "query_id":      query.get("id", "unknown"),
            "category":      query.get("category", ""),
            "query":         query.get("query", ""),
            "scores":        {k: round(v, 3) for k, v in final_scores.items()},
            "weighted_total": round(total, 3),
            "passed":        total >= 0.60,
            "hallucination_flagged": hal_flagged,
            "hallucination_flags":   hal_flags,
            "llm_reasoning": llm_reasoning,
            "duration_ms":   duration_ms,
        }

    def run_batch(
        self,
        queries:   List[Dict],
        run_fn:    Callable[[str], str],
        stop_on_error: bool = False,
    ) -> Dict[str, Any]:
        """Run all queries through `run_fn` and score each result.

        Parameters
        ----------
        queries    : list of query dicts (from test_queries.py)
        run_fn     : callable that takes a query string and returns response text
        stop_on_error : if True, raise on any exception; otherwise continue

        Returns
        -------
        dict: {results, summary, failed_cases}
        """
        results: List[Dict]  = []
        errors:  List[Dict]  = []

        for i, query in enumerate(queries, 1):
            _log.info(f"Running query {i}/{len(queries)}: {query['id']}")
            try:
                response = run_fn(query["query"])
                score    = self.score_response(query, response)
                results.append(score)
            except Exception as exc:
                if stop_on_error:
                    raise
                errors.append({"query_id": query.get("id"), "error": str(exc)})
                _log.error(f"Query {query.get('id')} raised: {exc}")

        return {
            "results":      results,
            "errors":       errors,
            "summary":      self._summarise(results),
        }

    @staticmethod
    def _summarise(results: List[Dict]) -> Dict[str, Any]:
        if not results:
            return {}

        total    = len(results)
        passed   = sum(1 for r in results if r.get("passed"))
        scores   = [r["weighted_total"] for r in results]

        per_category: Dict[str, Dict] = {}
        for r in results:
            cat = r.get("category", "unknown")
            if cat not in per_category:
                per_category[cat] = {"total": 0, "passed": 0, "scores": []}
            per_category[cat]["total"] += 1
            per_category[cat]["scores"].append(r["weighted_total"])
            if r.get("passed"):
                per_category[cat]["passed"] += 1

        cat_summary = {}
        for cat, d in per_category.items():
            cat_summary[cat] = {
                "total":   d["total"],
                "pass_rate": round(d["passed"] / d["total"], 3),
                "avg_score": round(sum(d["scores"]) / len(d["scores"]), 3),
            }

        hal_count = sum(1 for r in results if r.get("hallucination_flagged"))

        return {
            "total_queries":        total,
            "passed":               passed,
            "failed":               total - passed,
            "pass_rate_pct":        round(100 * passed / total, 1),
            "avg_score":            round(sum(scores) / total, 3),
            "min_score":            round(min(scores), 3),
            "max_score":            round(max(scores), 3),
            "hallucinations_flagged": hal_count,
            "per_category":         cat_summary,
        }
