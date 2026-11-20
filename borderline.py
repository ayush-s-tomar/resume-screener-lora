"""
Borderline flag for resume verdicts.

The name-swap boundary test showed that verdicts close to a threshold are unstable. This module marks a verdict as
borderline when the adapter's own score lies within a margin of a verdict boundary, so that an application can send
those resumes to a human instead of deciding them automatically. An output that is not valid JSON, or that has no
usable score, is always flagged when the flag is on.

The default margin is the one recommended by selective_prediction.py (results/selective_v1.json). If that analysis did
not recommend a margin, or the file is missing, the flag is off unless a margin is passed explicitly.
See docs/SELECTIVE_PREDICTION.md for how the margin was chosen and what the flag is worth.
"""
import json
import math
from pathlib import Path

BOUNDARIES = (49.0, 74.5)  # weak up to 48, moderate 50 to 74, strong 75 and above in the training data
RESULTS_FILE = Path(__file__).resolve().parent / "results" / "selective_v1.json"


def recommended_margin(path=RESULTS_FILE):
    """The margin chosen by selective_prediction.py, or None if there is no recommendation."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    margin = data.get("recommended_margin") if isinstance(data, dict) else None
    if isinstance(margin, (int, float)) and not isinstance(margin, bool) and margin > 0:
        return float(margin)
    return None


DEFAULT_MARGIN = recommended_margin()


def distance_to_boundary(score):
    return min(abs(score - b) for b in BOUNDARIES)


def assess(verdict, margin=None):
    """Flag fields for one verdict dict, or None when the flag is off.

    margin=None uses DEFAULT_MARGIN; margin=0 (or any value <= 0) turns the flag off.
    """
    margin = DEFAULT_MARGIN if margin is None else margin
    if not margin or margin <= 0:
        return None
    reason, dist = None, None
    if verdict.get("parse_error"):
        reason = "invalid_output"
    else:
        score = verdict.get("ats_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            reason = "missing_score"
        else:
            dist = distance_to_boundary(float(score))
            if dist < margin:
                reason = "near_boundary"
    return {
        "borderline": reason is not None,
        "borderline_reason": reason,
        "distance_to_boundary": dist,
        "borderline_margin": float(margin),
    }


def annotate(verdict, margin=None):
    """A copy of the verdict with the flag fields added (unchanged copy when the flag is off)."""
    out = dict(verdict)
    info = assess(verdict, margin)
    if info is not None:
        out.update(info)
    return out
