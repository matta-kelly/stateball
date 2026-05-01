"""Small utility functions for the eval runner."""

from __future__ import annotations


def bases_str(bases: int) -> str:
    """Convert bitmask to readable base state like '1B 3B' or '---'."""
    parts = []
    if bases & 1:
        parts.append("1B")
    if bases & 2:
        parts.append("2B")
    if bases & 4:
        parts.append("3B")
    return " ".join(parts) if parts else "---"


def half_str(half: int) -> str:
    return "Bot" if half else "Top"


def entry_phase(inning: int) -> str:
    if inning <= 3:
        return "early"
    elif inning <= 6:
        return "mid"
    return "late"


def reconstruct_scores(state_half: int, run_diff: int, home_score: int, away_score: int):
    """Reconstruct entry scores from run_diff and state."""
    if state_half:
        home_lead = run_diff
    else:
        home_lead = -run_diff
    return home_lead
