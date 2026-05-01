"""Compiled baserunning table — numpy arrays indexed by integers.

Pre-compiles the JSON baserunning table (string-keyed dict of transition
lists) into flat numpy arrays at Simulator init time. Follows the RE_TABLE
pattern: a static structure indexed by integers, no string formatting or
dict lookups at runtime.

Key space: n_outcomes × 8 bases × 3 outs = n_outcomes * 24 slots.
key_idx = outcome_idx * 24 + bases * 3 + outs

The outcome_labels list (from ModelBundle) defines the outcome → index
mapping. This is the data contract: the model defines which outcomes exist,
and the compilation maps model indices to baserunning table entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledBaserunning:
    """Pre-compiled baserunning table for vectorized resolution.

    All arrays are indexed by key_idx = outcome_idx * 24 + bases * 3 + outs.
    Transition arrays have shape (n_keys, max_transitions).
    """

    cum_probs: np.ndarray       # (n_keys, max_t) float32
    post_bases: np.ndarray      # (n_keys, max_t) int8
    runs_scored: np.ndarray     # (n_keys, max_t) int8
    outs_added: np.ndarray      # (n_keys, max_t) int8
    n_transitions: np.ndarray   # (n_keys,) int16 — 0 = unobserved key
    n_keys: int
    max_transitions: int


def compile_baserunning_table(
    raw_table: dict,
    outcome_labels: list[str],
) -> CompiledBaserunning:
    """Compile a JSON baserunning table into numpy lookup arrays.

    Args:
        raw_table: Loaded baserunning JSON with "transitions" dict.
            Keys are "outcome|bases|outs", values are transition lists.
        outcome_labels: Model's ordered outcome class names. Defines the
            outcome_idx → label mapping used by the batch engine.

    Returns:
        CompiledBaserunning ready for vectorized _batch_resolve_outcomes.
    """
    transitions = raw_table["transitions"]
    n_outcomes = len(outcome_labels)
    n_keys = n_outcomes * 24  # 8 bases × 3 outs per outcome
    label_to_idx = {label: i for i, label in enumerate(outcome_labels)}

    # First pass: find max transitions per key
    max_t = 0
    for key, t_list in transitions.items():
        outcome_str = key.split("|")[0]
        if outcome_str in label_to_idx:
            max_t = max(max_t, len(t_list))

    if max_t == 0:
        max_t = 1  # safety — at least one slot

    # Allocate arrays
    cum_probs = np.zeros((n_keys, max_t), dtype=np.float32)
    post_bases = np.zeros((n_keys, max_t), dtype=np.int8)
    runs_scored = np.zeros((n_keys, max_t), dtype=np.int8)
    outs_added = np.zeros((n_keys, max_t), dtype=np.int8)
    n_transitions = np.zeros(n_keys, dtype=np.int16)

    # Second pass: populate
    populated = 0
    for key, t_list in transitions.items():
        parts = key.split("|")
        outcome_str = parts[0]
        bases = int(parts[1])
        outs = int(parts[2])

        oi = label_to_idx.get(outcome_str)
        if oi is None:
            continue  # outcome not in model's class set

        key_idx = oi * 24 + bases * 3 + outs
        n_t = len(t_list)
        n_transitions[key_idx] = n_t

        cumulative = 0.0
        for j, t in enumerate(t_list):
            cumulative += t["p"]
            cum_probs[key_idx, j] = cumulative
            post_bases[key_idx, j] = t["post_bases"]
            runs_scored[key_idx, j] = t["runs_scored"]
            outs_added[key_idx, j] = t["outs_added"]

        # Fill remaining slots with last transition's values (safety for argmax)
        if n_t < max_t:
            cum_probs[key_idx, n_t:] = 1.0
            post_bases[key_idx, n_t:] = post_bases[key_idx, n_t - 1]
            runs_scored[key_idx, n_t:] = runs_scored[key_idx, n_t - 1]
            outs_added[key_idx, n_t:] = outs_added[key_idx, n_t - 1]

        populated += 1

    logger.info(
        "Compiled baserunning table: %d outcomes × 24 slots = %d keys, "
        "%d populated, max_transitions=%d",
        n_outcomes, n_keys, populated, max_t,
    )

    return CompiledBaserunning(
        cum_probs=cum_probs,
        post_bases=post_bases,
        runs_scored=runs_scored,
        outs_added=outs_added,
        n_transitions=n_transitions,
        n_keys=n_keys,
        max_transitions=max_t,
    )
