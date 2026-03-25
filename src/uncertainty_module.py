"""
Uncertainty Module — Part 4
============================
Computes confidence score and uncertain_flag for each prediction.

Three sources of uncertainty are combined:
  1. Model uncertainty     — low max predicted probability
  2. Input uncertainty     — very short or vague text
  3. Signal conflict       — metadata signals contradict each other

A strong system knows when it is unsure.
"""

import numpy as np


# ==============================================================================
# THRESHOLDS  (tunable)
# ==============================================================================

CONFIDENCE_THRESHOLD  = 0.45   # below this → uncertain_flag = 1
SHORT_TEXT_THRESHOLD  = 5      # word count <= this → treat as low-signal input
HIGH_STRESS_THRESHOLD = 4      # stress level considered high
LOW_ENERGY_THRESHOLD  = 2      # energy level considered low
HIGH_ENERGY_THRESHOLD = 4      # energy level considered high


def compute_confidence(state_proba: np.ndarray) -> float:
    """
    Confidence = max predicted probability across all emotional state classes.

    Why max probability?
    - If the model is sure, one class will dominate: e.g. [0.05, 0.82, 0.03, ...]
      → confidence = 0.82
    - If the model is confused, probabilities spread out: e.g. [0.18, 0.20, 0.17, ...]
      → confidence = 0.20 (very uncertain)

    Parameters
    ----------
    state_proba : np.ndarray
        1D array of class probabilities for one sample, shape (n_classes,)

    Returns
    -------
    float : confidence in [0, 1]
    """
    return float(np.max(state_proba))


def compute_entropy(state_proba: np.ndarray) -> float:
    """
    Shannon entropy of the probability distribution.
    High entropy = model is spread across many classes = uncertain.
    Low entropy  = model is concentrated on one class = confident.

    Normalised to [0, 1] by dividing by log(n_classes).
    """
    n = len(state_proba)
    # Clip to avoid log(0)
    proba = np.clip(state_proba, 1e-10, 1.0)
    raw_entropy = -np.sum(proba * np.log(proba))
    max_entropy = np.log(n)                        # entropy of uniform distribution
    return float(raw_entropy / max_entropy)        # normalised 0–1


def detect_signal_conflict(
    stress: float,
    energy: float,
    predicted_state: str,
) -> bool:
    """
    Detects when metadata signals contradict the predicted emotional state.

    Conflict examples:
      - Predicted 'calm' but stress=5 and energy=1  → contradictory
      - Predicted 'focused' but stress=5             → suspicious
      - Predicted 'overwhelmed' but stress=1         → suspicious
      - High energy + high stress together           → ambiguous state

    Returns True if a conflict is detected.
    """
    state = predicted_state.lower().strip()

    conflicts = [
        # Calm predicted but physiological signals say otherwise
        state == "calm" and stress >= HIGH_STRESS_THRESHOLD,

        # Focused predicted but very high stress (hard to focus under high stress)
        state == "focused" and stress >= 5,

        # Overwhelmed predicted but stress is actually low
        state == "overwhelmed" and stress <= 1,

        # High energy + high stress → body is activated, state is ambiguous
        stress >= HIGH_STRESS_THRESHOLD and energy >= HIGH_ENERGY_THRESHOLD,

        # Very low energy + restless predicted → physically contradictory
        state == "restless" and energy <= LOW_ENERGY_THRESHOLD,
    ]

    return any(conflicts)


def compute_uncertain_flag(
    confidence: float,
    word_count: int,
    stress: float,
    energy: float,
    predicted_state: str,
) -> int:
    """
    Sets uncertain_flag = 1 if ANY of these conditions are true:
      1. Confidence below threshold
      2. Text is very short (low-signal input)
      3. Metadata signals conflict with predicted state

    Returns
    -------
    int : 0 (certain) or 1 (uncertain)
    """
    low_confidence   = confidence < CONFIDENCE_THRESHOLD
    short_text       = word_count <= SHORT_TEXT_THRESHOLD
    signal_conflict  = detect_signal_conflict(stress, energy, predicted_state)

    is_uncertain = low_confidence or short_text or signal_conflict

    return int(is_uncertain)


def compute_uncertainty_batch(
    state_probas:      np.ndarray,
    word_counts:       np.ndarray,
    stress_values:     np.ndarray,
    energy_values:     np.ndarray,
    predicted_states:  list,
) -> dict:
    """
    Vectorised version — runs uncertainty computation for all test samples at once.

    Parameters
    ----------
    state_probas     : np.ndarray, shape (N, n_classes)
    word_counts      : np.ndarray, shape (N,)  — words in journal_text
    stress_values    : np.ndarray, shape (N,)
    energy_values    : np.ndarray, shape (N,)
    predicted_states : list of str, length N

    Returns
    -------
    dict with keys:
        confidence     : np.ndarray shape (N,)
        entropy        : np.ndarray shape (N,)
        uncertain_flag : np.ndarray shape (N,) — int 0 or 1
        reasons        : list of str, length N  — human-readable explanation
    """
    N = len(predicted_states)
    confidences    = np.zeros(N)
    entropies      = np.zeros(N)
    uncertain_flags= np.zeros(N, dtype=int)
    reasons        = []

    for i in range(N):
        conf  = compute_confidence(state_probas[i])
        entr  = compute_entropy(state_probas[i])
        flag  = compute_uncertain_flag(
            conf,
            int(word_counts[i]),
            float(stress_values[i]),
            float(energy_values[i]),
            predicted_states[i],
        )

        confidences[i]     = round(conf, 4)
        entropies[i]       = round(entr, 4)
        uncertain_flags[i] = flag

        # Build human-readable reason for why this was flagged
        reason_parts = []
        if conf < CONFIDENCE_THRESHOLD:
            reason_parts.append(f"low_confidence({conf:.2f})")
        if int(word_counts[i]) <= SHORT_TEXT_THRESHOLD:
            reason_parts.append(f"short_text({int(word_counts[i])}_words)")
        if detect_signal_conflict(
            float(stress_values[i]),
            float(energy_values[i]),
            predicted_states[i]
        ):
            reason_parts.append("signal_conflict")

        reasons.append("|".join(reason_parts) if reason_parts else "none")

    return {
        "confidence":      confidences,
        "entropy":         entropies,
        "uncertain_flag":  uncertain_flags,
        "reasons":         reasons,
    }