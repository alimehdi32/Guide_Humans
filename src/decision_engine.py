"""
Decision Engine — Part 3
=========================
Pure rule-based logic. No ML here — intentionally.
Takes predicted_state, intensity, stress, energy, time_of_day
and returns what_to_do + when_to_do.

Why rule-based and not ML?
- Interpretable: you can explain every decision to the interviewer
- Controllable: product team can tune rules without retraining
- Robust: works even when model confidence is low
- Safe: predictable behaviour on edge cases
"""

# ==============================================================================
# WHAT_TO_DO options (with intent mapping)
# ==============================================================================
# box_breathing    → calm anxiety, regulate nervous system
# grounding        → anchor to present, reduce overwhelm
# journaling       → process mixed/unclear emotions
# deep_work        → leverage focused/calm state
# light_planning   → gentle structure when energy is moderate
# rest             → recover when tired/drained
# movement         → release restless/tense energy physically
# yoga             → combine movement + mindfulness
# sound_therapy    → passive reset, low effort required
# pause            → do nothing, just observe — for conflicted states

# ==============================================================================
# WHEN_TO_DO options
# ==============================================================================
# now              → urgent, do immediately
# within_15_min    → soon, finish current task first
# later_today      → schedule for later in the day
# tonight          → wind-down activity
# tomorrow_morning → reset overnight, act fresh


def decide(
    predicted_state: str,
    intensity: int,
    stress: int,
    energy: int,
    time_of_day: str,
    confidence: float,
) -> dict:
    """
    Core decision function.

    Parameters
    ----------
    predicted_state : str
        One of: calm, focused, restless, mixed, neutral, overwhelmed
    intensity : int
        1–5 scale (1=very mild, 5=very strong)
    stress : int
        1–5 scale from metadata
    energy : int
        1–5 scale from metadata
    time_of_day : str
        One of: early_morning, morning, afternoon, night
    confidence : float
        Model confidence 0–1. Low confidence → more conservative decisions.

    Returns
    -------
    dict with keys: what_to_do, when_to_do, decision_reason
    """

    state   = predicted_state.lower().strip()
    is_low_confidence = confidence < 0.45

    # ── WHAT TO DO ────────────────────────────────────────────────────────────

    if state == "overwhelmed":
        if intensity >= 4:
            what = "box_breathing"       # high overwhelm → immediate regulation
        elif intensity == 3:
            what = "grounding"           # moderate overwhelm → anchor to present
        else:
            what = "pause"               # mild overwhelm → just observe

    elif state == "anxious" or state == "restless":
        if energy >= 4:
            what = "movement"            # high energy + restless → burn it off
        elif intensity >= 3:
            what = "box_breathing"       # high intensity restless → regulate
        else:
            what = "yoga"                # mild restless → mindful movement

    elif state == "focused":
        if energy >= 3:
            what = "deep_work"           # focused + energy → leverage the state
        else:
            what = "light_planning"      # focused but tired → plan, don't grind

    elif state == "calm":
        if energy >= 4:
            what = "deep_work"           # calm + high energy → best state for work
        elif energy >= 2:
            what = "light_planning"      # calm + moderate energy → plan ahead
        else:
            what = "rest"                # calm + low energy → recover

    elif state == "mixed":
        if stress >= 4:
            what = "journaling"          # mixed + high stress → process on paper
        elif intensity >= 3:
            what = "sound_therapy"       # mixed + high intensity → passive reset
        else:
            what = "pause"               # mild mixed → just observe

    elif state == "neutral":
        if energy >= 3:
            what = "light_planning"      # neutral + energy → gentle structure
        else:
            what = "rest"                # neutral + low energy → recover

    else:
        # Unknown / unseen state — safe default
        what = "pause"

    # Override: if model is uncertain, avoid high-commitment activities
    # deep_work requires sustained focus — wrong call when we're unsure
    if is_low_confidence and what == "deep_work":
        what = "light_planning"

    # ── WHEN TO DO ────────────────────────────────────────────────────────────

    if time_of_day in ("early_morning", "morning"):
        if state == "overwhelmed" and intensity >= 4:
            when = "now"                 # don't start the day overwhelmed
        elif state in ("focused", "calm") and energy >= 3:
            when = "now"                 # morning + good state → start immediately
        elif what in ("box_breathing", "grounding"):
            when = "within_15_min"       # regulation activities → do soon
        else:
            when = "within_15_min"

    elif time_of_day == "afternoon":
        if state == "overwhelmed" and intensity >= 4:
            when = "now"
        elif what == "deep_work" and energy >= 3:
            when = "now"                 # afternoon focus window → use it
        elif what in ("rest", "sound_therapy"):
            when = "later_today"         # rest activities → after work
        else:
            when = "within_15_min"

    elif time_of_day == "night":
        if what in ("deep_work", "movement"):
            when = "tomorrow_morning"    # high-stimulation at night → defer
            what = "rest"                # override: rest is more appropriate
        elif what in ("journaling", "sound_therapy", "yoga"):
            when = "tonight"             # wind-down compatible → do tonight
        elif state == "overwhelmed" and intensity >= 4:
            when = "now"                 # even at night, acute overwhelm → now
        else:
            when = "tonight"

    else:
        # Fallback for missing/unknown time_of_day
        when = "within_15_min"

    # Final override: low confidence → never say "now" for high-commitment tasks
    if is_low_confidence and when == "now" and what == "deep_work":
        when = "within_15_min"

    # ── DECISION REASON  (used in supportive message + error analysis) ────────
    reason = (
        f"state={state}, intensity={intensity}, stress={stress}, "
        f"energy={energy}, time={time_of_day}, confidence={confidence:.2f}"
    )

    return {
        "what_to_do":      what,
        "when_to_do":      when,
        "decision_reason": reason,
    }


# ==============================================================================
# SUPPORTIVE MESSAGE GENERATOR  (Bonus — Part 4)
# ==============================================================================

# Template map: (state, what_to_do) → message
# Written to feel human, not robotic
_MESSAGE_TEMPLATES = {
    ("overwhelmed", "box_breathing"):
        "You're carrying a lot right now. Before anything else, let's slow your "
        "system down. Try 4-7-8 breathing for just 5 minutes — it directly "
        "signals your nervous system to calm. Everything else can wait.",

    ("overwhelmed", "grounding"):
        "Things feel like too much right now, and that's okay. Try the 5-4-3-2-1 "
        "grounding exercise — name 5 things you can see, 4 you can touch, and so on. "
        "It brings you back to the present moment.",

    ("overwhelmed", "pause"):
        "You don't need to fix everything right now. Just pause. Sit with it for "
        "a few minutes before deciding your next move.",

    ("restless", "movement"):
        "Your body has energy that needs somewhere to go. A short walk or 10 minutes "
        "of movement will do more for your focus than pushing through right now.",

    ("restless", "box_breathing"):
        "You seem a bit unsettled. Try box breathing — inhale 4 counts, hold 4, "
        "exhale 4, hold 4. Just two rounds can take the edge off.",

    ("restless", "yoga"):
        "There's a low-level restlessness here. A short yoga flow — even 10 minutes "
        "— can channel that energy and leave you clearer.",

    ("focused", "deep_work"):
        "You're in a good place mentally. This is the time to tackle your most "
        "important work. Protect this window — close distractions and go deep.",

    ("focused", "light_planning"):
        "Your mind is clear but your energy is a bit low. Use this clarity to plan "
        "your next steps — you don't need to execute yet, just map it out.",

    ("calm", "deep_work"):
        "You're calm and energised — a rare combination. Use it. Pick your hardest "
        "task and give it your full attention.",

    ("calm", "light_planning"):
        "You're in a settled state. A good time to review your priorities and plan "
        "the next few hours without pressure.",

    ("calm", "rest"):
        "Your body and mind seem to be asking for rest. Listen to that. Even a "
        "20-minute break now will pay back later.",

    ("mixed", "journaling"):
        "There's a lot going on internally. Writing it out — even just stream of "
        "consciousness for 5 minutes — often brings surprising clarity.",

    ("mixed", "sound_therapy"):
        "Your state feels a bit split right now. Some ambient sound or calming "
        "music can help you reset without requiring much from you.",

    ("mixed", "pause"):
        "It's okay not to have it figured out. Sit quietly for a few minutes "
        "and let things settle before deciding your next step.",

    ("neutral", "light_planning"):
        "You're in a neutral, stable place. A good moment to gently plan your "
        "day without pressure — nothing urgent, just some gentle structure.",

    ("neutral", "rest"):
        "Things feel steady but your energy seems low. Rest is productive too — "
        "a short break now sets you up better for later.",
}

_DEFAULT_MESSAGE = (
    "Take a moment to check in with yourself. Whatever you're feeling is valid. "
    "Start small — one breath, one step, one task at a time."
)


def generate_message(predicted_state: str, what_to_do: str) -> str:
    """
    Returns a short human-like supportive message based on state + recommendation.
    Falls back to a generic message if the combination isn't in the template map.
    """
    state = predicted_state.lower().strip()
    key   = (state, what_to_do)
    return _MESSAGE_TEMPLATES.get(key, _DEFAULT_MESSAGE)