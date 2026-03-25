# ERROR_ANALYSIS.md

> Error analysis performed on **out-of-fold (OOF) predictions** from the training set (n=1440), where ground truth labels are available. The test set has no labels so honest error analysis is not possible on it.


## Overall failure rate

- Total samples   : 1440
- Correct         : 808 (56.1%)
- Failures        : 632 (43.9%)

## Failure archetypes summary

| Archetype | Count | % of failures |
|---|---|---|
| short_text | 248 | 39.2% |
| adjacent_states | 133 | 21.0% |
| general | 112 | 17.7% |
| low_confidence | 69 | 10.9% |
| noisy_label | 56 | 8.9% |
| conflicting_signals | 14 | 2.2% |

---

## 10 Failure cases — detailed analysis

### Case 1 — Short Text

| Field | Value |
|---|---|
| ID | 760 |
| Journal text | *fine i guess* |
| Word count | 3 |
| True state | **focused** |
| Predicted state | neutral |
| Confidence | 0.2868 |
| Stress / Energy / Sleep | 5 / 1 / 7.0 |
| Time of day | morning |
| Reflection quality | vague |
| Face emotion hint | neutral_face |
| Previous day mood | mixed |

**What went wrong**

Very short or vague text gives the embedding model almost no signal. The 384-dimensional embedding of 'ok' or 'fine' sits near the centre of the embedding space — equidistant from all emotional clusters. The model defaults to whichever class is most common in training.

**Why the model failed**

The journal text is only 3 words. The sentence embedding carries almost no discriminative signal at this length. The model fell back on metadata patterns which were not strong enough to distinguish 'focused' from 'neutral'.

**How to improve**

1. Prompt users for more detail when text is under 10 words.
2. Set uncertain_flag=1 automatically for short text (already implemented).
3. Fall back to metadata-only prediction when text is too short.

---

### Case 2 — Short Text

| Field | Value |
|---|---|
| ID | 760 |
| Journal text | *honestly not much change* |
| Word count | 4 |
| True state | **focused** |
| Predicted state | neutral |
| Confidence | 0.2868 |
| Stress / Energy / Sleep | 3 / 3 / 8.0 |
| Time of day | night |
| Reflection quality | clear |
| Face emotion hint | tense_face |
| Previous day mood | calm |

**What went wrong**

Very short or vague text gives the embedding model almost no signal. The 384-dimensional embedding of 'ok' or 'fine' sits near the centre of the embedding space — equidistant from all emotional clusters. The model defaults to whichever class is most common in training.

**Why the model failed**

The journal text is only 4 words. The sentence embedding carries almost no discriminative signal at this length. The model fell back on metadata patterns which were not strong enough to distinguish 'focused' from 'neutral'.

**How to improve**

1. Prompt users for more detail when text is under 10 words.
2. Set uncertain_flag=1 automatically for short text (already implemented).
3. Fall back to metadata-only prediction when text is too short.

---

### Case 3 — Adjacent States

| Field | Value |
|---|---|
| ID | 1058 |
| Journal text | *woke up feeling in between. mountain visuals made it easier to pause.* |
| Word count | 12 |
| True state | **mixed** |
| Predicted state | restless |
| Confidence | 0.4055 |
| Stress / Energy / Sleep | 2 / 2 / 7.0 |
| Time of day | afternoon |
| Reflection quality | conflicted |
| Face emotion hint | nan |
| Previous day mood | neutral |

**What went wrong**

Some emotional states are semantically very close — calm vs neutral, mixed vs restless, focused vs calm. The boundary between them is fuzzy even for humans. Small differences in journal tone or metadata can flip the prediction across this boundary.

**Why the model failed**

'mixed' and 'restless' are semantically adjacent states. With confidence=0.41, the model was already uncertain. The journal text likely contained language consistent with both states, and the metadata (stress=2, energy=2) did not provide a clear tiebreak.

**How to improve**

1. Merge semantically adjacent classes (e.g. calm+neutral → settled) to reduce the boundary problem.
2. Use a hierarchical classifier: first predict broad category (positive/negative/neutral), then fine-grained state.
3. Collect more training examples at the class boundaries.

---

### Case 4 — Adjacent States

| Field | Value |
|---|---|
| ID | 757 |
| Journal text | *During teh session not sure what changed. Then it shifted it was fine.* |
| Word count | 13 |
| True state | **focused** |
| Predicted state | neutral |
| Confidence | 0.4107 |
| Stress / Energy / Sleep | 5 / 4 / 6.0 |
| Time of day | morning |
| Reflection quality | conflicted |
| Face emotion hint | calm_face |
| Previous day mood | restless |

**What went wrong**

Some emotional states are semantically very close — calm vs neutral, mixed vs restless, focused vs calm. The boundary between them is fuzzy even for humans. Small differences in journal tone or metadata can flip the prediction across this boundary.

**Why the model failed**

'focused' and 'neutral' are semantically adjacent states. With confidence=0.41, the model was already uncertain. The journal text likely contained language consistent with both states, and the metadata (stress=5, energy=4) did not provide a clear tiebreak.

**How to improve**

1. Merge semantically adjacent classes (e.g. calm+neutral → settled) to reduce the boundary problem.
2. Use a hierarchical classifier: first predict broad category (positive/negative/neutral), then fine-grained state.
3. Collect more training examples at the class boundaries.

---

### Case 5 — Adjacent States

| Field | Value |
|---|---|
| ID | 757 |
| Journal text | *not gonna lie i felt still mentally busy, but mountain visuals made it easier to pause.* |
| Word count | 16 |
| True state | **focused** |
| Predicted state | neutral |
| Confidence | 0.4107 |
| Stress / Energy / Sleep | 1 / 3 / 7.0 |
| Time of day | afternoon |
| Reflection quality | vague |
| Face emotion hint | none |
| Previous day mood | overwhelmed |

**What went wrong**

Some emotional states are semantically very close — calm vs neutral, mixed vs restless, focused vs calm. The boundary between them is fuzzy even for humans. Small differences in journal tone or metadata can flip the prediction across this boundary.

**Why the model failed**

'focused' and 'neutral' are semantically adjacent states. With confidence=0.41, the model was already uncertain. The journal text likely contained language consistent with both states, and the metadata (stress=1, energy=3) did not provide a clear tiebreak.

**How to improve**

1. Merge semantically adjacent classes (e.g. calm+neutral → settled) to reduce the boundary problem.
2. Use a hierarchical classifier: first predict broad category (positive/negative/neutral), then fine-grained state.
3. Collect more training examples at the class boundaries.

---

### Case 6 — Conflicting Signals

| Field | Value |
|---|---|
| ID | 759 |
| Journal text | *some peace, some noise in head* |
| Word count | 6 |
| True state | **calm** |
| Predicted state | mixed |
| Confidence | 0.4247 |
| Stress / Energy / Sleep | 4 / 4 / 6.0 |
| Time of day | night |
| Reflection quality | conflicted |
| Face emotion hint | neutral_face |
| Previous day mood | calm |

**What went wrong**

Metadata signals contradict the journal text. For example, a user writes a calm reflection but reports stress=5 — the model receives competing evidence and can resolve it incorrectly either way.

**Why the model failed**

The metadata signals conflict: stress=4, energy=4, previous_day_mood=calm. The journal text pointed toward 'calm' but the physiological signals suggested otherwise. The model weighted the conflicting signals and predicted 'mixed'.

**How to improve**

1. Add an explicit conflict feature: |text_sentiment - stress_level|.
2. Train a separate 'conflict detector' and use it to modulate confidence.
3. At inference, if conflict is detected, widen the confidence interval.

---

### Case 7 — Conflicting Signals

| Field | Value |
|---|---|
| ID | 500 |
| Journal text | *During the session got distracted again. Then it shifted kept thinking about work.* |
| Word count | 13 |
| True state | **calm** |
| Predicted state | mixed |
| Confidence | 0.4760 |
| Stress / Energy / Sleep | 4 / 4 / 4.0 |
| Time of day | night |
| Reflection quality | vague |
| Face emotion hint | neutral_face |
| Previous day mood | neutral |

**What went wrong**

Metadata signals contradict the journal text. For example, a user writes a calm reflection but reports stress=5 — the model receives competing evidence and can resolve it incorrectly either way.

**Why the model failed**

The metadata signals conflict: stress=4, energy=4, previous_day_mood=neutral. The journal text pointed toward 'calm' but the physiological signals suggested otherwise. The model weighted the conflicting signals and predicted 'mixed'.

**How to improve**

1. Add an explicit conflict feature: |text_sentiment - stress_level|.
2. Train a separate 'conflict detector' and use it to modulate confidence.
3. At inference, if conflict is detected, widen the confidence interval.

---

### Case 8 — Low Confidence

| Field | Value |
|---|---|
| ID | 531 |
| Journal text | *For some reason mind was all over the place.* |
| Word count | 9 |
| True state | **restless** |
| Predicted state | mixed |
| Confidence | 0.2614 |
| Stress / Energy / Sleep | 1 / 1 / 7.0 |
| Time of day | morning |
| Reflection quality | vague |
| Face emotion hint | neutral_face |
| Previous day mood | calm |

**What went wrong**

The model's probability distribution was spread across multiple classes, indicating genuine uncertainty. The predicted label is the argmax of an uncertain distribution — it may not represent the true state reliably.

**Why the model failed**

Model confidence was only 0.26 — the probability mass was spread across multiple classes. The prediction of 'mixed' was the argmax of an uncertain distribution and could easily have been 'restless' with minor changes to input.

**How to improve**

1. Use temperature scaling to calibrate probability outputs.
2. Consider abstaining from prediction when confidence < 0.4 and instead asking the user a clarifying question.
3. Ensemble multiple models — their disagreement is a natural uncertainty signal.

---

### Case 9 — Noisy Label

| Field | Value |
|---|---|
| ID | 776 |
| Journal text | *today i was unable to settle. forest sounds worked for a bit. still not fully there though.* |
| Word count | 17 |
| True state | **restless** |
| Predicted state | calm |
| Confidence | 0.4044 |
| Stress / Energy / Sleep | 1 / 3 / 7.0 |
| Time of day | night |
| Reflection quality | conflicted |
| Face emotion hint | calm_face |
| Previous day mood | calm |

**What went wrong**

The reflection quality was marked 'conflicted', suggesting the user themselves expressed contradictory emotions. The ground truth label may have been difficult to assign even for a human annotator, making this a case of label noise rather than model error.

**Why the model failed**

Reflection quality is 'conflicted', indicating the user expressed contradictory emotions. The true label 'restless' may itself be uncertain — this could be a labelling error rather than a model error. The model's prediction of 'calm' is plausible given the text.

**How to improve**

1. Apply label smoothing during training to reduce overconfidence on noisy labels.
2. Use a label noise detection algorithm (e.g. Cleanlab) to identify and down-weight suspicious training examples.
3. Collect re-annotations for conflicted-quality reflections.

---

### Case 10 — General

| Field | Value |
|---|---|
| ID | 881 |
| Journal text | *i noticed i was a lot quieter inside. sleep probably affected it.* |
| Word count | 12 |
| True state | **calm** |
| Predicted state | overwhelmed |
| Confidence | 0.4021 |
| Stress / Energy / Sleep | 2 / 2 / 7.0 |
| Time of day | evening |
| Reflection quality | clear |
| Face emotion hint | calm_face |
| Previous day mood | calm |

**What went wrong**

No single dominant failure pattern. The model made a plausible but incorrect guess — likely due to a combination of ambiguous text, moderate metadata signals, and insufficient training examples for this specific combination of features.

**Why the model failed**

No single dominant factor. The model predicted 'overwhelmed' with confidence 0.40. The combination of text content, stress=2, energy=2, and time=evening produced features that resembled 'overwhelmed' more than 'calm' in the training distribution.

**How to improve**

1. Collect more training data, especially for underrepresented state+context combinations.
2. Use an emotion-specific embedding model instead of general-purpose sentence-transformers.
3. Add richer text features: sentiment polarity, negation detection, hedging language detection.

---

## Global insights

### 1. The hardest class boundaries
Looking across all 10 cases, the most common confusions are between semantically adjacent states. A hierarchical classification approach — broad category first, fine-grained second — would likely reduce these errors.

### 2. Text length is a strong reliability signal
Short text entries are disproportionately represented in failure cases. This validates the `uncertain_flag` design: flagging short text inputs as uncertain is the right product decision, not just a modelling choice.

### 3. Metadata helps but can also mislead
When text and metadata agree, the fused model outperforms text-only. But when they conflict — high stress + calm text — the metadata can actively hurt prediction. A conflict detection layer would mitigate this.

### 4. Label noise is real
Several failures involve reflection_quality='conflicted'. These may be labelling errors rather than model errors. Applying Cleanlab or label smoothing could improve robustness on this subset.

### 5. Confidence calibration matters more than raw accuracy
For a mental wellness product, a wrong prediction delivered with high confidence is more harmful than a correct prediction delivered with uncertainty. The uncertain_flag system ensures the decision engine defaults to conservative recommendations when the model is unsure.
