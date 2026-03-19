import os
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
 
from sentence_transformers import SentenceTransformer
from decision_engine   import decide, generate_message
from uncertainty_module import compute_uncertainty_batch
 
# ==============================================================================
# CONFIG
# ==============================================================================
 
TEST_CSV       = "../dataset/Test_data.csv"
MODEL_NAME     = "all-MiniLM-L6-v2"
 
# ==============================================================================
# 1.  LOAD TEST DATA
# ==============================================================================
 
print("=" * 60)
print("STEP 1 — Loading test data")
print("=" * 60)
 
df = pd.read_csv(TEST_CSV, encoding="utf-8")
print(f"Test set shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
 
# Verify no label columns leaked into test set
assert "emotional_state" not in df.columns, \
    "emotional_state found in test data — check your CSV!"
assert "intensity" not in df.columns, \
    "intensity found in test data — check your CSV!"
 
# ==============================================================================
# 2.  LOAD TRAINED ARTEFACTS
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 2 — Loading trained models and preprocessor")
print("=" * 60)
 
with open("model_state.pkl", "rb") as f:
    clf_state = pickle.load(f)
 
with open("model_intensity_clf.pkl", "rb") as f:
    clf_intensity = pickle.load(f)
 
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
 
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
 
print("All models loaded ✓")
print(f"Emotional state classes: {list(le.classes_)}")
 
# ==============================================================================
# 3.  FEATURE ENGINEERING ON TEST DATA
#     Must apply EXACTLY the same transformations as training.
#     Preprocessor is already fitted — just call transform(), NOT fit_transform()
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 3 — Feature engineering on test data")
print("=" * 60)
 
# ── 3a. Text embeddings ───────────────────────────────────────────────────────
embedding_model = SentenceTransformer(MODEL_NAME)
 
texts = df["journal_text"].fillna("").astype(str).tolist()
df["text_word_count"] = df["journal_text"].fillna("").apply(
    lambda x: len(str(x).split())
)
 
print(f"Embedding {len(texts)} test texts...")
embeddings = embedding_model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"Embeddings shape: {embeddings.shape}")
 
# ── 3b. Metadata preprocessing ───────────────────────────────────────────────
# CRITICAL: use transform() not fit_transform()
# fit_transform() would re-learn the encoding from test data → data leakage
metadata_features = preprocessor.transform(df)
print(f"Metadata features shape: {metadata_features.shape}")
 
# ── 3c. Fuse ─────────────────────────────────────────────────────────────────
X_test = np.hstack([embeddings, metadata_features])
print(f"Final test feature matrix: {X_test.shape}")
 
# ==============================================================================
# 4.  PREDICT EMOTIONAL STATE
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 4 — Predicting emotional state")
print("=" * 60)
 
state_proba  = clf_state.predict_proba(X_test)       # shape (N, n_classes)
state_enc    = clf_state.predict(X_test)              # shape (N,) — int labels
state_labels = le.inverse_transform(state_enc)        # back to strings
 
print(f"State distribution in predictions:")
unique, counts = np.unique(state_labels, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {cls:15s}: {cnt}")
 
# ==============================================================================
# 5.  PREDICT INTENSITY
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 5 — Predicting intensity")
print("=" * 60)
 
intensity_0based = clf_intensity.predict(X_test)
intensity_pred   = intensity_0based + 1               # shift back to 1–5
intensity_proba  = clf_intensity.predict_proba(X_test)
 
print(f"Intensity distribution in predictions:")
unique_i, counts_i = np.unique(intensity_pred, return_counts=True)
for val, cnt in zip(unique_i, counts_i):
    print(f"  intensity={val}: {cnt}")
 
# ==============================================================================
# 6.  UNCERTAINTY MODULE
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 6 — Computing uncertainty")
print("=" * 60)
 
# Extract raw metadata values for conflict detection
# These come from original test dataframe — not the encoded version
stress_vals = pd.to_numeric(df["stress_level"], errors="coerce").fillna(3).values
energy_vals = pd.to_numeric(df["energy_level"], errors="coerce").fillna(3).values
word_counts = df["text_word_count"].values
 
uncertainty = compute_uncertainty_batch(
    state_probas     = state_proba,
    word_counts      = word_counts,
    stress_values    = stress_vals,
    energy_values    = energy_vals,
    predicted_states = list(state_labels),
)
 
confidence_scores = uncertainty["confidence"]
uncertain_flags   = uncertainty["uncertain_flag"]
uncertainty_reasons = uncertainty["reasons"]
 
n_uncertain = uncertain_flags.sum()
print(f"Uncertain predictions: {n_uncertain} / {len(df)} "
      f"({n_uncertain/len(df)*100:.1f}%)")
print(f"Mean confidence: {confidence_scores.mean():.4f}")
 
# ==============================================================================
# 7.  DECISION ENGINE
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 7 — Decision engine (what + when)")
print("=" * 60)
 
# Extract time_of_day from test data
# Handle possible missing values with a safe default
time_of_day_vals = df["time_of_day"].fillna("morning").astype(str).values
 
what_to_do_list  = []
when_to_do_list  = []
messages_list    = []
reasons_list     = []
 
for i in range(len(df)):
    decision = decide(
        predicted_state = state_labels[i],
        intensity       = int(intensity_pred[i]),
        stress          = int(stress_vals[i]),
        energy          = int(energy_vals[i]),
        time_of_day     = time_of_day_vals[i],
        confidence      = float(confidence_scores[i]),
    )
 
    what_to_do_list.append(decision["what_to_do"])
    when_to_do_list.append(decision["when_to_do"])
    reasons_list.append(decision["decision_reason"])
 
    msg = generate_message(state_labels[i], decision["what_to_do"])
    messages_list.append(msg)
 
print(f"What-to-do distribution:")
what_series = pd.Series(what_to_do_list)
print(what_series.value_counts().to_string())
 
print(f"\nWhen-to-do distribution:")
when_series = pd.Series(when_to_do_list)
print(when_series.value_counts().to_string())
 
# ==============================================================================
# 8.  ASSEMBLE predictions.csv
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 8 — Assembling predictions.csv")
print("=" * 60)
 
predictions = pd.DataFrame({
    "id":                  df["id"].values,
    "predicted_state":     state_labels,
    "predicted_intensity": intensity_pred,
    "confidence":          confidence_scores.round(4),
    "uncertain_flag":      uncertain_flags,
    "what_to_do":          what_to_do_list,
    "when_to_do":          when_to_do_list,
    "supportive_message":  messages_list,   # bonus column
    "uncertainty_reason":  uncertainty_reasons,  # helpful for review
})
 
predictions.to_csv("predictions.csv", index=False, encoding="utf-8")
print(f"Saved → predictions.csv  ({len(predictions)} rows)")
 
# ==============================================================================
# 9.  SANITY CHECKS
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 9 — Sanity checks")
print("=" * 60)
 
valid_states = set(le.classes_)
valid_what   = {
    "box_breathing", "journaling", "grounding", "deep_work",
    "yoga", "sound_therapy", "light_planning", "rest", "movement", "pause"
}
valid_when   = {
    "now", "within_15_min", "later_today", "tonight", "tomorrow_morning"
}
 
assert predictions["predicted_state"].isin(valid_states).all(), \
    "Invalid state in predictions!"
assert predictions["predicted_intensity"].between(1, 5).all(), \
    "Intensity out of [1,5]!"
assert predictions["confidence"].between(0, 1).all(), \
    "Confidence out of [0,1]!"
assert predictions["uncertain_flag"].isin([0, 1]).all(), \
    "uncertain_flag must be 0 or 1!"
assert predictions["what_to_do"].isin(valid_what).all(), \
    "Invalid what_to_do value!"
assert predictions["when_to_do"].isin(valid_when).all(), \
    "Invalid when_to_do value!"
assert len(predictions) == len(df), \
    "Row count mismatch!"
 
print("All predicted_state values valid   ✓")
print("Intensity in [1,5]                 ✓")
print("Confidence in [0,1]                ✓")
print("uncertain_flag in {0,1}            ✓")
print("what_to_do values valid            ✓")
print("when_to_do values valid            ✓")
print("Row count matches test set         ✓")
 
# ==============================================================================
# 10.  PREVIEW
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 10 — Preview")
print("=" * 60)
 
print(predictions[[
    "id", "predicted_state", "predicted_intensity",
    "confidence", "uncertain_flag", "what_to_do", "when_to_do"
]].head(10).to_string(index=False))
 
print("\n" + "=" * 60)