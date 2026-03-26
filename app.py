"""
Endpoints:
  POST /predict        — full prediction for one user input
  GET  /health         — sanity check the server is running
  GET  /classes        — list valid emotional state classes

Example request:
  curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "journal_text":      "I feel scattered but the session helped a little.",
      "ambience_type":     "ocean",
      "duration_min":      15,
      "sleep_hours":       6.5,
      "energy_level":      3,
      "stress_level":      4,
      "time_of_day":       "morning",
      "previous_day_mood": "mixed",
      "face_emotion_hint": "neutral_face",
      "reflection_quality":"clear"
    }'
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from src.decision_engine    import decide, generate_message
from src.uncertainty_module import compute_uncertainty_batch

# ==============================================================================
# 1.  LOAD MODELS ONCE AT STARTUP
#     Loading inside each request would be 10–30s per call — never do that
# ==============================================================================

print("Loading models...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open("src/model_state.pkl", "rb") as f:
    clf_state = pickle.load(f)

with open("src/model_intensity_clf.pkl", "rb") as f:
    clf_intensity = pickle.load(f)

with open("src/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("src/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("All models loaded ✓")

# ==============================================================================
# 2.  FLASK APP
# ==============================================================================

app = Flask(__name__)
CORS(app)

# ==============================================================================
# 3.  INPUT SCHEMA
#     Define expected fields, their types, and defaults for missing values.
#     This directly implements Part 9 robustness requirements.
# ==============================================================================

FIELD_SCHEMA = {
    # field_name          : (type,  default,        valid_values or None)
    "journal_text"        : (str,   "",              None),
    "ambience_type"       : (str,   "forest",        ["forest","ocean","rain","mountain","cafe"]),
    "duration_min"        : (float, 15.0,            None),
    "sleep_hours"         : (float, 6.5,             None),   # median from training
    "energy_level"        : (int,   3,               [1,2,3,4,5]),
    "stress_level"        : (int,   3,               [1,2,3,4,5]),
    "time_of_day"         : (str,   "morning",       ["early_morning","morning","afternoon","night"]),
    "previous_day_mood"   : (str,   "neutral",       None),
    "face_emotion_hint"   : (str,   "none",          None),
    "reflection_quality"  : (str,   "clear",         ["vague","conflicted","clear"]),
}


def parse_and_validate(data: dict) -> tuple[dict, list]:
    """
    Parse incoming JSON, apply defaults for missing fields,
    coerce types, and collect warnings for invalid values.

    Returns
    -------
    parsed  : dict — cleaned input ready for inference
    warnings: list of str — non-fatal issues found in input
    """
    parsed   = {}
    warnings = []

    for field, (dtype, default, valid_values) in FIELD_SCHEMA.items():
        raw = data.get(field, None)

        # Missing field — use default
        if raw is None:
            parsed[field] = default
            warnings.append(f"'{field}' missing — using default: {default}")
            continue

        # Type coercion
        try:
            value = dtype(raw)
        except (ValueError, TypeError):
            parsed[field] = default
            warnings.append(
                f"'{field}' = {raw!r} could not be cast to {dtype.__name__} "
                f"— using default: {default}"
            )
            continue

        # Valid values check
        if valid_values is not None and value not in valid_values:
            parsed[field] = default
            warnings.append(
                f"'{field}' = {value!r} not in {valid_values} "
                f"— using default: {default}"
            )
            continue

        parsed[field] = value

    return parsed, warnings


def build_feature_vector(parsed: dict) -> np.ndarray:
    """
    Replicates day1_morning.py feature engineering for a single sample.
    Must be identical to training — same preprocessor, same embedding model.
    """
    # ── Text embedding ────────────────────────────────────────────────────────
    text = parsed["journal_text"]
    if not text or len(text.split()) == 0:
        text = "no reflection provided"   # safe fallback for empty text

    embedding = embedding_model.encode(
        [text],
        normalize_embeddings=True
    )   # shape (1, 384)

    # ── Metadata ──────────────────────────────────────────────────────────────
    row = pd.DataFrame([{
        "sleep_hours"      : parsed["sleep_hours"],
        "duration_min"     : parsed["duration_min"],
        "time_of_day"      : parsed["time_of_day"],
        "reflection_quality": parsed["reflection_quality"],
        "energy_level"     : parsed["energy_level"],
        "stress_level"     : parsed["stress_level"],
        "ambience_type"    : parsed["ambience_type"],
        "face_emotion_hint": parsed["face_emotion_hint"],
        "previous_day_mood": parsed["previous_day_mood"],
    }])

    # CRITICAL: transform() not fit_transform()
    metadata = preprocessor.transform(row)   # shape (1, K)

    # ── Fuse ─────────────────────────────────────────────────────────────────
    X = np.hstack([embedding, metadata])     # shape (1, 384+K)
    return X


# ==============================================================================
# 4.  ROUTES
# ==============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check."""
    return jsonify({
        "status" : "ok",
        "models" : "loaded",
        "classes": list(le.classes_),
    })


@app.route("/classes", methods=["GET"])
def classes():
    """Return valid emotional state classes."""
    return jsonify({
        "emotional_states": list(le.classes_),
        "intensity_range" : [1, 2, 3, 4, 5],
        "what_to_do_options": [
            "box_breathing", "journaling", "grounding", "deep_work",
            "yoga", "sound_therapy", "light_planning", "rest",
            "movement", "pause"
        ],
        "when_to_do_options": [
            "now", "within_15_min", "later_today",
            "tonight", "tomorrow_morning"
        ],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Accepts JSON body with user journal + context signals.
    Returns full prediction: state, intensity, confidence,
    uncertain_flag, what_to_do, when_to_do, supportive_message.
    """
    # ── Parse request ─────────────────────────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    parsed, input_warnings = parse_and_validate(data)

    # ── Build features ────────────────────────────────────────────────────────
    try:
        X = build_feature_vector(parsed)
    except Exception as e:
        return jsonify({"error": f"Feature engineering failed: {str(e)}"}), 500

    # ── Predict state ─────────────────────────────────────────────────────────
    state_proba  = clf_state.predict_proba(X)[0]     # shape (n_classes,)
    state_enc    = clf_state.predict(X)[0]
    state_label  = le.inverse_transform([state_enc])[0]

    # ── Predict intensity ─────────────────────────────────────────────────────
    intensity_0based = clf_intensity.predict(X)[0]
    intensity        = int(intensity_0based) + 1     # back to 1–5

    # ── Uncertainty ───────────────────────────────────────────────────────────
    word_count = len(str(parsed["journal_text"]).split())

    uncertainty = compute_uncertainty_batch(
        state_probas     = state_proba[np.newaxis, :],   # (1, n_classes)
        word_counts      = np.array([word_count]),
        stress_values    = np.array([parsed["stress_level"]]),
        energy_values    = np.array([parsed["energy_level"]]),
        predicted_states = [state_label],
    )

    confidence      = float(uncertainty["confidence"][0])
    uncertain_flag  = int(uncertainty["uncertain_flag"][0])
    uncertainty_reason = uncertainty["reasons"][0]

    # ── Decision engine ───────────────────────────────────────────────────────
    decision = decide(
        predicted_state = state_label,
        intensity       = intensity,
        stress          = parsed["stress_level"],
        energy          = parsed["energy_level"],
        time_of_day     = parsed["time_of_day"],
        confidence      = confidence,
    )

    message = generate_message(state_label, decision["what_to_do"])

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        "prediction": {
            "predicted_state"    : state_label,
            "predicted_intensity": intensity,
            "confidence"         : round(confidence, 4),
            "uncertain_flag"     : uncertain_flag,
            "uncertainty_reason" : uncertainty_reason,
        },
        "recommendation": {
            "what_to_do"         : decision["what_to_do"],
            "when_to_do"         : decision["when_to_do"],
            "supportive_message" : message,
        },
        "debug": {
            "decision_reason"    : decision["decision_reason"],
            "input_warnings"     : input_warnings,   # flags missing/invalid fields
            "word_count"         : word_count,
            "all_class_probas"   : {
                cls: round(float(prob), 4)
                for cls, prob in zip(le.classes_, state_proba)
            },
        }
    }

    return jsonify(response), 200


# ==============================================================================
# 5.  RUN
# ==============================================================================

if __name__ == "__main__":
    print("\nStarting Guide Humans API...")
    print("  Health check : http://localhost:5000/health")
    print("  Predict      : POST http://localhost:5000/predict")
    print("  Classes      : http://localhost:5000/classes\n")
    app.run(debug=True, host="0.0.0.0", port=5000)