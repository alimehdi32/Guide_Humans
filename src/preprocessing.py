import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           
warnings.filterwarnings("ignore")


from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


df=pd.read_csv("../dataset/Train_data.csv")
MODEL_NAME = "all-MiniLM-L6-v2"

# =======================================
# Text Embeddings via sentence-transformers
# =======================================

print("\n" + "=" * 60)
print("STEP 4 — Generating sentence embeddings (local, no API)")
print("=" * 60)
 
model = SentenceTransformer(MODEL_NAME)
 
# Handle missing / very short text gracefully
texts = df["journal_text"].fillna("").astype(str).tolist()
 
# Flag short texts — will be used later for uncertain_flag
df["text_word_count"] = df["journal_text"].fillna("").apply(
    lambda x: len(str(x).split())
)
df["is_short_text"] = df["text_word_count"] <= 5
 
print(f"Embedding {len(texts)} texts with '{MODEL_NAME}'...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True   # cosine-friendly unit vectors
)
print(f"Embedding matrix shape: {embeddings.shape}") 

# ==============================================================================
# 5.  METADATA PREPROCESSING
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 5 — Preprocessing metadata features")
print("=" * 60)
 
# ── Column definitions ────────────────────────────────────────────────────────
 
# Continuous: median imputation for missing sleep_hours, then scale
NUMERIC_COLS = ["sleep_hours", "duration_min"]
 
# Ordinal with known order: encode as integers preserving rank
ORDINAL_COLS = ["time_of_day", "reflection_quality", "energy_level", "stress_level"]
ORDINAL_CATEGORIES = [
    ["early_morning", "morning", "afternoon","evening", "night"],  # time_of_day
    ["vague", "conflicted", "clear"],                              # reflection_quality
    [1, 2, 3, 4, 5],                                               # energy_level
    [1, 2, 3, 4, 5],                                               # stress_level
]
 
# Nominal: no natural order — one-hot encode
# previous_day_mood has states like calm/restless/focused — no clear order
# face_emotion_hint: 'none' is a valid literal value (not NaN), will become its own column
NOMINAL_COLS = ["ambience_type", "face_emotion_hint", "previous_day_mood"]
 
# ── Pipeline for each type ────────────────────────────────────────────────────
 
numeric_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # handles missing sleep_hours
    ("scale",  StandardScaler()),
])
 
ordinal_pipeline = Pipeline([
    ("impute",  SimpleImputer(strategy="most_frequent")),
    ("encode",  OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1               # unseen category → -1
                )),
])
 
nominal_pipeline = Pipeline([
    ("impute",  SimpleImputer(strategy="most_frequent")),
    ("encode",  OneHotEncoder(
                    handle_unknown="ignore",       # unseen category → all zeros
                    sparse_output=False            # return dense array
                )),
])
 
preprocessor = ColumnTransformer(
    transformers=[
        ("num",     numeric_pipeline,  NUMERIC_COLS),
        ("ordinal", ordinal_pipeline,  ORDINAL_COLS),
        ("onehot",  nominal_pipeline,  NOMINAL_COLS),
    ],
    remainder="drop"    # drop 'id', 'journal_text', targets, helper cols
)
 
# ── Fit and transform ─────────────────────────────────────────────────────────
metadata_features = preprocessor.fit_transform(df)
print(f"Metadata feature matrix shape: {metadata_features.shape}")
 
# Print what the one-hot columns look like
ohe = preprocessor.named_transformers_["onehot"]["encode"]
onehot_feature_names = ohe.get_feature_names_out(NOMINAL_COLS).tolist()
all_feature_names = (
    NUMERIC_COLS +
    ORDINAL_COLS +
    onehot_feature_names
)
print(f"Total metadata features: {len(all_feature_names)}")
print(f"  Numeric  ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
print(f"  Ordinal  ({len(ORDINAL_COLS)}): {ORDINAL_COLS}")
print(f"  One-hot  ({len(onehot_feature_names)}): {onehot_feature_names}")
 
# ==============================================================================
# 6.  FUSE EMBEDDINGS + METADATA → FINAL FEATURE MATRIX
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 6 — Fusing embeddings + metadata")
print("=" * 60)
 
# Concatenate horizontally: [384-dim text embedding | metadata features]
X = np.hstack([embeddings, metadata_features])
print(f"Final feature matrix shape: {X.shape}")
print(f"  Text embedding dims : 384")
print(f"  Metadata dims       : {metadata_features.shape[1]}")
print(f"  Total dims          : {X.shape[1]}")
 
# ==============================================================================
# 7.  EXTRACT TARGETS
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 7 — Extracting targets")
print("=" * 60)
 
y_state     = df["emotional_state"].values          # string labels
y_intensity = df["intensity"].values.astype(int)    # 1–5 integers
 
print(f"y_state     shape: {y_state.shape}     unique: {np.unique(y_state)}")
print(f"y_intensity shape: {y_intensity.shape}  unique: {np.unique(y_intensity)}")
 
# Also save the is_short_text flag — used for uncertain_flag later
is_short_text = df["is_short_text"].values
 
# ==============================================================================
# 8.  SAVE ARTEFACTS
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 8 — Saving artefacts")
print("=" * 60)
 
with open("feature_matrix.pkl", "wb") as f:
    pickle.dump({
        "X":               X,
        "embeddings":      embeddings,          # kept separate for ablation study
        "metadata":        metadata_features,   # kept separate for ablation study
        "feature_names":   all_feature_names,
        "is_short_text":   is_short_text,
    }, f)
print("  Saved → feature_matrix.pkl")
 
with open("targets.pkl", "wb") as f:
    pickle.dump({
        "y_state":     y_state,
        "y_intensity": y_intensity,
        "ids":         df["id"].values,
    }, f)
print("  Saved → targets.pkl")
 
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)
print("  Saved → preprocessor.pkl  (reuse this at inference time!)")
 
# ==============================================================================
# 9.  QUICK SANITY CHECK
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 9 — Sanity checks")
print("=" * 60)
 
assert not np.isnan(X).any(),       "NaN found in feature matrix!"
assert not np.isinf(X).any(),       "Inf found in feature matrix!"
assert len(X) == len(y_state),      "Row count mismatch!"
assert len(X) == len(y_intensity),  "Row count mismatch!"
 
print("No NaN / Inf in feature matrix ✓")
print("Row counts consistent ✓")
print(f"\nSample row 0 — state='{y_state[0]}' intensity={y_intensity[0]}")
print(f"  First 5 embedding dims : {X[0, :5].round(4)}")
print(f"  First 5 metadata dims  : {X[0, 384:389].round(4)}")