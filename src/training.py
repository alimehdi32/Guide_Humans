import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
matplotlib.use("Agg")
warnings.filterwarnings("ignore")
 
from sklearn.model_selection   import StratifiedKFold, cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing     import LabelEncoder
from sklearn.linear_model      import Ridge
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, f1_score
)
from xgboost import XGBClassifier, XGBRegressor
 
PLOTS_DIR = "plots"

print(sklearn.__version__)
# ==============================================================================
# 1.  LOAD ARTEFACTS FROM DAY 1 MORNING
# ==============================================================================
 
print("=" * 60)
print("STEP 1 — Loading saved artefacts")
print("=" * 60)
 
with open("feature_matrix.pkl", "rb") as f:
    fm = pickle.load(f)
 
X             = fm["X"]               # full matrix: embeddings + metadata
X_text        = fm["embeddings"]      # 384-dim text only  (for ablation)
X_meta        = fm["metadata"]        # metadata only      (for ablation)
feature_names = fm["feature_names"]   # metadata column names
is_short_text = fm["is_short_text"]   # bool flag, used later for uncertainty
 
with open("targets.pkl", "rb") as f:
    tgt = pickle.load(f)
 
y_state     = tgt["y_state"]      # string labels e.g. 'calm', 'anxious'
y_intensity = tgt["y_intensity"]  # integers 1–5
ids         = tgt["ids"]
 
print(f"Feature matrix : {X.shape}")
print(f"y_state unique : {np.unique(y_state)}")
print(f"y_intensity    : min={y_intensity.min()}  max={y_intensity.max()}")
 
# ==============================================================================
# 2.  ENCODE STRING LABELS → INTEGERS  (XGBoost needs numeric targets)
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 2 — Label encoding emotional_state")
print("=" * 60)
 
le = LabelEncoder()
y_state_enc = le.fit_transform(y_state)   # e.g. calm→0, focused→1, ...
 
print("Class mapping:")
for idx, cls in enumerate(le.classes_):
    count = (y_state == cls).sum()
    print(f"  {idx} → {cls:15s}  (n={count})")
 
# ==============================================================================
# 3.  CROSS-VALIDATION SETUP
# ==============================================================================
 
# StratifiedKFold preserves class proportions in each fold
# — important because emotional_state may be imbalanced
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
# ==============================================================================
# 4.  PART 1 — EMOTIONAL STATE CLASSIFIER
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 3 — Part 1: Emotional State Classifier (XGBClassifier)")
print("=" * 60)
 
n_classes = len(le.classes_)
 
clf_state = XGBClassifier(
    n_estimators      = 500,
    max_depth         = 5,
    learning_rate     = 0.1,
    subsample         = 0.7,
    colsample_bytree  = 0.6,
    min_child_weight  = 5,
    # Handle class imbalance automatically
    # 'balanced' weights minority classes higher
    # Note: XGBoost doesn't have class_weight='balanced' like sklearn,
    # so we use scale_pos_weight only for binary; for multiclass we rely
    # on sample_weight passed in fit() — computed below
    use_label_encoder = False,
    eval_metric       = "mlogloss",
    random_state      = 42,
    n_jobs            = -1,
)
 
# Compute per-sample weights to handle class imbalance
# (equivalent to class_weight='balanced' in sklearn)
class_counts  = np.bincount(y_state_enc)
class_weights = len(y_state_enc) / (n_classes * class_counts)
sample_weights = class_weights[y_state_enc]
 
print("Training with 5-fold cross-validation...")
 
# cross_val_predict gives out-of-fold predictions — honest evaluation
y_state_pred_cv = cross_val_predict(
    clf_state, X, y_state_enc,
    cv         = CV,
    params = {"sample_weight": sample_weights},
    n_jobs     = -1,
)
 
# cross_val_predict uses clf_state as a blueprint only —
# it clones it internally and discards those clones after evaluation.
# clf_state itself remains unfitted after this call.
y_state_proba_cv = cross_val_predict(
    clf_state, X, y_state_enc,
    cv         = CV,
    method     = "predict_proba",
    params = {"sample_weight": sample_weights},
    n_jobs     = -1,
)


# Hyperparameter tuning with RandomizedSearchCV 
# param_dist = {
#     "n_estimators":     [200, 300, 500],
#     "max_depth":        [3, 4, 5, 6],
#     "learning_rate":    [0.01, 0.05, 0.1],
#     "subsample":        [0.7, 0.8, 0.9],
#     "colsample_bytree": [0.6, 0.7, 0.8],
#     "min_child_weight": [1, 3, 5],
# }

# search = RandomizedSearchCV(
#     clf_state, param_dist,
#     n_iter     = 30,
#     cv         = CV,
#     scoring    = "f1_macro",
#     random_state = 42,
#     n_jobs     = -1,
# )


# search.fit(X, y_state_enc)
# print("Best params:", search.best_params_)
# print("Best F1:", search.best_score_)


# Now fit the actual production model on 100% of data
clf_state.fit(X, y_state_enc, sample_weight=sample_weights)
 
# Decode predictions back to string labels
y_state_pred_labels = le.inverse_transform(y_state_pred_cv)
 
# ── Metrics ───────────────────────────────────────────────────────────────────
report_state = classification_report(
    y_state, y_state_pred_labels,
    target_names=le.classes_
)
macro_f1 = f1_score(y_state_enc, y_state_pred_cv, average="macro")
 
print(f"\nClassification Report (5-fold OOF):\n{report_state}")
print(f"Macro F1: {macro_f1:.4f}")
print("Saved → results_state.txt")
 
# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_state, y_state_pred_labels, labels=le.classes_)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n_classes))
ax.set_yticks(range(n_classes))
ax.set_xticklabels(le.classes_, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(le.classes_, fontsize=9)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix — emotional_state (5-fold OOF)")
plt.colorbar(im, ax=ax)
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_confusion_matrix.png", dpi=120)
plt.close()
print(f"Saved → {PLOTS_DIR}/05_confusion_matrix.png")
 
# ==============================================================================
# 5.  PART 2 — INTENSITY PREDICTOR
#     Approach: run BOTH regression and classification, compare, pick best
#     Write-up argument: intensity is ordinal (1–5), not purely discrete,
#     so regression captures the "distance" between 1 and 5 better than
#     treating them as unrelated classes.  We validate with both metrics.
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 4 — Part 2: Intensity Predictor")
print("=" * 60)
 
# ── 4a. Ridge Regression ──────────────────────────────────────────────────────
print("\n[4a] Ridge Regression")
 
ridge = Ridge(alpha=1.0)
 
y_intensity_pred_ridge_cv = cross_val_predict(
    ridge, X, y_intensity, cv=CV, n_jobs=-1
)
 
# Clip to valid range and round to nearest integer for discrete metrics
y_ridge_clipped   = np.clip(y_intensity_pred_ridge_cv, 1, 5)
y_ridge_rounded   = np.round(y_ridge_clipped).astype(int)
 
mae_ridge  = mean_absolute_error(y_intensity, y_ridge_clipped)
rmse_ridge = mean_squared_error(y_intensity, y_ridge_clipped) ** 0.5
f1_ridge   = f1_score(y_intensity, y_ridge_rounded, average="macro")
 
print(f"  MAE  : {mae_ridge:.4f}")
print(f"  RMSE : {rmse_ridge:.4f}")
print(f"  Macro F1 (after rounding): {f1_ridge:.4f}")
 
# ── 4b. XGBoost Classifier (treat intensity as 5 classes) ────────────────────
print("\n[4b] XGBClassifier (ordinal classification)")
 
# Shift labels to 0-based for XGBoost: intensity 1–5 → 0–4
y_intensity_0based = y_intensity - 1
 
clf_intensity = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)
 
y_intensity_pred_clf_cv = cross_val_predict(
    clf_intensity, X, y_intensity_0based, cv=CV, n_jobs=-1
)
y_intensity_proba_cv = cross_val_predict(
    clf_intensity, X, y_intensity_0based,
    cv=CV, method="predict_proba", n_jobs=-1
)
 
# Shift predictions back to 1-based
y_intensity_pred_clf_1based = y_intensity_pred_clf_cv + 1
 
mae_clf  = mean_absolute_error(y_intensity, y_intensity_pred_clf_1based)
rmse_clf = mean_squared_error(y_intensity, y_intensity_pred_clf_1based) ** 0.5
f1_clf   = f1_score(y_intensity, y_intensity_pred_clf_1based, average="macro")
 
print(f"  MAE  : {mae_clf:.4f}")
print(f"  RMSE : {rmse_clf:.4f}")
print(f"  Macro F1: {f1_clf:.4f}")
 
# ── 4c. Compare and pick the better model ────────────────────────────────────
print("\n[4c] Comparison:")
print(f"  Ridge  — MAE={mae_ridge:.3f}  RMSE={rmse_ridge:.3f}  F1={f1_ridge:.3f}")
print(f"  XGBClf — MAE={mae_clf:.3f}  RMSE={rmse_clf:.3f}  F1={f1_clf:.3f}")
 
# Choose based on MAE — lower is better for intensity
if mae_clf <= mae_ridge:
    best_intensity_model_name = "XGBClassifier"
    y_intensity_pred_final    = y_intensity_pred_clf_1based
    y_intensity_proba_final   = y_intensity_proba_cv     # shape (N, 5)
    print("  Winner: XGBClassifier")
else:
    best_intensity_model_name = "Ridge"
    y_intensity_pred_final    = y_ridge_rounded
    # Ridge has no predict_proba — approximate with softmax of distances
    # distance from each class center (1–5), then invert and softmax
    dists = np.abs(
        y_ridge_clipped[:, None] - np.arange(1, 6)[None, :]
    )                                                    # shape (N, 5)
    scores = np.exp(-dists)
    y_intensity_proba_final = scores / scores.sum(axis=1, keepdims=True)
    print("  Winner: Ridge Regression")
 
# Save results
intensity_report = (
    f"INTENSITY PREDICTION REPORT (5-fold OOF)\n"
    f"{'=' * 60}\n\n"
    f"Approach: Both regression (Ridge) and classification (XGBoost) were run.\n"
    f"Intensity is treated as ordinal — values 1-5 have inherent distance,\n"
    f"so regression is a natural fit. Classification validates discrete accuracy.\n\n"
    f"Ridge Regression:\n"
    f"  MAE={mae_ridge:.4f}  RMSE={rmse_ridge:.4f}  MacroF1={f1_ridge:.4f}\n\n"
    f"XGBoost Classifier:\n"
    f"  MAE={mae_clf:.4f}  RMSE={rmse_clf:.4f}  MacroF1={f1_clf:.4f}\n\n"
    f"Selected model: {best_intensity_model_name} (lower MAE)\n"
)
 
with open("results_intensity.txt", "w", encoding="utf-8") as f:
    f.write(intensity_report)
print("\nSaved → results_intensity.txt")
 
# ── Scatter plot: actual vs predicted intensity ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title in zip(
    axes,
    [y_ridge_rounded, y_intensity_pred_clf_1based],
    ["Ridge Regression", "XGBClassifier"]
):
    ax.scatter(y_intensity, preds, alpha=0.4, color="steelblue", s=20)
    ax.plot([1, 5], [1, 5], "r--", linewidth=1, label="perfect prediction")
    ax.set_xlabel("Actual intensity")
    ax.set_ylabel("Predicted intensity")
    ax.set_title(f"Intensity: {title}")
    ax.legend()
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_intensity_scatter.png", dpi=120)
plt.close()
print(f"Saved → {PLOTS_DIR}/06_intensity_scatter.png")
 
# ==============================================================================
# 6.  PART 5 — FEATURE IMPORTANCE  (XGBClassifier for state)
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 5 — Part 5: Feature Importance")
print("=" * 60)
 
importances = clf_state.feature_importances_   # shape: (n_features,)
 
# Feature names: 384 embedding dims + metadata names
embedding_names = [f"emb_{i}" for i in range(384)]
all_names       = embedding_names + feature_names  # total = 384 + K
 
# ── Aggregate embedding importance into one bucket ────────────────────────────
# (We can't interpret 384 individual dims, so we sum them)
emb_total_importance  = importances[:384].sum()
meta_importances      = importances[384:]          # one per metadata feature
 
print(f"\nAggregated text embedding importance : {emb_total_importance:.4f}")
print(f"Metadata feature importances:")
 
meta_imp_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": meta_importances,
}).sort_values("importance", ascending=False)
 
for _, row in meta_imp_df.iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")
 
# ── Bar chart: top 15 metadata features + text embedding bucket ───────────────
top15 = meta_imp_df.head(15).copy()
# Add text embedding as one row
text_row = pd.DataFrame([{
    "feature":    "TEXT EMBEDDINGS (sum)",
    "importance": emb_total_importance
}])
plot_df = pd.concat([text_row, top15], ignore_index=True)
plot_df = plot_df.sort_values("importance", ascending=True)
 
fig, ax = plt.subplots(figsize=(9, 6))
colors = [
    "steelblue" if f != "TEXT EMBEDDINGS (sum)" else "coral"
    for f in plot_df["feature"]
]
ax.barh(plot_df["feature"], plot_df["importance"], color=colors)
ax.set_xlabel("Feature importance (XGBoost gain)")
ax.set_title("Feature importance — emotional state model\n(coral = text, blue = metadata)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_feature_importance.png", dpi=120)
plt.close()
print(f"\nSaved → {PLOTS_DIR}/07_feature_importance.png")
 
# ==============================================================================
# 7.  PART 6 — ABLATION STUDY
#     Compare: text-only vs metadata-only vs text+metadata
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 6 — Part 6: Ablation Study")
print("=" * 60)
 
ablation_results = {}
 
for name, X_ablation in [
    ("text_only",      X_text),
    ("metadata_only",  X_meta),
    ("text+metadata",  X),
]:
    print(f"\n  Running: {name}  (shape={X_ablation.shape})")
 
    clf_abl = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = 42,
        n_jobs            = -1,
    )
 
    preds = cross_val_predict(
        clf_abl, X_ablation, y_state_enc, cv=CV, n_jobs=-1
    )
    f1 = f1_score(y_state_enc, preds, average="macro")
    ablation_results[name] = f1
    print(f"  Macro F1 = {f1:.4f}")
 
print("\n  Ablation summary:")
for name, f1 in ablation_results.items():
    print(f"    {name:20s}: Macro F1 = {f1:.4f}")
 
# Find the actual winner
best_name = max(ablation_results, key=ablation_results.get)
worst_name = min(ablation_results, key=ablation_results.get)

f1_text  = ablation_results["text_only"]
f1_meta  = ablation_results["metadata_only"]
f1_fused = ablation_results["text+metadata"]

# Dynamically generate the insight based on what actually happened
if best_name == "text+metadata":
    insight = (
        "Fusion wins: text+metadata outperforms either alone.\n"
        "Contextual signals (sleep, stress, time of day) add meaningful\n"
        "signal beyond what journal text captures alone.\n"
    )
elif best_name == "text_only":
    insight = (
        "Text alone wins: journal reflections contain enough signal\n"
        "that metadata does not add value — possibly because metadata\n"
        "features are noisy or weakly correlated with emotional state.\n"
        "Consider dropping or re-engineering metadata features.\n"
    )
elif best_name == "metadata_only":
    insight = (
        "Metadata alone wins: structured signals (sleep, stress, energy)\n"
        "are more predictive than the raw text embeddings.\n"
        "The embedding model may not be well-suited for this emotional domain.\n"
        "Consider swapping to an emotion-specific embedding model.\n"
    )

# Also flag if fusion is worse than text-only (a common real-world failure)
if best_name != "text+metadata":
    insight += (
        f"\nWARNING: fusion did not help. This suggests the metadata\n"
        f"features may be adding noise rather than signal.\n"
        f"Try removing low-importance metadata features and re-running.\n"
    )

ablation_text = (
    "ABLATION STUDY\n"
    + "=" * 60 + "\n\n"
    + "Comparing three input configurations for emotional_state prediction:\n\n"
    + "\n".join(
        f"  {name:20s}: Macro F1 = {f1:.4f}"
        + (" ← best" if name == best_name else "")
        + (" ← worst" if name == worst_name else "")
        for name, f1 in ablation_results.items()
    )
    + f"\n\nWinner: {best_name}\n\n"
    + insight
)
 
with open("results_ablation.txt", "w", encoding="utf-8") as f:
    f.write(ablation_text)
print("\nSaved → results_ablation.txt")
 
# ==============================================================================
# 8.  FIT FINAL MODELS ON FULL DATASET + SAVE
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 7 — Fitting final models on full data + saving")
print("=" * 60)
 
# State classifier — already fit above (step 3)
# Re-confirming it's trained on full X
 
# Intensity — fit whichever won
ridge.fit(X, y_intensity)
clf_intensity.fit(X, y_intensity_0based)
 
with open("model_state.pkl", "wb") as f:
    pickle.dump(clf_state, f)
print("Saved → model_state.pkl")
 
with open("model_intensity_clf.pkl", "wb") as f:
    pickle.dump(clf_intensity, f)
print("Saved → model_intensity_clf.pkl")
 
with open("model_intensity_reg.pkl", "wb") as f:
    pickle.dump(ridge, f)
print("Saved → model_intensity_reg.pkl")
 
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Saved → label_encoder.pkl")
 
# Save OOF predictions — needed for error analysis (Day 2) and predictions.csv
oof = pd.DataFrame({
    "id":                    ids,
    "true_state":            y_state,
    "pred_state":            y_state_pred_labels,
    "state_confidence":      y_state_proba_cv.max(axis=1).round(4),
    "true_intensity":        y_intensity,
    "pred_intensity_clf":    y_intensity_pred_clf_1based,
    "pred_intensity_ridge":  y_ridge_rounded,
    "intensity_confidence":  y_intensity_proba_final.max(axis=1).round(4),
    "is_short_text":         is_short_text,
})
oof.to_csv("oof_predictions.csv", index=False)
print("Saved → oof_predictions.csv")
 
# ==============================================================================
# 9.  SANITY CHECK
# ==============================================================================
 
print("\n" + "=" * 60)
print("STEP 8 — Sanity checks")
print("=" * 60)
 
assert len(oof) == len(y_state),           "Row count mismatch in OOF!"
assert oof["state_confidence"].between(0, 1).all(), "Confidence out of [0,1]!"
assert oof["pred_intensity_clf"].between(1, 5).all(), "Intensity out of [1,5]!"
 
print("Row counts consistent         ✓")
print("Confidence scores in [0,1]    ✓")
print("Intensity predictions in [1,5]✓")
 
print("\nSample OOF rows:")
print(oof[["id", "true_state", "pred_state", "state_confidence",
           "true_intensity", "pred_intensity_clf"]].head(5).to_string(index=False))
 
print("\n" + "=" * 60)