"""
Run this ONCE to train and save the model.
    python train_model.py
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "data", "city_day.csv")
MDL_DIR  = os.path.join(BASE, "models")
os.makedirs(MDL_DIR, exist_ok=True)

print("="*50)
print("  AQI MODEL TRAINER")
print("="*50)

# ── Load data ──────────────────────────────────────────────
print("\n[1/4] Loading dataset...")
df = pd.read_csv(DATA_CSV)
print(f"  Loaded {len(df):,} rows")

# ── Clean ──────────────────────────────────────────────────
FEATURES = ["PM2.5","PM10","NO2","SO2","CO","O3","NH3","NOx"]
BUCKET_ORDER = ["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]

# Impute missing with city median
df = df.sort_values(["City","Date"])
for col in FEATURES + ["AQI","AQI_Bucket"]:
    if col in df.columns:
        df[col] = df.groupby("City")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df[col].fillna(df[col].median() if col != "AQI_Bucket" else "Moderate")

df2 = df[FEATURES + ["AQI","AQI_Bucket","City"]].dropna()
print(f"  Clean rows: {len(df2):,}")

# ── Train ──────────────────────────────────────────────────
print("\n[2/4] Training models...")
X  = df2[FEATURES].values
yr = df2["AQI"].values
le = LabelEncoder()
le.fit(BUCKET_ORDER)
yc = le.transform(df2["AQI_Bucket"])

Xtr,Xte,ytr,yte = train_test_split(X, yr, test_size=0.2, random_state=42)
_,  _,  cttr,ctte = train_test_split(X, yc, test_size=0.2, random_state=42)

reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
reg.fit(Xtr, ytr)
r2 = r2_score(yte, reg.predict(Xte))
print(f"  Regressor R²   = {r2:.4f}")

clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(Xtr, cttr)
acc = accuracy_score(ctte, clf.predict(Xte))
print(f"  Classifier Acc = {acc:.4f}")

# ── City presets ───────────────────────────────────────────
print("\n[3/4] Building city presets...")
city_stats = df2.groupby("City")[FEATURES+["AQI"]].median().round(2)
city_stats_dict = city_stats.to_dict("index")
print(f"  Cities: {len(city_stats_dict)}")

# ── Save ───────────────────────────────────────────────────
print("\n[4/4] Saving models...")
with open(os.path.join(MDL_DIR, "models.pkl"), "wb") as f:
    pickle.dump({
        "regressor":    reg,
        "classifier":   clf,
        "label_encoder":le,
        "features":     FEATURES,
        "city_stats":   city_stats_dict,
    }, f)

print(f"  Saved → models/models.pkl")
print("\n✅  Done! Now run: python app.py")
