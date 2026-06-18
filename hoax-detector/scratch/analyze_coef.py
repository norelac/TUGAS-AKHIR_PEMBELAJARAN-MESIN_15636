import joblib
from pathlib import Path
import pandas as pd

# Load vectorizer and model
MODEL_DIR = Path("D:/SEM 4/TUGAS AKHIR PEMBELAJARAN MESIN/hoax-detector/model")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")

print("Model class:", type(model).__name__)

# If it is CalibratedClassifierCV wrapping LinearSVC
if hasattr(model, "calibrated_classifiers_"):
    # Get coefficients from the base estimator of the first fold
    base_estimator = model.calibrated_classifiers_[0].estimator
    coef = base_estimator.coef_[0]
else:
    # Otherwise check if it has coef_ directly
    coef = model.coef_[0]

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Create dataframe of terms and coefficients
coef_df = pd.DataFrame({
    'term': feature_names,
    'coef': coef
})

# Sort by coef
# Positive coefficients point to class 1 (Hoaks)
# Negative coefficients point to class 0 (Fakta)
print("\nTop 30 terms indicating HOAKS (class 1):")
print(coef_df.sort_values(by='coef', ascending=False).head(30))

print("\nTop 30 terms indicating FAKTA (class 0):")
print(coef_df.sort_values(by='coef', ascending=True).head(30))
