import joblib
from pathlib import Path
import sys

# Add root dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.text_cleaner import preprocess_text

# Load vectorizer and model
MODEL_DIR = Path("D:/SEM 4/TUGAS AKHIR PEMBELAJARAN MESIN/hoax-detector/model")
tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")

# Texts to test
texts = [
    "Pemerintah resmikan proyek jalan tol baru hari ini",
    "VIRAL!!! Jokowi membagi-bagikan uang 50 juta gratis lewat WhatsApp"
]

for text in texts:
    print("\n" + "="*50)
    print("Original Text:", text)
    clean_text = preprocess_text(text, apply_stemming=True)
    print("Clean Text   :", clean_text)
    
    # Transform
    X = tfidf.transform([clean_text])
    
    # Get active features
    feature_indices = X.nonzero()[1]
    feature_names = tfidf.get_feature_names_out()
    
    active_features = []
    for idx in feature_indices:
        term = feature_names[idx]
        tfidf_val = X[0, idx]
        
        # Get coef from first calibrated classifier's base estimator if calibrated
        if hasattr(model, "calibrated_classifiers_"):
            coef = model.calibrated_classifiers_[0].estimator.coef_[0][idx]
            intercept = model.calibrated_classifiers_[0].estimator.intercept_[0]
        else:
            coef = model.coef_[0][idx]
            intercept = model.intercept_[0]
            
        contribution = tfidf_val * coef
        active_features.append({
            'term': term,
            'tfidf': tfidf_val,
            'coef': coef,
            'contribution': contribution
        })
        
    print("\nActive features and contributions:")
    import pandas as pd
    df_act = pd.DataFrame(active_features)
    if not df_act.empty:
        print(df_act.to_string(index=False))
        total_contrib = df_act['contribution'].sum() + intercept
        print(f"\nIntercept: {intercept:.4f}")
        print(f"Total Logits / Decision Function value: {total_contrib:.4f}")
    else:
        print("No active features found in TF-IDF vocabulary!")
        
    # Get predict and predict_proba
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    print(f"Prediction: {pred} (0=Fakta, 1=Hoaks)")
    print(f"Probabilities: Fakta={proba[0]*100:.1f}%, Hoaks={proba[1]*100:.1f}%")
