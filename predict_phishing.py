import os
import pandas as pd
import joblib
model_file = "phishing_best_model.joblib" if os.path.exists("phishing_best_model.joblib") else "assignment/phishing_best_model.joblib"

if not os.path.exists(model_file):
    print("Error: Model file not found! Please run 'python train_phishing_models.py' first.")
    exit()

print("Loading saved model from:", model_file)
saved_data = joblib.load(model_file)
model = saved_data["model"]
feature_names = saved_data["feature_names"]
print("Model loaded successfully!\n")
def predict_sample(features_df, sample_name="Sample"):
    prediction = model.predict(features_df)[0]
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features_df)[0]
        confidence = max(probability) * 100
    else:
        confidence = 100.0

    print("=" * 55)
    print(f"Prediction for: {sample_name}")
    print("=" * 55)
    
    if prediction == 1:
        print("Verdict   : [SAFE] LEGITIMATE WEBSITE")
        print(f"Confidence: {confidence:.2f}%")
        print("Details   : This website has normal code size and verified links.")
    else:
        print("Verdict   : [DANGER] PHISHING WEBSITE (MALICIOUS)")
        print(f"Confidence: {confidence:.2f}%")
        print("Details   : Warning! This website mimics a real site to steal credentials.")
    print("=" * 55 + "\n")
cleaned_file = "PhiUSIIL_cleaned.csv" if os.path.exists("PhiUSIIL_cleaned.csv") else "assignment/PhiUSIIL_cleaned.csv"
df = pd.read_csv(cleaned_file, nrows=50)
sample_legit = df[df['label'] == 1].drop(columns=['label']).iloc[[0]]
sample_phish = df[df['label'] == 0].drop(columns=['label']).iloc[[0]]

print("Running Demonstration on 2 Test Samples:\n")
predict_sample(sample_legit, "Sample 1 (Known Legitimate Site: https://www.southbankmosaics.com)")
predict_sample(sample_phish, "Sample 2 (Known Phishing Site: http://www.teramill.com)")
def predict_my_csv(csv_path):
    if os.path.exists(csv_path):
        new_data = pd.read_csv(csv_path)
        X_new = new_data[feature_names]
        predictions = model.predict(X_new)
        new_data['Prediction'] = ['Legitimate' if p == 1 else 'Phishing' for p in predictions]
        new_data.to_csv("my_predictions_output.csv", index=False)
        print(f"Saved predictions for {len(new_data)} rows to 'my_predictions_output.csv'")
    else:
        print(f"File '{csv_path}' not found.")

