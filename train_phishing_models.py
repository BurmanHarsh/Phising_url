# ============================================================
# PHISHING WEBSITE DETECTION USING MACHINE LEARNING
# Step-by-step student-friendly implementation
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Import ML algorithms and evaluation metrics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# ------------------------------------------------------------
# STEP 1: Load the Dataset
# ------------------------------------------------------------
print("--- STEP 1: Loading Dataset ---")

# Check if cleaned CSV exists in current folder or assignment folder
csv_file = "PhiUSIIL_cleaned.csv" if os.path.exists("PhiUSIIL_cleaned.csv") else "assignment/PhiUSIIL_cleaned.csv"
df = pd.read_csv(csv_file)

print("Dataset loaded successfully!")
print("Dataset Shape (Rows, Columns):", df.shape)

# Separate input features (X) and target label (y)
# Label: 0 = Phishing, 1 = Legitimate
X = df.drop(columns=['label'])
y = df['label']

print("Number of Features:", X.shape[1])
print("Target classes count:")
print(y.value_counts())  # 1: Legitimate, 0: Phishing


# ------------------------------------------------------------
# STEP 2: Split into Training (80%) and Testing (20%) Sets
# ------------------------------------------------------------
print("\n--- STEP 2: Splitting Data into Train & Test Sets ---")

# Stratify=y keeps the same ratio of 0s and 1s in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ------------------------------------------------------------
# STEP 3: Define Different Machine Learning Models
# ------------------------------------------------------------
print("\n--- STEP 3: Initializing Machine Learning Classifiers ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
}


# ------------------------------------------------------------
# STEP 4: Train Each Model and Evaluate Performance
# ------------------------------------------------------------
print("\n--- STEP 4: Training and Evaluating Models ---")

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Fit (train) the model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate basic student metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"-> {name} Results:")
    print(f"   Accuracy : {acc * 100:.2f}%")
    print(f"   Precision: {prec * 100:.2f}%")
    print(f"   Recall   : {rec * 100:.2f}%")
    print(f"   F1-Score : {f1 * 100:.2f}%")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1
    })

# Convert results to a clean summary table
results_df = pd.DataFrame(results)
print("\n" + "="*50)
print("FINAL MODEL COMPARISON TABLE")
print("="*50)
print(results_df.to_string(index=False))

# Save table to CSV
results_df.to_csv("phishing_model_comparison.csv", index=False)
print("\nSaved comparison table to 'phishing_model_comparison.csv'")


# ------------------------------------------------------------
# STEP 5: Visualizations (Confusion Matrix & Comparison Bar Chart)
# ------------------------------------------------------------
print("\n--- STEP 5: Generating Simple Plots ---")

# 1. Bar Chart comparing Accuracies
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"] * 100, color=['#4a7298', '#6baed6', '#3182bd', '#08519c'])
plt.title("Model Accuracy Comparison (Phishing URL Detection)")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(95, 101)  # Zoom in because accuracies are very high
for i, v in enumerate(results_df["Accuracy"] * 100):
    plt.text(i, v + 0.3, f"{v:.2f}%", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig("figure3_model_performance.png")
plt.close()
print("Saved accuracy chart as 'figure3_model_performance.png'")

# 2. Confusion Matrix for Random Forest (Best model)
best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.colorbar()
plt.xticks([0, 1], ["Phishing (0)", "Legitimate (1)"])
plt.yticks([0, 1], ["Phishing (0)", "Legitimate (1)"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add text numbers inside matrix boxes
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("figure4_confusion_matrices.png")
plt.close()
print("Saved confusion matrix as 'figure4_confusion_matrices.png'")


# ------------------------------------------------------------
# STEP 6: Save Best Model for Future Predictions
# ------------------------------------------------------------
print("\n--- STEP 6: Saving Trained Model ---")

# Save model and column names so we can reuse them anytime
model_data = {
    "model": best_model,
    "feature_names": list(X.columns)
}

joblib.dump(model_data, "phishing_best_model.joblib")
print("Saved trained model to 'phishing_best_model.joblib'")
print("\nAll done! You can now run 'predict_phishing.py' to test URLs.")
