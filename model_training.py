import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv('nhai_processed_data.csv')

# Define Features (X) and Target (y)
X = df.drop(['Accident_Risk_Label', 'Latitude', 'Longitude'], axis=1)
y = df['Accident_Risk_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("--- Model Evaluation Metrics ---")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# SAVE OVER THE OLD MODEL
model.save_model('nhai_risk_model.json')
print("✅ New AI Model successfully saved as nhai_risk_model.json!")