from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import pandas as pd
from datetime import datetime

df = pd.read_csv('Data_Collection\datasets\isl_data.csv')

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])  # All columns except 'label'
y = df['label']

# Assuming X contains all coordinate columns and y contains labels
print("Training run start")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train.values, y_train)

print("Training run ended")

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate with multiple metrics
accuracy = rf_model.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model
current_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
model_filename = f'models/random_forest_model+{current_time}.joblib'
joblib.dump(rf_model, model_filename)
print(f"Model saved to {model_filename}")
