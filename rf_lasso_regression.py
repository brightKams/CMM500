import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import Lasso
import numpy as np

# Step 1: Load the data
# file_path = 'test_features.csv'  # Replace with your CSV file path
file_path = './files/features_merged_combination_updated.csv'  # Replace with your CSV file path

data = pd.read_csv(file_path)

# Step 2: Drop the 'Count' column
data = data.drop(columns=['Count'])

# Step 3: Transform date columns to numerical values (e.g., days since the current date)
current_date = pd.to_datetime('2024-12-10')
data['Creation Date'] = pd.to_datetime(data['Creation Date'], errors='coerce')
data['Expiration Date'] = pd.to_datetime(data['Expiration Date'], errors='coerce')

# Handle any NaT values that might occur during the conversion (e.g., invalid dates)
data['Creation Date'].fillna(current_date, inplace=True)
data['Expiration Date'].fillna(current_date, inplace=True)

# Create new columns for days since creation and days until expiration
data['Days Since Creation'] = (current_date - data['Creation Date']).dt.days
data['Days Until Expiration'] = (data['Expiration Date'] - current_date).dt.days

# Drop the original date columns
data = data.drop(columns=['Creation Date', 'Expiration Date'])

# Step 4: Encode categorical columns (e.g., 'Domain Name' and 'Name Servers')
label_encoder = LabelEncoder()

# Encode 'Domain Name'
data['Domain Name'] = label_encoder.fit_transform(data['Domain Name'])

# Encode 'Name Servers' if needed; handle non-numeric columns by converting them to strings
# For simplicity, use LabelEncoder on 'Name Servers'
data['Name Servers'] = data['Name Servers'].astype(str)
data['Name Servers'] = label_encoder.fit_transform(data['Name Servers'])

# Step 5: Separate features and target variable
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Use RandomForestClassifier to evaluate feature importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Print feature importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
print("Feature Importances:")
print(feature_importances.sort_values(ascending=False))

# Step 8: Select the most important features (e.g., keep top 50% of features)
threshold = np.percentile(feature_importances, 50)
selected_features = feature_importances[feature_importances >= threshold].index
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Step 9: Train a Lasso model for further feature selection and regularization
lasso = Lasso(alpha=0.1)  # Adjust alpha for regularization strength
lasso.fit(X_train_selected, y_train)

# Get selected features from Lasso model
lasso_selected_features = X_train_selected.columns[(lasso.coef_ != 0)]

print("Lasso Selected Features:")
print(lasso_selected_features)

# Step 10: Use the selected features for final model training (e.g., RandomForest)
rf_model_final = RandomForestClassifier(random_state=42)
rf_model_final.fit(X_train_selected[lasso_selected_features], y_train)

# Evaluate the final model
train_accuracy = rf_model_final.score(X_train_selected[lasso_selected_features], y_train)
test_accuracy = rf_model_final.score(X_test_selected[lasso_selected_features], y_test)

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Step 11: Make predictions (optional)
predictions = rf_model_final.predict(X_test_selected[lasso_selected_features])
print("Predictions:", predictions)
