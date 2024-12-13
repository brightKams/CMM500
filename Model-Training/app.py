import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold


# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from collections import Counter
from math import log2

# Title for the app
st.title("Malicious Domain Detection Using Random Forest & K-Fold Cross-Validation Model Evaluation and Prediction App")

# File uploader for CSV dataset
uploaded_file = st.file_uploader("Upload a CSV file for training", type="csv")

# Initialize global variables for the trained model and dataset
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Function to calculate entropy of a string
def calculate_entropy(string):
    probabilities = [freq / len(string) for freq in Counter(string).values()]
    return -sum(p * log2(p) for p in probabilities if p > 0)

# Function to process data and prepare it for training
def process_data(file_path):
    try:
        data = pd.read_csv(file_path)
        st.success("Dataset loaded successfully.")
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
        return None, None

    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns

    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Separate features and labels
    try:
        X = data.drop("label", axis=1)
        y = data["label"]
    except KeyError:
        st.error("The dataset must contain a 'label' column for classification.")
        return None, None

    return X, y

from sklearn.model_selection import KFold

# Function to train the model using K-Fold Cross-Validation
def train_random_forest_with_kfold(X, y, n_splits=5):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Create the preprocessing pipeline for numeric and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Initialize the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Perform K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }

    fold_idx = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model on the training fold
        model.fit(X_train, y_train)

        # Predict on the validation fold
        y_pred = model.predict(X_val)

        # Evaluate metrics for the current fold
        metrics['accuracy'].append(accuracy_score(y_val, y_pred))
        metrics['f1'].append(f1_score(y_val, y_pred, average='weighted'))
        metrics['precision'].append(precision_score(y_val, y_pred, average='weighted'))
        metrics['recall'].append(recall_score(y_val, y_pred, average='weighted'))

        st.write(f"### Fold {fold_idx} Results:")
        st.write(f"**Accuracy**: {metrics['accuracy'][-1]:.4f}")
        st.write(f"**F1-Score**: {metrics['f1'][-1]:.4f}")
        st.write(f"**Precision**: {metrics['precision'][-1]:.4f}")
        st.write(f"**Recall**: {metrics['recall'][-1]:.4f}")
        fold_idx += 1

    # Display overall cross-validation results
    st.write("### Overall Cross-Validation Results:")
    st.write(f"**Mean Accuracy**: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    st.write(f"**Mean F1-Score**: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")
    st.write(f"**Mean Precision**: {np.mean(metrics['precision']):.4f} ± {np.std(metrics['precision']):.4f}")
    st.write(f"**Mean Recall**: {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")

    # Save the last trained model in session state
    st.session_state.trained_model = model


# Function to extract features from a domain name
def extract_features(domain):
    domain = domain.lower()
    features = {
        'Domain Length': len(domain),
        'Numeric Count': sum(c.isdigit() for c in domain),
        'Vowel Count': sum(c in 'aeiou' for c in domain),
        'Consonant Count': sum(c.isalpha() and c not in 'aeiou' for c in domain),
        'Other Character Count': sum(not c.isalnum() for c in domain),
        'Entropy': calculate_entropy(domain),
        # Add the missing features here, even with placeholder values if needed:
        'Creation Date': '01/01/2000',  # Replace with actual values or remove if not relevant
        'Name Servers': 'ns1.example.com,ns2.example.com',  # Replace with actual values or remove if not relevant
        'Expiration Date': '01/01/2030',  # Replace with actual values or remove if not relevant
        'Domain Name': domain,  # Include the domain name itself
        'Count': 0  # Replace with actual count or remove if not relevant
    }
    return pd.DataFrame([features])
    # Add any other relevant features from your dataset here
    # return pd.DataFrame([features])

# Function to predict a single domain
def predict_domain(domain_name):
    if st.session_state.trained_model is None:
        st.error("No trained model is available. Please upload a dataset and train the model first.")
        return

    try:
        # Extract features from the domain name
        input_features = extract_features(domain_name)

        # Ensure input columns match training data
        input_features = input_features[st.session_state.X.columns.intersection(input_features.columns)]

        # Make prediction using the trained model
        prediction = st.session_state.trained_model.predict(input_features)

        # Display prediction with a message
        if prediction[0] == 0:
            st.write(f"### Predicted Label: {prediction[0]} ({domain_name} is Benign)")
        else:
            st.write(f"### Predicted Label: {prediction[0]} (Malicious)")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Display the app UI logic
if uploaded_file is not None:
    st.session_state.X, st.session_state.y = process_data(uploaded_file)

    # Only proceed if the data is valid and we have X and y
    if st.session_state.X is not None and st.session_state.y is not None:
        if st.button("Train Random Forest Model with K-Fold Cross-Validation"):
            train_random_forest_with_kfold(st.session_state.X, st.session_state.y)

# Input form for domain testing
if st.session_state.X is not None:
    with st.form("domain_prediction"):
        st.write("### Test Domain Prediction")
        domain_name = st.text_input("Enter a domain name:")
        submitted = st.form_submit_button("Predict")
        if submitted:
            if st.session_state.trained_model is None:
                st.error("You need to train the model first by uploading a dataset and clicking 'Train Random Forest Model'.")
            else:
                predict_domain(domain_name)