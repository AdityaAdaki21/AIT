import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=FutureWarning) # Ignore some future warnings from libs

# --- Configuration ---
DATA_FILE = 'api_logs.csv' # Make sure this matches your CSV filename
ANOMALY_FEATURES = [
    'response_time',  # Corrected column name
    'latency_ms',
    'bytes_transferred',
    'simulated_cpu_cost',
    'simulated_memory_mb'
]
ERROR_PREDICTION_FEATURES_NUM = [
    # 'response_time',  # Commented out to prevent data leakage
    'latency_ms',
    'bytes_transferred',
    'hour_of_day',
    'simulated_cpu_cost',
    'simulated_memory_mb'
]
ERROR_PREDICTION_FEATURES_CAT = [
    'api_id',
    'env'
]
TARGET_VARIABLE = 'is_error' # We will create this column
RANDOM_STATE = 42 # For reproducibility

# --- 1. Load and Prepare Data ---
print("--- Loading and Preparing Data ---")
try:
    # Load data, parsing the timestamp column correctly
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    print(f"Successfully loaded {DATA_FILE}. Shape: {df.shape}")
    print("Sample data:")
    print(df.head())
    print("\nData Types:")
    # Use df.info(verbose=True, show_counts=True) for more detail if needed
    df.info()

except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please ensure the CSV file is in the same directory or provide the correct path.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# Basic Data Cleaning / Feature Engineering
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
# Optional: Handle missing values if necessary (e.g., df.dropna(), df.fillna(...))
# For this example, we assume no critical missing values based on the previous output

# Create the target variable for error prediction: 1 if status_code is not 200, else 0
df[TARGET_VARIABLE] = (df['status_code'] != 200).astype(int)
print(f"\nCreated target variable '{TARGET_VARIABLE}'. Error counts:")
print(df[TARGET_VARIABLE].value_counts())

# Encode Categorical Features for Error Prediction
# Using LabelEncoder for simplicity in a hackathon. OneHotEncoder is often better.
encoders = {}
df_encoded = df.copy() # Work on a copy for encoding
for col in ERROR_PREDICTION_FEATURES_CAT:
    if col in df_encoded.columns:
        le = LabelEncoder()
        # Handle potential NaN values before encoding if they exist
        # df_encoded[col] = df_encoded[col].fillna('Missing') # Example strategy
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) # Ensure string type
        encoders[col] = le # Store encoder to potentially decode later if needed
        print(f"Label encoded column: {col}")
    else:
         print(f"Warning: Categorical feature '{col}' not found in DataFrame.")

print("\nSample data after encoding (showing only features used later):")
# Ensure all selected numeric features exist before trying to display them
display_cols_num = [f for f in ERROR_PREDICTION_FEATURES_NUM if f in df_encoded.columns]
print(df_encoded[ERROR_PREDICTION_FEATURES_CAT + display_cols_num].head())

# --- 2. Anomaly Detection using Isolation Forest ---
print("\n--- Running Anomaly Detection (Isolation Forest) ---")

# Check if all needed features exist *before* the if/else
all_anomaly_features_present = all(feature in df.columns for feature in ANOMALY_FEATURES)

if not all_anomaly_features_present:
    print(f"Error: Not all anomaly features {ANOMALY_FEATURES} found in the dataframe.")
    missing_features = [f for f in ANOMALY_FEATURES if f not in df.columns]
    print(f"Missing features: {missing_features}")
    print("Skipping anomaly detection.")
else:
    # This block should execute if all features are present
    print("DEBUG: Entering anomaly detection 'else' block.", flush=True) # DEBUG PRINT
    available_anomaly_features = ANOMALY_FEATURES # Use the original list as all are present
    print(f"Using available features for anomaly detection: {available_anomaly_features}", flush=True)

    anomaly_data = df[available_anomaly_features].copy()

    # Check for non-numeric data types just in case
    numeric_cols = anomaly_data.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) != len(available_anomaly_features):
        print("Warning: Non-numeric columns found in anomaly features. Attempting to use only numeric.", flush=True)
        non_numeric = [f for f in available_anomaly_features if f not in numeric_cols]
        print(f"Non-numeric columns skipped: {non_numeric}", flush=True)
        anomaly_data = anomaly_data[numeric_cols]
        available_anomaly_features = numeric_cols # Update the list of features actually used

    if not available_anomaly_features:
        print("Error: No numeric features left for anomaly detection after checks. Skipping.", flush=True)
    else:
        # Handle potential NaNs in the numeric data selected
        if anomaly_data.isnull().any().any():
             print("Warning: NaNs found in anomaly detection features. Filling with median.", flush=True)
             anomaly_data.fillna(anomaly_data.median(), inplace=True)

        # Initialize and fit Isolation Forest
        # Contamination='auto' is data-dependent, setting a small fixed value might be more stable
        iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_STATE) # e.g., assume 5% anomalies
        print("Fitting Isolation Forest model...", flush=True)
        try:
            anomaly_preds = iso_forest.fit_predict(anomaly_data)
            print("Isolation Forest fitting complete.", flush=True)

            # Add predictions to the original DataFrame (-1 indicates anomaly, 1 indicates normal)
            df['anomaly_flag'] = anomaly_preds
            df['is_anomaly'] = (df['anomaly_flag'] == -1).astype(int) # Easier boolean flag

            print("\nAnomaly Detection Results:", flush=True) # Added flush=True
            print(df['is_anomaly'].value_counts(), flush=True) # Added flush=True

            # Show some detected anomalies
            anomalies = df[df['is_anomaly'] == 1]
            print(f"\nTop {min(10, len(anomalies))} detected anomalies:", flush=True) # Added flush=True
            # Display relevant columns for anomalies
            print(anomalies[['timestamp', 'api_id', 'env', 'is_error'] + available_anomaly_features].head(10), flush=True) # Added flush=True

            # Optional: Visualize anomalies (Plotting is still commented out)
            # Ensure plotting code is also within this block if re-enabled
            # if len(available_anomaly_features) >= 2:
            #     plt.figure(figsize=(10, 6))
            #     sns.scatterplot(
            #         data=df,
            #         x=available_anomaly_features[0],
            #         y=available_anomaly_features[1],
            #         hue='is_anomaly',
            #         palette={0: 'blue', 1: 'red'},
            #         style='is_anomaly',
            #         markers={0: '.', 1: 'X'},
            #         s=50
            #     )
            #     plt.title(f'Anomaly Detection: {available_anomaly_features[0]} vs {available_anomaly_features[1]}')
            #     plt.xlabel(available_anomaly_features[0])
            #     plt.ylabel(available_anomaly_features[1])
            #     plt.legend(title='Is Anomaly?', labels=['Normal', 'Anomaly'])
            #     plt.tight_layout()
            #     plt.show()

        except Exception as e:
            print(f"An error occurred during Isolation Forest fitting/prediction: {e}", flush=True)


# --- 3. Error Prediction using LightGBM ---
print("\n--- Running Error Prediction (LightGBM Classifier) ---")

# Define features (X) and target (y)
# Ensure features actually exist in the encoded dataframe
error_features_num_present = [f for f in ERROR_PREDICTION_FEATURES_NUM if f in df_encoded.columns]
error_features_cat_present = [f for f in ERROR_PREDICTION_FEATURES_CAT if f in df_encoded.columns]
selected_features = error_features_num_present + error_features_cat_present

if not selected_features:
     print("Error: No features available for error prediction. Skipping.")
elif TARGET_VARIABLE not in df_encoded.columns:
     print(f"Error: Target variable '{TARGET_VARIABLE}' not found. Skipping error prediction.")
else:
    X = df_encoded[selected_features]
    y = df_encoded[TARGET_VARIABLE]

    print(f"\nFeatures for Error Prediction: {selected_features}")
    print(f"Target variable: {TARGET_VARIABLE}")

    # Check if target variable has more than one class
    if len(y.unique()) <= 1:
        print(f"Warning: Target variable '{TARGET_VARIABLE}' has only one class ({y.unique()[0]}). Classification model cannot be trained. Skipping.")
    else:
        # Split data into training and testing sets
        # Use stratify=y if classes are imbalanced (likely for errors)
        # For time-series logs, a chronological split is better, but random split is simpler here.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
        )
        print(f"\nTraining set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Error distribution in training set:\n{y_train.value_counts(normalize=True)}")
        print(f"Error distribution in test set:\n{y_test.value_counts(normalize=True)}")


        # Initialize and train LightGBM Classifier
        # Add scale_pos_weight for imbalanced datasets if needed
        # error_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
        # lgbm_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE, scale_pos_weight=error_ratio)
        lgbm_clf = lgb.LGBMClassifier(random_state=RANDOM_STATE) # Using default weights for now

        print("\nTraining LightGBM model...")
        try:
            lgbm_clf.fit(X_train, y_train,
                        # Optional: Specify categorical features for LGBM
                         categorical_feature=[f for f in error_features_cat_present if f in X_train.columns])
            print("Training complete.")

            # Make predictions on the test set
            y_pred = lgbm_clf.predict(X_test)
            y_pred_proba = lgbm_clf.predict_proba(X_test)[:, 1] # Probability of error

            # Evaluate the model
            print("\n--- Error Prediction Model Evaluation ---")
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")

            print("\nClassification Report:")
            # Use zero_division=0 to avoid warnings if a class has no predicted samples
            print(classification_report(y_test, y_pred, target_names=['No Error (0)', 'Error (1)'], zero_division=0))

            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            print(cm)

            # Plot Confusion Matrix
            try:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Error', 'Error'])
                disp.plot(cmap=plt.cm.Blues)
                plt.title('Confusion Matrix - Error Prediction')
                plt.show() # Display the plot
            except Exception as plot_e:
                print(f"Could not display Confusion Matrix plot: {plot_e}")


            # Plot Feature Importances
            print("\nFeature Importances:")
            try:
                # Map importance scores back to feature names
                feature_importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': lgbm_clf.feature_importances_
                }).sort_values('importance', ascending=False)
                print(feature_importance_df)

                lgb.plot_importance(lgbm_clf, max_num_features=20, figsize=(10, 8), importance_type='gain') # 'gain' often more informative
                plt.title('LightGBM Feature Importances (Gain)')
                plt.tight_layout()
                plt.show() # Display the plot
            except Exception as plot_e:
                 print(f"Could not display Feature Importance plot: {plot_e}")


            # Add predictions back to a copy of the test part of the original dataframe for inspection
            # Use .loc to avoid SettingWithCopyWarning
            df_test_results = df.loc[X_test.index].copy()
            df_test_results['predicted_error'] = y_pred
            df_test_results['predicted_error_probability'] = y_pred_proba
            print("\nSample of test data with predictions:")
            # Ensure columns exist before trying to display
            display_cols = ['timestamp', 'api_id', 'status_code', TARGET_VARIABLE,
                            'predicted_error', 'predicted_error_probability'] + selected_features
            display_cols_present = [c for c in display_cols if c in df_test_results.columns]
            print(df_test_results[display_cols_present].head())

        except Exception as e:
             print(f"An error occurred during LightGBM training/evaluation: {e}")


print("\n--- Analysis Complete ---")