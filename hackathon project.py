import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

# Load the datasets
train_features = pd.read_csv('training_set_features.csv')
train_labels = pd.read_csv('training_set_labels.csv')
test_features = pd.read_csv('test_set_features.csv')

# Merge training features with labels
train_data = pd.merge(train_features, train_labels, on='respondent_id')

# Separate features and labels
X = train_data.drop(['respondent_id', 'xyz_vaccine', 'seasonal_vaccine'], axis=1)
y = train_data[['xyz_vaccine', 'seasonal_vaccine']]
X_test = test_features.drop(['respondent_id'], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Encode categorical variables
categorical_cols = X_imputed.select_dtypes(include=['object']).columns
onehot_encoder = OneHotEncoder(sparse=False, drop='first')

encoded_train_categorical_cols = pd.DataFrame(onehot_encoder.fit_transform(X_imputed[categorical_cols]), 
                                              columns=onehot_encoder.get_feature_names_out(categorical_cols))
encoded_test_categorical_cols = pd.DataFrame(onehot_encoder.transform(X_test_imputed[categorical_cols]), 
                                             columns=onehot_encoder.get_feature_names_out(categorical_cols))

# Replace the original categorical columns with encoded columns
X_imputed = X_imputed.drop(categorical_cols, axis=1)
X_train_preprocessed = pd.concat([X_imputed.reset_index(drop=True), encoded_train_categorical_cols.reset_index(drop=True)], axis=1)

X_test_imputed = X_test_imputed.drop(categorical_cols, axis=1)
X_test_preprocessed = pd.concat([X_test_imputed.reset_index(drop=True), encoded_test_categorical_cols.reset_index(drop=True)], axis=1)

# Normalize or scale the features if necessary
scaler = StandardScaler()
numerical_cols = X_train_preprocessed.select_dtypes(include=['int64', 'float64']).columns
X_train_preprocessed[numerical_cols] = scaler.fit_transform(X_train_preprocessed[numerical_cols])
X_test_preprocessed[numerical_cols] = scaler.transform(X_test_preprocessed[numerical_cols])

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_preprocessed, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
multi_target_rf = MultiOutputClassifier(rf_model, n_jobs=1)  # Reduced n_jobs to 1

# Train the model
multi_target_rf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_proba = multi_target_rf.predict_proba(X_val)

# Extract probabilities for each class
xyz_vaccine_probs = y_pred_proba[0][:, 1]
seasonal_vaccine_probs = y_pred_proba[1][:, 1]

# Calculate ROC AUC for each target
auc_xyz = roc_auc_score(y_val['xyz_vaccine'], xyz_vaccine_probs)
auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], seasonal_vaccine_probs)
print('ROC AUC for xyz_vaccine:', auc_xyz)
print('ROC AUC for seasonal_vaccine:', auc_seasonal)
print('Mean ROC AUC:', np.mean([auc_xyz, auc_seasonal]))

# Make predictions on the test set
test_pred_proba = multi_target_rf.predict_proba(X_test_preprocessed)

# Prepare the submission file
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': test_pred_proba[0][:, 1],
    'seasonal_vaccine': test_pred_proba[1][:, 1]
})

submission.to_csv('submission.csv', index=False)
print('Submission file created successfully!')

