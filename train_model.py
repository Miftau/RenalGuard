import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# --- 1. Load Data ---
print("Loading datasets...")
general_df = pd.read_csv('nhanes_kidney_general.csv')
clinical_df = pd.read_csv('nhanes_kidney_clinical.csv')

# Check for the target variable
target_col = 'KIDNEY_DISEASE'
if target_col not in general_df.columns or target_col not in clinical_df.columns:
    raise ValueError(f"Target column '{target_col}' not found in one or both datasets.")

# --- 2. Prepare Data for Modeling ---
def prepare_data(df, target_col):
    """Separate features (X) and target (y), handle missing values, encode categories."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    # Remove 'SEQN' if it's present in features
    if 'SEQN' in numerical_cols:
        numerical_cols.remove('SEQN')
    print(f"  - Numerical features: {len(numerical_cols)}, Categorical features: {len(categorical_cols)}")

    # Preprocessing pipeline for numerical data
    numerical_transformer = SimpleImputer(strategy='median') # Use median for robustness

    # Preprocessing pipeline for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('label_encoder', LabelEncoder()) # Label encoding for simplicity with Random Forest
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            # Note: ColumnTransformer doesn't work directly with LabelEncoder on pandas DataFrame.
            # We will handle categorical encoding manually after imputation for simplicity here.
        ]
    )

    # Apply numerical imputation
    X_num_imputed = pd.DataFrame(
        numerical_transformer.fit_transform(X[numerical_cols]),
        columns=numerical_cols,
        index=X.index
    )

    # Handle categorical variables manually
    X_cat_encoded = X[categorical_cols].copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit and transform, handling unknown labels by adding them as 'missing'
        X_cat_encoded[col] = X_cat_encoded[col].fillna('missing')
        le.fit(X_cat_encoded[col])
        X_cat_encoded[col] = le.transform(X_cat_encoded[col])
        encoders[col] = le # Store encoder for later use

    # Combine processed numerical and categorical features
    X_processed = pd.concat([X_num_imputed, X_cat_encoded], axis=1)

    # Final imputation if any NAs remain after encoding (e.g., from unseen categories if applicable)
    # Though unlikely with our current strategy, a final check is good practice.
    final_imputer = SimpleImputer(strategy='median')
    X_final = pd.DataFrame(
        final_imputer.fit_transform(X_processed),
        columns=X_processed.columns,
        index=X_processed.index
    )

    return X_final, y, encoders, numerical_cols, categorical_cols

print("\n--- Preparing General (Lifestyle) Data ---")
X_gen, y_gen, gen_encoders, gen_num_cols, gen_cat_cols = prepare_data(general_df, target_col)

print("\n--- Preparing Clinical Data ---")
X_cli, y_cli, cli_encoders, cli_num_cols, cli_cat_cols = prepare_data(clinical_df, target_col)

# --- 3. Train Models ---
def train_and_evaluate(X, y, model_name):
    """Train a model, evaluate it, and return the trained model and metrics."""
    print(f"\n--- Training {model_name} Model ---")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create and train the model with hyperparameter tuning
    # Using a basic pipeline for the classifier itself
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1] # Probability of positive class

    # Evaluation Metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} - Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} - AUC-ROC (Test): {auc_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --- 4. Visualizations ---
    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'cm_{model_name.lower()}.png') # Save the plot
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_{model_name.lower()}.png') # Save the plot
    plt.show()

    return best_model, auc_roc

# Train General (Lifestyle) Model
model_gen, auc_gen = train_and_evaluate(X_gen, y_gen, "Lifestyle")

# Train Clinical Model
model_cli, auc_cli = train_and_evaluate(X_cli, y_cli, "Clinical")

# --- 5. Save Models and Preprocessing Objects ---
print("\n--- Saving Models and Preprocessors ---")
# Save the trained models
joblib.dump(model_gen, 'kidney_model_lifestyle.pkl')
joblib.dump(model_cli, 'kidney_model_clinical.pkl')

# Save the preprocessing objects (encoders, column lists)
preprocessing_info_gen = {
    'encoders': gen_encoders,
    'numerical_columns': gen_num_cols,
    'categorical_columns': gen_cat_cols
}
preprocessing_info_cli = {
    'encoders': cli_encoders,
    'numerical_columns': cli_num_cols,
    'categorical_columns': cli_cat_cols
}

joblib.dump(preprocessing_info_gen, 'preprocessing_info_gen.pkl')
joblib.dump(preprocessing_info_cli, 'preprocessing_info_cli.pkl')

print("\n✅ Models and preprocessing info saved successfully!")
print("  - Lifestyle Model: kidney_model_lifestyle.pkl")
print("  - Clinical Model: kidney_model_clinical.pkl")
print("  - Lifestyle Preprocessing: preprocessing_info_gen.pkl")
print("  - Clinical Preprocessing: preprocessing_info_cli.pkl")
print("  - Confusion Matrix Plots: cm_lifestyle.png, cm_clinical.png")
print("  - ROC Curve Plots: roc_lifestyle.png, roc_clinical.png")