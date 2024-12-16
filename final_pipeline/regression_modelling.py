import pandas as pd
import numpy as np
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, DecisionTreeEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
from icecream import ic
def train_and_evaluate_regression(X, y, categorical_features, numerical_features, embedded_text_columns):
    """
    Train and evaluate the Random Forest model.
    
    Args:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.
        embedded_text_columns (list): List of text embedding column names.
    
    Returns:
        object: Trained Random Forest model.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Dynamically adjust feature lists to ensure they align with the actual DataFrame
    all_columns = X.columns.tolist()
    categorical_features = [f for f in categorical_features if f in all_columns]
    numerical_features = [f for f in numerical_features if f in all_columns]
    embedded_text_columns = [f for f in embedded_text_columns if f in all_columns]

    # Separate by feature type
    X_train_categorical = X_train[categorical_features]
    X_test_categorical = X_test[categorical_features]

    X_train_numerical = X_train[numerical_features]
    X_test_numerical = X_test[numerical_features]

    X_train_text = X_train[embedded_text_columns]
    X_test_text = X_test[embedded_text_columns]

    # Encode categorical features
    encoder = DecisionTreeEncoder(random_state=42, regression=True)
    X_train_encode = encoder.fit_transform(X_train_categorical, y_train)
    X_test_encode = encoder.transform(X_test_categorical)

    # Scale and combine features
    mmx_scaler = MinMaxScaler()
    X_train_scaled = mmx_scaler.fit_transform(np.hstack((X_train_numerical, X_train_encode, X_train_text)))
    X_test_scaled = mmx_scaler.transform(np.hstack((X_test_numerical, X_test_encode, X_test_text)))

    # Train Random Forest Regressor
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train_scaled, y_train)

    # Evaluate Model
    y_pred_train = regressor.predict(X_train_scaled)
    y_pred_test = regressor.predict(X_test_scaled)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")
    print(f"Train R2: {train_r2}")
    print(f"Test R2: {test_r2}")

    return regressor, X_test_scaled, y_test, y_pred_test

def analyze_feature_importance_regression(model, X, feature_names):
    """
    Analyze and visualize feature importance.
    
    Args:
        model (RandomForestRegressor): Trained Random Forest Regressor model.
        X (pd.DataFrame): Feature DataFrame.
        feature_names (list): Names of features in X.
    """
    feature_importances = model.feature_importances_
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    plt.show()

def shap_analysis(model, X_test_scaled, feature_names, sample_index=0):
    """
    Perform SHAP analysis for a single prediction.
    
    Args:
        model (RandomForestRegressor): Trained Random Forest Regressor model.
        X_test_scaled (np.ndarray): Scaled test set.
        feature_names (list): Names of features in the test set.
        sample_index (int): Index of the sample to analyze.
    """
    explainer = shap.TreeExplainer(model)
    shap.initjs()
    scaled_data_frame = pd.DataFrame(X_test_scaled, columns=feature_names)
    ic(scaled_data_frame.shape)
    X_sample = scaled_data_frame[88:89]
    shap_values = explainer.shap_values(X_sample)
    shap.force_plot(explainer.expected_value, shap_values, X_sample)
