from eda import load_and_clean_data, engineer_features, perform_eda
from advance_preprocessing_yolo_sbert import preprocess_data
from classification_modelling import prepare_data, feature_engineering_pipeline, train_and_evaluate, analyze_feature_importance
from regression_modelling import train_and_evaluate_regression, analyze_feature_importance_regression, shap_analysis
from icecream import ic
ic.configureOutput(includeContext=True, prefix='DEV DEBUG: ')

def main():
    # Load and clean data
    file_path = 'test_data.csv' 
    data = load_and_clean_data(file_path)
    ic(data.shape)
    ic(data.isnull().sum())
    # Feature engineering
    data = engineer_features(data)
    ic(data.shape)
    ic(data.isnull().sum())
    # Perform EDA
    perform_eda(data)

    # Preprocessing for modeling
    processed_data = preprocess_data(data)
    ic(data.shape)
    ic(data.columns)
    # Prepare data for classification
    X, y, categorical_features, numerical_features, embedded_text_columns = prepare_data(processed_data)
    ic(X)
    ic(y)
    ic(categorical_features)
    ic(numerical_features)
    ic(embedded_text_columns)
    # Apply feature engineering pipeline
    X_final, categorical_features, numerical_features = feature_engineering_pipeline(X, categorical_features, numerical_features)
    ic(X_final.shape)
    ic(X_final.columns)
    # Train and evaluate model Classification
    model, X_train_scaled = train_and_evaluate(X_final, y, categorical_features, numerical_features, embedded_text_columns)
    analyze_feature_importance(model, X_final)
    # Train and evaluate model Regression
    regressor, X_test_scaled, y_test, y_pred_test = train_and_evaluate_regression(X_final, processed_data['engagement_rate'], categorical_features, numerical_features, embedded_text_columns)
    analyze_feature_importance_regression(regressor, X_final, list(categorical_features + numerical_features + embedded_text_columns))
    shap_analysis(regressor, X_test_scaled, list(categorical_features + numerical_features + embedded_text_columns))
if __name__ == "__main__":
    main()
