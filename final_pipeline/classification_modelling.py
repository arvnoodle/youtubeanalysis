# classification_modelling.py

import pandas as pd
import numpy as np
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures
from sklearn.impute import SimpleImputer
from feature_engine.pipeline import Pipeline
from feature_engine.encoding import RareLabelEncoder, DecisionTreeEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

def prepare_data(data):
    """
    Prepare data for classification by separating features and applying a pipeline.
    
    Args:
        data (pd.DataFrame): Preprocessed DataFrame from preprocessing step.
    
    Returns:
        tuple: Processed feature DataFrame (X), target variable (y), feature lists.
    """
    # Define feature groups
    categorical_features = [
        'video_category_id', 'is_weekend',
        'video_definition', 'video_dimension', 'video_licensed_content',
        'channel_have_hidden_subscribers', 'published_morning', 'published_afternoon',
        'published_evening', 'published_night', 'contain_1', 'contain_2', 'contain_3'
    ]

    numerical_features = [
        'video_duration_seconds', 'channel_video_count', 'channel_video_channel_publish_difference',
        'dominant_color_r', 'dominant_color_g', 'dominant_color_b',
        'brightness', 'color_diversity', 'tag_count'
    ]

    embedded_text_columns = [col for col in data.columns if col.startswith('text_embeddings_')]

    # Combine features and target
    X_categorical = data[categorical_features]
    X_numerical = data[numerical_features]
    X_text = data[embedded_text_columns]
    y = data['is_trending']

    # Combine all features
    X = pd.concat([X_numerical, X_categorical, X_text], axis=1)

    return X, y, categorical_features, numerical_features, embedded_text_columns

def feature_engineering_pipeline(X, categorical_features, numerical_features):
    """
    Apply feature selection and encoding pipeline.
    
    Args:
        X (pd.DataFrame): Feature DataFrame.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.
    
    Returns:
        pd.DataFrame: Processed features after pipeline.
    """
    imputer = SimpleImputer(strategy='constant', fill_value="Empty")
    
    # Fit the imputer to the data and transform it
    X[['contain_1', 'contain_2', 'contain_3']] = X[['contain_1', 'contain_2', 'contain_3']].fillna("Empty")
    ic(X.isnull().sum())
    pipeline = Pipeline([
        ("rare_label_encode", RareLabelEncoder(tol=0.005, ignore_format=False)),
        ("num_rare_label_encode", RareLabelEncoder(tol=0.005, ignore_format=True, variables=['tag_count'], replace_with=-1)),
        ("drop_constant_features", DropConstantFeatures(tol=0.90)),
        ("drop_duplicate_features", DropDuplicateFeatures()),
        ("drop_correlated_features", DropCorrelatedFeatures(method='pearson', threshold=0.90))
    ])
    ic(pipeline)
    X_transformed = pipeline.fit_transform(X)
    
    # Dynamically filter the feature lists to align with the transformed data
    remaining_features = X_transformed.columns.tolist()
    categorical_features = [f for f in categorical_features if f in remaining_features]
    numerical_features = [f for f in numerical_features if f in remaining_features]
    
    return X_transformed, categorical_features, numerical_features

def train_and_evaluate(X, y, categorical_features, numerical_features, embedded_text_columns):
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
    encoder = DecisionTreeEncoder(random_state=42, regression=False)
    X_train_encode = encoder.fit_transform(X_train_categorical, y_train)
    X_test_encode = encoder.transform(X_test_categorical)

    # Scale and combine features
    mmx_scaler = MinMaxScaler()
    X_train_scaled = np.hstack((X_train_numerical, X_train_encode, X_train_text))
    X_test_scaled = np.hstack((X_test_numerical, X_test_encode, X_test_text))

    # Train Random Forest Classifier
    model = RandomForestClassifier(
        max_depth=300,
        max_features='sqrt',
        n_estimators=1000,
        min_samples_leaf=1,
        min_samples_split=3,
        random_state=33
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate Model
    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model Accuracy: {accuracy}")

    # Classification Report and Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Feature Importance
    return model, X_train_scaled

def analyze_feature_importance(model, X):
    """
    Analyze and visualize feature importance.
    
    Args:
        model (RandomForestClassifier): Trained Random Forest model.
        X (pd.DataFrame): Feature DataFrame.
    """
    feature_names = X.columns
    feature_importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot Top 20 Feature Importances
    top_features = importance_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Top 20 Feature Importances in Random Forest')
    plt.gca().invert_yaxis()
    plt.show()

    print(importance_df)

