"""
DEPRECATED
R2 = 0.6239745591498792 0.02682099573307188
MAE = -4.426934456650981 0.1205837895100812
"""

import numpy as np
import pandas as pd
from sklearn import (
    decomposition,
    ensemble,
    feature_selection,
    impute,
    linear_model,
    model_selection,
    pipeline,
    preprocessing,
    svm,
)
from sklearn.neighbors import LocalOutlierFactor
from lightgbm import LGBMRegressor  # Import LGBMRegressor


def main():
    X_train, y_train, X_test = load_data()
    print("Initial shapes:", X_train.shape, y_train.shape, X_test.shape)

    X_train, y_train = remove_outliers(X_train, y_train)
    X_train, X_test = preprocess(X_train, X_test)
    X_train, X_test = select_features(X_train, y_train, X_test)
    print("Shapes after preprocessing and feature selection:", X_train.shape, y_train.shape, X_test.shape)

    # Use LGBMRegressor as the model
    model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate the model using cross-validation
    score = model_selection.cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error")
    print(score.mean(), score.std())
    #print(f"Mean R² Score: {score.mean():.4f}, Std: {score.std():.4f}")

    # Create submission
    create_submission(model, X_train, y_train, X_test)


def load_data():
    X_train = np.genfromtxt("data/X_train.csv", delimiter=",", skip_header=1)[:, 1:]
    y_train = np.genfromtxt("data/y_train.csv", delimiter=",", skip_header=1)[:, 1:]
    X_test = np.genfromtxt("data/X_test.csv", delimiter=",", skip_header=1)[:, 1:]
    y_train = y_train.ravel()
    return X_train, y_train, X_test


def remove_outliers(X_train, y_train):
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        impute.SimpleImputer(strategy="median"),
        decomposition.PCA(n_components=3),
        LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    )
    pred = model.fit_predict(X_train)

    # Keep only inliers
    X_train_clean = X_train[pred > 0]
    y_train_clean = y_train[pred > 0]
    return X_train_clean, y_train_clean


def preprocess(X_train, X_test):
    model = pipeline.Pipeline([
        ('imputer', impute.SimpleImputer(strategy="median")),
        ('scaler', preprocessing.StandardScaler())
    ])
    X_train = model.fit_transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def select_features(X_train, y_train, X_test):
    model = pipeline.Pipeline([
        ('variance_threshold', feature_selection.VarianceThreshold()),
        ('select_kbest', feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=200)),
        ('select_from_model', feature_selection.SelectFromModel(estimator=linear_model.Lasso(alpha=0.1, max_iter=5000))),
    ])
    model.fit(X_train, y_train)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    return X_train, X_test


def create_submission(model, X_train, y_train, X_test):
    # Fit the model on the entire training data
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.vstack((np.arange(X_test.shape[0]), pred)).T
    print("Creating submission file...")
    pred = np.round(pred).astype(int)
    np.savetxt("submission_7_git_lof_LGBMRegressor.csv", pred, delimiter=",", header="id,y", comments="", fmt=["%d", "%d"])
    print("Submission file created.")


if __name__ == "__main__":
    main()
