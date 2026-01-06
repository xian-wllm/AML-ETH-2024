"""
Total time elapsed: 1734.96 seconds
Grid Search complete.
Best parameters: {'feature_selection__selectfrommodel__estimator__alpha': 0.1, 'feature_selection__selectkbest__k': 200, 'regressor__gbm__learning_rate': 0.095, 'regressor__svr__C': 50.0, 'regressor__svr__epsilon': 1e-05}
Best score (Negative MAE): -4.1361
Creating submission file...
Submission file created.

Best Parameters (SVR C=50):
R2 = 0.6725107072230471 0.031236919338198017
MAE = -4.139083940625904 0.179692961503697

"""


import numpy as np
import time
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
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def main():
    X_train, y_train, X_test = load_data()
    print(Fore.CYAN + "Initial shapes:", X_train.shape, y_train.shape, X_test.shape)

    X_train, y_train = remove_outliers(X_train, y_train)
    print(Fore.CYAN + "After outlier removal:", X_train.shape, y_train.shape)

    # Define the full pipeline including preprocessing, feature selection, and the model
    full_pipeline = pipeline.Pipeline([
        ('preprocessing', preprocessing_pipeline()),
        ('feature_selection', feature_selection_pipeline()),
        ('regressor', model_pipeline())
    ])

    # Define the parameter grid with hyperparameters to tune
    param_grid = {
        # Feature Selection Parameters
        'feature_selection__selectkbest__k': [180, 200, 220],  # Around default k=200
        'feature_selection__selectfrommodel__estimator__alpha': [0.1, 0.15, 0.2],  # Around default alpha=0.1

        # SVR Hyperparameters
        'regressor__svr__C': [50.0, 60.0, 70.0],  # Around default C=60.0
        'regressor__svr__epsilon': [1e-6, 1e-5, 1e-4],  # Around default epsilon=1e-5

        # Gradient Boosting Hyperparameters
        'regressor__gbm__learning_rate': [0.09, 0.095, 0.1],  # Around default learning_rate=0.095
    }

    # Set up cross-validation
    cv = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize custom GridSearchCV with detailed logging
    grid_search = CustomGridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_jobs=-1,
    )

    # Perform grid search
    print(Fore.GREEN + Style.BRIGHT + "Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    print(Fore.GREEN + Style.BRIGHT + "Grid Search complete.")

    # Output best parameters and score
    print(Fore.YELLOW + Style.BRIGHT + f"Best parameters: {grid_search.best_params_}")
    print(Fore.YELLOW + Style.BRIGHT + f"Best score (Negative MAE): {-grid_search.best_score_:.4f}")

    # Use the best estimator to create submission
    best_model = grid_search.best_estimator_
    create_submission(best_model, X_test)


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
        ensemble.IsolationForest(contamination=0.05, random_state=42),
    )
    pred = model.fit_predict(X_train)
    X_train_clean = X_train[pred > 0]
    y_train_clean = y_train[pred > 0]
    return X_train_clean, y_train_clean


def preprocessing_pipeline():
    return pipeline.Pipeline([
        ('scaler', preprocessing.StandardScaler()),
        ('imputer', impute.SimpleImputer(strategy="median")),
    ])


def feature_selection_pipeline():
    return pipeline.Pipeline([
        ('variance_threshold', feature_selection.VarianceThreshold()),
        ('selectkbest', feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=200)),
        ('selectfrommodel', feature_selection.SelectFromModel(estimator=linear_model.Lasso(alpha=0.1))),
    ])


def model_pipeline():
    return ensemble.StackingRegressor(
        estimators=[
            ("svr", svm.SVR(C=60.0, epsilon=1e-05)),
            ("gbm", ensemble.GradientBoostingRegressor(learning_rate=0.095)),
            ("etr", ensemble.ExtraTreesRegressor()),
        ],
        final_estimator=linear_model.Ridge(),
        n_jobs=-1
    )


def create_submission(model, X_test):
    pred = model.predict(X_test)
    pred = np.vstack((np.arange(X_test.shape[0]), pred)).T
    print(Fore.CYAN + "Creating submission file...")
    pred = np.round(pred).astype(int)
    np.savetxt("submission_4_git_GRID_best_params.csv", pred, delimiter=",", header="id,y", comments="", fmt=["%d", "%d"])
    print(Fore.CYAN + "Submission file created.")


# Custom GridSearchCV with detailed logging
class CustomGridSearchCV(model_selection.GridSearchCV):
    def fit(self, X, y=None, **fit_params):
        start_time = time.time()
        total_candidates = len(list(model_selection.ParameterGrid(self.param_grid)))
        print(Fore.BLUE + Style.BRIGHT + f"Total parameter combinations to try: {total_candidates}")

        # Prepare the parameter grid
        param_grid = list(model_selection.ParameterGrid(self.param_grid))
        n_candidates = len(param_grid)

        results = []
        for index, params in enumerate(param_grid):
            iter_start_time = time.time()
            print(Fore.MAGENTA + Style.BRIGHT + f"\n{'='*80}")
            print(Fore.MAGENTA + Style.BRIGHT + f"Iteration {index + 1}/{n_candidates}")
            print(Fore.WHITE + Style.BRIGHT + "Testing parameters:")
            for key, value in params.items():
                print(Fore.YELLOW + f"  {key}: {value}")
            estimator = self.estimator.set_params(**params)
            cv_results = model_selection.cross_validate(
                estimator, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs, return_train_score=False
            )
            mean_score = cv_results['test_score'].mean()
            std_score = cv_results['test_score'].std()
            iter_time = time.time() - iter_start_time
            total_time = time.time() - start_time
            print(Fore.GREEN + f"Mean CV score (Negative MAE): {mean_score:.4f} ± {std_score:.4f}")
            print(Fore.GREEN + f"Iteration time: {iter_time:.2f} seconds")
            print(Fore.GREEN + f"Total time elapsed: {total_time:.2f} seconds")
            results.append({
                'params': params,
                'mean_test_score': mean_score,
                'std_test_score': std_score,
                'iter_time': iter_time,
                'total_time': total_time,
            })

        # Identify the best parameters
        best_index = np.argmax([res['mean_test_score'] for res in results])
        self.best_params_ = results[best_index]['params']
        self.best_score_ = results[best_index]['mean_test_score']
        self.best_estimator_ = self.estimator.set_params(**self.best_params_)
        # Fit the best estimator on the entire dataset
        self.best_estimator_.fit(X, y)
        self.cv_results_ = results
        return self


if __name__ == "__main__":
    main()
