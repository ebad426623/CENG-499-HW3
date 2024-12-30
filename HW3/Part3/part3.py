import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle

# Load dataset
dataset, labels = pickle.load(open("../datasets/part3_dataset.data", "rb"))

# Normalization function
def normalize_data(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# Define classifiers and their hyperparameter grids
classifiers = {
    # "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    # "SVM": (SVC(), {"C": [1, 10], "kernel": ["poly", "rbf"]}),
    # "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [None, 10]}),
    # "Random Forest": (RandomForestClassifier(), {"n_estimators": [100, 200], "max_depth": [10, None]}),
    # "MLP": (MLPClassifier(max_iter=500), {"hidden_layer_sizes": [(24,)], "alpha": [0.0001, 0.1]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}),
}

# Nested cross-validation setup
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state = 42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state = 42)

# Evaluate all classifiers
results = {}
for name, (model, param_grid) in classifiers.items():
    print(f"Evaluating {name}...")
    
    
    grid_search = GridSearchCV(
        estimator = model, 
        param_grid = param_grid, 
        cv = inner_cv, 
        scoring = make_scorer(f1_score, average='weighted'),
        n_jobs = -1)
    
    scores = []
    
    for train_idx, test_idx in outer_cv.split(dataset, labels):
        X_train, X_test = dataset[train_idx], dataset[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        
        X_train, X_test = normalize_data(X_train, X_test)
        
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        
        f1 = f1_score(y_test, y_pred, average='weighted')
        scores.append(f1)
    
    
    results[name] = {
        "mean_f1": np.mean(scores),
        "std_f1": np.std(scores),
        "confidence_interval": (np.mean(scores) - 1.96 * np.std(scores), np.mean(scores) + 1.96 * np.std(scores)),
        "best_params": grid_search.best_params_,
    }


for name, metrics in results.items():
    print(f"\n{name}")
    print(f"Mean F1-score: {metrics['mean_f1']:.4f}")
    print(f"Confidence Interval: {metrics['confidence_interval']}")
    print(f"Best Hyperparameters: {metrics['best_params']}")
