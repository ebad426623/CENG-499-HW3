import pickle
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import time


dataset, labels = pickle.load(open("../datasets/part3_dataset.data", "rb"))


classifiers = {
    "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 7], "model__weights": ["uniform", "distance"], "model__metric": ["euclidean", "manhattan"]}),
    "SVM": (SVC(), {"model__C": [1, 10], "model__kernel": ["poly", "rbf"]}),
    "Decision Tree": (DecisionTreeClassifier(), {"model__criterion": ["entropy", "gini"], "model__max_depth": [10, 50]}),
    "Random Forest": (RandomForestClassifier(), {"model__n_estimators": [50, 100], "model__criterion": ["entropy", "gini"]}),
    "MLP": (MLPClassifier(), {"model__hidden_layer_sizes": [(30,)], "model__activation": ["relu", "tanh"], "model__learning_rate_init": [0.1, 1]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"model__n_estimators": [50], "model__loss": ["log_loss"], "model__learning_rate": [0.05, 0.1]})
}


stochastic_models = {"Random Forest", "MLP", "Gradient Boosting"}

outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=42)
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

final = {}
for name, (model, param_grid) in classifiers.items():
    print(f"Evaluating {name}")
    start = time.time()

    pipeline = Pipeline([
        ("scaler", MinMaxScaler(feature_range=(-1, 1))), 
        ("model", model)
        ])
    
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        scoring="f1_micro",
        cv=inner_cv, 
        n_jobs=-1
    )
    
       

    f1scores = []
    final_f1_score_of_model = []

    for train_idx, test_idx in outer_cv.split(dataset, labels):

        Train_Data, Test_Data = dataset[train_idx], dataset[test_idx]
        Train_Label, Test_Label = labels[train_idx], labels[test_idx]

        temp_score = []

        if name in stochastic_models:
            # For stochastic models
            run_scores = []
            for _ in range(5):
                grid_search.fit(Train_Data, Train_Label)
                score = grid_search.cv_results_["mean_test_score"]
                run_scores.append(score)
                
            run_scores = np.array(run_scores)
            temp_score = np.mean(run_scores, axis=0)
        
        else:
            # For non-stochastic models
            grid_search.fit(Train_Data, Train_Label)
            temp_score = grid_search.cv_results_["mean_test_score"]

        
        best_para_for_cv = grid_search.cv_results_["params"][np.argmax(temp_score)]
        f1scores.append(temp_score)

        outerPipe = Pipeline([
            ("scaler", MinMaxScaler(feature_range=(-1, 1))), 
            ("model", model)
        ])        

        outerPipe.set_params(**best_para_for_cv)
        outerPipe.fit(Train_Data, Train_Label)
        predicted = outerPipe.predict(Test_Data)
        
        final_f1_score_of_model.append(f1_score(Test_Label, predicted, average = "micro"))

        
    
    # These are the f1 scores from each
    f1scores = np.array(f1scores)

    mean_f1_score_final = np.mean(f1scores, axis=0)
    std_f1_score_final = np.std(f1scores, axis=0)
    final_interval = 1.96 * std_f1_score_final / np.sqrt(outer_cv.get_n_splits())

    configs = grid_search.cv_results_["params"]

    best_config = None
    best_f1 = 0
    best_interval = 0

    for i, config in enumerate(configs):
        print(f"Configuration {config}, Mean F1 Score: {mean_f1_score_final[i]:.4f} {u"\u00B1"} {final_interval[i]:.4f}")
        if(mean_f1_score_final[i] > best_f1):
            best_config = config
            best_f1 = mean_f1_score_final[i]
            best_interval = final_interval[i]

    print()
    print(f"Best Configuration {best_config}, Mean F1 Score: {best_f1:.4f} {u"\u00B1"} {best_interval:.4f}")

    final_final_f1_mean = np.mean(final_f1_score_of_model)
    final_final_interval = 1.96 * np.std(final_final_f1_mean)/np.sqrt(len(final_f1_score_of_model))
    
    print(f"Average F1 Score for {name}: {final_final_f1_mean:.4f} {u"\u00B1"} {final_final_interval:.4f}")
    print(f"Total Time Taken: {(time.time() - start):.2f}") 
    print()
    print()