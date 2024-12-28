import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

estimator = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])

param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__kernel": ["linear", "poly", "rbf", "sigmoid"]
}

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)


grid = GridSearchCV(
    estimator=estimator,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

grid.fit(dataset, labels)
results = grid.cv_results_

final = []
for i, mean in enumerate(results["mean_test_score"]):
    std = results["std_test_score"][i]
    params = results["params"][i]
    interval = 1.96 * std / cv.get_n_splits()
    final.append((params, mean, interval))


best_C = 0
best_kernel = None
best_acc = 0
best_interval = 0
index = 1

for params, mean, interval in final:
    C = params["svc__C"]
    kernel = params["svc__kernel"]
    
    if mean > best_acc:
        best_acc = mean
        best_kernel = kernel
        best_C = C
        best_interval = interval

    print(f"Configuration {index}:  [Kernel: {kernel}, C: {C}], Accuracy: {(mean*100):.2f} {u"\u00B1"} {(interval*100):.4f}")
    index += 1

print()
print(f"Best Configuration: [Kernel: {best_kernel}, C: {best_C}], Accuracy: {(best_acc*100):.2f} {u"\u00B1"} {(best_interval*100):.4f}")