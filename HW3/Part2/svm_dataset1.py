import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset, labels = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))


kernels = ["linear", "poly", "rbf", "sigmoid"]
C = [0.1, 1, 10]


def model_display_boundary(X, model, label, title, filename):
    h = .01  # step size in the mesh, we can decrease this value for smooth plots, i.e 0.01 (but ploting may slow down)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 3, X[:, 0].max() + 3
    y_min, y_max = X[:, 1].min() - 3, X[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    aa = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(aa)

    Z = Z.reshape(xx.shape)
    # plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
    plt.contourf(xx, yy, Z, alpha=0.25) # cmap="Paired_r",
    # plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=label, cmap="Paired_r", edgecolors='k');
    x_ = np.array([x_min, x_max])

    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    plt.savefig(filename)  
    plt.close() 



for kernel in kernels:
    for C_val in C:
        print(f"Training configuration: C={C_val}, kernel={kernel}")
        svm = SVC(C=C_val, kernel=kernel)
        svm.fit(dataset, labels)


        predicted = svm.predict(dataset)
        accuracy = accuracy_score(labels, predicted) * 100
        print(f"Accuracy : {accuracy:.2f}%")


        title = f"SVM Decision Boundary (C={C_val}, kernel={kernel})"
        filename = f"svm_plot_C{C_val}_kernel_{kernel}.png"
        model_display_boundary(dataset, svm, labels, title, filename)
        print()