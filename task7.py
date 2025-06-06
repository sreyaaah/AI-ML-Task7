import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("breast-cancer.csv")

X = data[['radius_mean', 'texture_mean']] 
y = data['diagnosis'].map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train, y_train)

svm_rbf = SVC(kernel='rbf', C=1, gamma=0.5)
svm_rbf.fit(X_train, y_train)

def plot_decision_boundary(model, X, y, title, bg_color='YlGnBu', dot_color='Set1'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.get_cmap(bg_color))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.get_cmap(dot_color))
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_decision_boundary(svm_linear, X_train, y_train, "SVM with Linear Kernel", bg_color='PuBuGn', dot_color='Set2')
plot_decision_boundary(svm_rbf, X_train, y_train, "SVM with RBF Kernel", bg_color='BuPu', dot_color='Dark2')

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [1, 0.1, 0.01],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5, verbose=1)
grid.fit(X_train, y_train)

print("\nBest Parameters from Grid Search:", grid.best_params_)
print("Best Cross-validation Score:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("\nAccuracy on test set:", accuracy_score(y_test, y_pred))

X_all = np.vstack((X_train, X_test))
y_all = np.hstack((y_train, y_test))
scores = cross_val_score(best_model, X_all, y_all, cv=5)
print("\nCross-validation scores:", scores)
print("Mean cross-validation accuracy:", scores.mean())
