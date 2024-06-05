import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint

file_path = os.path.join(os.path.dirname(__file__), 'veri-seti.txt')

data = pd.read_csv(file_path, delimiter='\t', header=None)
data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

imputer = SimpleImputer(strategy='mean')
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.fit_transform(data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_reg = LogisticRegression()
tree_clf = DecisionTreeClassifier()
forest_clf = RandomForestClassifier()
svm_clf = SVC()


param_dist_lr = {
    'C': uniform(0.01, 10),
    'solver': ['lbfgs', 'liblinear']
}
random_search_lr = RandomizedSearchCV(log_reg, param_dist_lr, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_lr.fit(X_train, y_train)
best_lr = random_search_lr.best_estimator_

param_dist_tree = {
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
random_search_tree = RandomizedSearchCV(tree_clf, param_dist_tree, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_tree.fit(X_train, y_train)
best_tree = random_search_tree.best_estimator_

param_dist_forest = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
random_search_forest = RandomizedSearchCV(forest_clf, param_dist_forest, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_forest.fit(X_train, y_train)
best_forest = random_search_forest.best_estimator_

param_dist_svm = {
    'C': uniform(0.1, 10),
    'gamma': uniform(0.01, 1),
    'kernel': ['rbf']
}
random_search_svm = RandomizedSearchCV(svm_clf, param_dist_svm, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_svm.fit(X_train, y_train)
best_svm = random_search_svm.best_estimator_

y_pred_lr = best_lr.predict(X_test)
print("Lojistik Regresyon Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

y_pred_tree = best_tree.predict(X_test)
print("Karar Ağacı Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

y_pred_forest = best_forest.predict(X_test)
print("Rastgele Orman Accuracy:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))

y_pred_svm = best_svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))
