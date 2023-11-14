import pandas as pd
df = pd.read_csv('anomaly detection/creditcard.csv')
print("Shape of the dataset:", df.shape)



from sklearn.preprocessing import StandardScaler
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))



from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.01), max_features=1.0, random_state=42)
clf.fit(df)
y_pred = clf.predict(df)
y_pred = y_pred.reshape(-1,1)
print("Number of outliers:", len(df[y_pred == -1]))



from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20, contamination=float(0.01))
y_pred = clf.fit_predict(df)
y_pred = y_pred.reshape(-1,1)
print("Number of outliers:", len(df[y_pred == -1]))




from sklearn.svm import OneClassSVM
clf = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.01)
clf.fit(df)
y_pred = clf.predict(df)
y_pred = y_pred.reshape(-1,1)
print("Number of outliers:", len(df[y_pred == -1]))



from sklearn.model_selection import train_test_split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
classifiers = [LogisticRegression(), DecisionTreeClassifier()]
lr_params = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
dt_params = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 7]}
rf_params = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7]}
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
param_grids = [lr_params, dt_params, rf_params, knn_params]
for i, classifier in enumerate(classifiers):
    clf = GridSearchCV(classifier, param_grids[i], cv=5)
    clf.fit(X_train, y_train)
    print(classifier.__class__.__name__)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {rec}")
print(f"F1 Score: {f1}")