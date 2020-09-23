from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def classify_random_forest(data, test_data):
    instances, attr = data.shape
    attr -= 1
    x_train = data.iloc[:, 0:attr]
    y_train = data.iloc[:, attr]
    test_instances, test_attr = test_data.shape
    test_attr -= 1
    x_test = test_data.iloc[:, 0:test_attr]
    y_test = test_data.iloc[:, test_attr]
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print("Accuracy obtained using Random Forests is",accuracy)