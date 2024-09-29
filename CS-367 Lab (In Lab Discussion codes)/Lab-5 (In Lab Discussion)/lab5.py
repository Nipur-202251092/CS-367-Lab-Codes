import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np


data = pd.read_csv('2020_bn_nb_data.txt', sep='\t')


grade_mapping = {
    'AA': 10, 'AB': 9, 'BB': 8, 'BC': 7, 'CC': 6, 'CD': 5, 'DD': 4, 'F': 0
}
data_encoded = data.copy()
for column in data_encoded.columns[:-1]:  
    data_encoded[column] = data_encoded[column].map(grade_mapping)

print("Encoded Data Sample:")
print(data_encoded.head())

# model = BayesianNetwork([
#     ('EC100', 'QP'),
#     ('EC160', 'QP'),
#     ('IT101', 'QP'),
#     ('IT161', 'QP'),
#     ('MA101', 'QP'),
#     ('PH100', 'QP'),
#     ('PH160', 'QP'),
#     ('HS101', 'QP')
# ])

model = BayesianNetwork([
    ('EC100', 'PH100'),
    ('EC160', 'QP'),
    ('IT101', 'PH100'),
    ('IT161', 'QP'),
    ('MA101', 'PH100'),
    ('PH100', 'QP'),
    ('PH160', 'QP'),
    ('HS101', 'QP')
])

if model.get_cpds():
    for cpd in model.get_cpds():
        model.remove_cpds(cpd)

model.fit(data_encoded, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

given_grades = {
    'EC100': grade_mapping['DD'],
    'IT101': grade_mapping['CC'],
    'MA101': grade_mapping['CD']
}

ph100_grade = inference.map_query(variables=['PH100'], evidence=given_grades)
predicted_grade = {v: k for k, v in grade_mapping.items()}[ph100_grade['PH100']]
print("\nPredicted Grade for PH100:", predicted_grade)

X = data_encoded.iloc[:, :-1]  
y = data_encoded['QP'].apply(lambda x: 1 if x == 'y' else 0)  

accuracies = []

for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)

    y_pred = model_nb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print("\nAccuracy of Naive Bayes Classifier over 20 trials:")
print("Mean Accuracy:", np.mean(accuracies))
print("Standard Deviation of Accuracy:", np.std(accuracies))

bayesian_classifier_accuracies = []

for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(data_encoded, data_encoded['QP'].apply(lambda x: 1 if x == 'y' else 0), test_size=0.3, random_state=None)

    if model.get_cpds():
        for cpd in model.get_cpds():
            model.remove_cpds(cpd)

    model.fit(X_train, estimator=MaximumLikelihoodEstimator)

    y_pred_bayesian = []
    for index, row in X_test.iterrows():
        evidence = row[:-1].to_dict() 
        prediction = inference.map_query(variables=['QP'], evidence=evidence)
        y_pred_bayesian.append(1 if prediction['QP'] == 'y' else 0)

    accuracy_bayesian = accuracy_score(y_test, y_pred_bayesian)
    bayesian_classifier_accuracies.append(accuracy_bayesian)

print("\nAccuracy of Bayesian Classifier over 20 trials considering dependencies:")
print("Mean Accuracy:", np.mean(bayesian_classifier_accuracies))
print("Standard Deviation of Accuracy:", np.std(bayesian_classifier_accuracies))
