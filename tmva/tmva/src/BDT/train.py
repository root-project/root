# First XGBoost model for Pima Indians dataset
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb
import timeit

setup = """
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb

# load data
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

# split data into X and y
num_features = 5  # max is 8
X = dataset[:, 0:num_features]
Y = dataset[:, 8]
# print(X)
# print(Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_size, random_state=seed
)

# fit model no training data
model = xgb.XGBClassifier(n_estimators=10)
model.fit(X_train, y_train)

# make predictions for test data
"""

it_number = 100_000
my_time = timeit.timeit("model.predict(X_test)", setup=setup, number=it_number)
print(f"{my_time / it_number * 1000}  ms")
# load data
dataset = np.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

# split data into X and y
num_features = 5  # max is 8
X = dataset[:, 0:num_features]
Y = dataset[:, 8]
# print(X)
# print(Y)

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=test_size, random_state=seed
)

# fit model no training data
model = xgb.XGBClassifier(n_estimators=10)
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
y_scores = model.apply(X_test)

# from eli5 import show_weights

# show_weights(model, vec=vec)

print(X_test)

# print(y_pred)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

model.get_booster().dump_model("model.json", dump_format="json")
# xgboostModel.booster.saveModel("/tmp/xgbm")

fig, ax = plt.subplots(figsize=(30, 30))
xgb.plot_tree(model, ax=ax)
# xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
plt.savefig("temp.pdf")

np.savetxt("data_files/events.csv", X_test, delimiter=",", fmt="%f")
np.savetxt("data_files/python_predictions.csv", y_pred, delimiter=",", fmt="%d")
np.savetxt("data_files/python_groundtruths.csv", y_test, delimiter=",", fmt="%d")
np.savetxt("data_files/python_scores.csv", y_scores, delimiter=",", fmt="%f")


for idx in range(20):
    in_ = X[idx]
    out_ = model.predict_proba(np.array([X[idx]]))
    print(f"{in_.squeeze()} -> {out_.squeeze()}")
