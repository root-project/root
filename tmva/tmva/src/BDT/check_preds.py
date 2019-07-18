import numpy as np


b = np.genfromtxt("data_files/test.csv", delimiter=",")
a = np.genfromtxt("data_files/events.csv", delimiter=",")
print(f"Are the cpp and python events the same? \n {np.equal(a, b).all()}")

bb = np.genfromtxt("data_files/cpp_predictions.csv", delimiter=",")
aa = np.genfromtxt("data_files/python_predictions.csv", delimiter=",")
if aa.shape[0] != bb.shape[0]:
    print("cpp scores and python scores don't have the same number of rows")
# elif aa.shape[1] != bb.shape[1]:
#    print("cpp scores and python scores don't have the same number of columns")
else:
    print(f"Are the cpp and python predictions the same? \n {np.equal(aa, bb).all()}")
