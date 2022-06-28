import ROOT
import numpy as np
import pandas as pd
import random

from helpers import map_col_to_sym
from filters import *
from DatasetCreation import CreateData, DeleteData

CreateData()


rdf = ROOT.RDataFrame("TestData", "./Data/TestData.root")
test_cols =[str(c) for c in rdf.GetColumnNames()]
df = pd.DataFrame(rdf.AsNumpy())
for col_name in test_cols:
    filter = "filter_" + map_col_to_sym[col_name]
    # print(str(filter))
    
    func = eval(filter); x = df[col_name].mean()
    if col_name == 'Bool_t': x = random.choice([True, False])
    # res_pd = df[df[col_name].apply(func, x=x)][col_name].to_numpy()
    # print(col_name)
    filtered = rdf.Filter(func, {"x":x},)
    res_root = filtered.AsNumpy()[col_name]
    if not isinstance(x, bool):
        filtered2 = rdf._OriginalFilter(f"{col_name} > {x}")
    else:
        if x:
            filtered2 = rdf._OriginalFilter(f"{col_name} == true")
        else:
            filtered2 = rdf._OriginalFilter(f"{col_name} == false")

    res_root2 = filtered2.AsNumpy()[col_name]
    truth_val = np.array(res_root != res_root2)
    truth_val = np.sum(truth_val)
    if truth_val == 0:
        print(f"PyFilter Works for {col_name}")
    else:
        print(f"PyFilter Failed for {col_name}")

remaining_cols = ["Long64_t", "ULong64_t", "Long_t", "ULong_t"]
for col_name in remaining_cols:
    file_name = "./Data/" + col_name + ".root"
    rdf = ROOT.RDataFrame(col_name, file_name)
    df = pd.DataFrame(rdf.AsNumpy())
    filter = "filter_" + map_col_to_sym[col_name]
    # print(str(filter))
    func = eval(filter); x = df[col_name].mean()
    if col_name == 'Bool_t': x = random.choice([True, False])
    # res_pd = df[df[col_name].apply(func, x=x)][col_name].to_numpy()
    filtered = rdf.Filter(func, {col_name:col_name, "x":x})
    res_root = filtered.AsNumpy()[col_name]
    if not isinstance(x, bool):
        filtered2 = rdf._OriginalFilter(f"{col_name} > {x}")
    else:
        if x:
            filtered2 = rdf._OriginalFilter(f"{col_name} == true")
        else:
            filtered2 = rdf._OriginalFilter(f"{col_name} == false")

    res_root2 = filtered2.AsNumpy()[col_name]
    truth_val = np.array(res_root != res_root2)
    truth_val = np.sum(truth_val)
    if truth_val == 0:
        print(f"PyFilter Works for {col_name}")

DeleteData()