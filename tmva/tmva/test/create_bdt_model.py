import numpy as np
np.random.seed(1234)
import os
import uuid
import json
from xgboost import XGBClassifier
import ROOT


if __name__ == "__main__":
    num_events = 10000
    x = np.random.normal(0.0, 1.0, (num_events, 1))
    y = np.random.choice(a=[0, 1], size=(num_events), p=[0.3, 0.7])

    model = XGBClassifier(n_estimators=1, max_depth=0)
    model.fit(x, y)

    ROOT.TMVA.Experimental.SaveXGB(model, "myModel", "/tmp/model.root")
