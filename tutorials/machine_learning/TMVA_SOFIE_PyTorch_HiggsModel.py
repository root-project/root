### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### This macro trains a simple deep neural network on the Higgs dataset with
### PyTorch, exports the model to ONNX and runs the SOFIE parser on it to
### generate and compile C++ inference code.
###
### The trained model is saved as HiggsModel.onnx and is used as input by
### other SOFIE tutorials (e.g. TMVA_SOFIE_RDataFrame.C), so this macro needs
### to be run before them.
###
### The PyTorch export and ROOT's SOFIE parser are both linked against protobuf,
### but usually against different versions, so loading them in the same process
### leads to a symbol clash. We therefore run the PyTorch training and ONNX
### export in a separate Python process and only use ROOT before and afterwards.
###
### \macro_code
### \macro_output

import os
import subprocess
import sys

import numpy as np
import ROOT

# The PyTorch training and ONNX export, as a small standalone script run in its
# own process. It takes as arguments the .npz file with the training data and
# the model name, and writes <modelName>.onnx together with the PyTorch
# predictions for the validation inputs in <modelName>_torch_output.npy.
TRAIN_SCRIPT = r"""
import sys
import inspect
import warnings
import contextlib

import numpy as np
import torch
import torch.nn as nn

dataFile = sys.argv[1]
modelName = sys.argv[2]


@contextlib.contextmanager
def expect_warning(category, message):
    # Silence a known third-party warning and raise if it stops firing.

    # Notifies us to drop the workaround once the upstream library is fixed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    seen = False
    for w in caught:
        if issubclass(w.category, category) and message in str(w.message):
            seen = True
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if not seen:
        raise RuntimeError(
            f"Expected {category.__name__} containing {message!r} was not "
            "emitted. This tutorial's workaround can probably be removed."
        )


def CreateModel(nlayers=4, nunits=64):
    layers = []
    ninputs = 7
    for i in range(1, nlayers):
        layers += [nn.Linear(ninputs, nunits), nn.ReLU()]
        ninputs = nunits
    layers += [nn.Linear(ninputs, 1), nn.Sigmoid()]
    model = nn.Sequential(*layers)
    print(model)
    return model


def TrainModel(model, x, y, epochs=5, batch_size=50):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    nbatches = x.shape[0] // batch_size
    for epoch in range(epochs):
        perm = torch.randperm(x.shape[0])
        running_loss = 0.0
        for i in range(nbatches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            optimizer.zero_grad()
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - average loss: {running_loss / nbatches:.4f}")


def ExportModel(model, modelName):
    # need to evaluate the model before exporting to ONNX
    # and to provide a dummy input tensor to set the input model shape
    # (the batch size is fixed to 1 for the SOFIE inference)
    model.eval()

    modelFile = modelName + ".onnx"
    dummy_x = torch.randn(1, 7)
    model(dummy_x)

    # check for torch.onnx.export parameters
    def filtered_kwargs(func, **candidate_kwargs):
        sig = inspect.signature(func)
        return {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}

    kwargs = filtered_kwargs(
        torch.onnx.export,
        input_names=["input"],
        output_names=["output"],
        external_data=False,  # may not exist
        dynamo=True,  # may not exist
    )
    print("calling torch.onnx.export with parameters", kwargs)

    try:
        # torch.onnx.export (dynamo path) pickles its export program through
        # copyreg, which still references the deprecated LeafSpec. The warning
        # is emitted from inside PyTorch and cannot be avoided from user code.
        with expect_warning(FutureWarning, "isinstance(treespec, LeafSpec)"):
            torch.onnx.export(model, dummy_x, modelFile, **kwargs)
        print("model exported to ONNX as", modelFile)
    except TypeError:
        print("Cannot export model from pytorch to ONNX - with version ", torch.__version__)
        # leave no .onnx behind: which the parent process treats as a RuntimeError
        sys.exit()


data = np.load(dataFile)

# create dense model with 3 layers of 64 units and train it
model = CreateModel(3, 64)
TrainModel(model, data["x_train"], data["y_train"])
ExportModel(model, modelName)

# evaluate the trained model on the validation inputs, for comparison with SOFIE
with torch.no_grad():
    y = model(torch.from_numpy(data["x_check"])).numpy()
np.save(modelName + "_torch_output.npy", y)
"""


def PrepareData():
    # get the input data
    inputFile = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"

    df1 = ROOT.RDataFrame("sig_tree", inputFile)
    sigData = df1.AsNumpy(columns=["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"])
    # print(sigData)

    # stack all the 7 numpy array in a single array (nevents x nvars)
    xsig = np.column_stack(list(sigData.values()))
    data_sig_size = xsig.shape[0]
    print("size of data", data_sig_size)

    # make SOFIE inference on background data
    df2 = ROOT.RDataFrame("bkg_tree", inputFile)
    bkgData = df2.AsNumpy(columns=["m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"])
    xbkg = np.column_stack(list(bkgData.values()))
    data_bkg_size = xbkg.shape[0]

    ysig = np.ones(data_sig_size)
    ybkg = np.zeros(data_bkg_size)
    inputs_data = np.concatenate((xsig, xbkg), axis=0).astype(np.float32)
    inputs_targets = np.concatenate((ysig, ybkg), axis=0).astype(np.float32)

    # split data in training and test data
    rng = np.random.default_rng(1234)
    idx = rng.permutation(inputs_data.shape[0])
    ntrain = inputs_data.shape[0] // 2

    x_train = inputs_data[idx[:ntrain]]
    y_train = inputs_targets[idx[:ntrain]].reshape(-1, 1)
    x_test = inputs_data[idx[ntrain:]]
    y_test = inputs_targets[idx[ntrain:]].reshape(-1, 1)

    return x_train, y_train, x_test, y_test


def TrainModel(x_train, y_train, x_check, name):
    # train the model with PyTorch and export it to ONNX
    # (done in a separate process to avoid the protobuf clash, see above)
    dataFile = name + "_train_data.npz"
    np.savez(dataFile, x_train=x_train, y_train=y_train, x_check=x_check)

    modelFile = name + ".onnx"
    torchOutputFile = name + "_torch_output.npy"
    subprocess.run([sys.executable, "-c", TRAIN_SCRIPT, dataFile, name], check=True)
    os.remove(dataFile)
    if not os.path.exists(modelFile) or not os.path.exists(torchOutputFile):
        raise RuntimeError("ONNX model could not be exported")

    ytorch = np.load(torchOutputFile)
    os.remove(torchOutputFile)
    return modelFile, ytorch


def GenerateCode(modelFile="model.onnx"):

    # check if the input file exists
    if not os.path.exists(modelFile):
        raise FileNotFoundError("Input model file is missing. The PyTorch training did not produce " + modelFile)

    # parse the input ONNX model into an RModel object
    parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
    model = parser.Parse(modelFile)

    # Generating inference code
    model.Generate()
    model.OutputGenerated()

    modelName = modelFile.replace(".onnx", "")
    return modelName


###################################################################
## Step 1 : Create and train the model, export it to ONNX
###################################################################

x_train, y_train, x_test, y_test = PrepareData()
# validate the exported model on the first test events
x_check = x_test[:10]
modelFile, ytorch = TrainModel(x_train, y_train, x_check, "HiggsModel")

###################################################################
## Step 2 : Parse model and generate inference code with SOFIE
###################################################################

modelName = GenerateCode(modelFile)
modelHeaderFile = modelName + ".hxx"

###################################################################
## Step 3 : Compile the generated C++ model code
###################################################################

ROOT.gInterpreter.Declare('#include "' + modelHeaderFile + '"')

###################################################################
## Step 4: Evaluate the model
###################################################################

# get first the SOFIE session namespace
sofie = getattr(ROOT, "TMVA_SOFIE_" + modelName)
session = sofie.Session()

for i in range(x_check.shape[0]):
    y = session.infer(x_check[i])
    print("input to model is ", x_check[i], "\n\t -> output using SOFIE = ", y[0], " using PyTorch = ", ytorch[i, 0])
    if abs(y[0] - ytorch[i, 0]) > 0.01:
        raise RuntimeError("ERROR: Result is different between SOFIE and PyTorch")

print("OK")
