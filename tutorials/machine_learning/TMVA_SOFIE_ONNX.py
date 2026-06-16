## \file
## \ingroup tutorial_ml
## \notebook -nodraw
## This macro provides a simple example for:
##  - creating a model with Pytorch and export to ONNX
##  - parsing the ONNX file with SOFIE and generate C++ code
##  - compiling the model using ROOT Cling
##  - run the code and optionally compare with ONNXRuntime
##
## The PyTorch export and ROOT's SOFIE parser are both linked against protobuf,
## but usually against different versions, so loading them in the same process
## leads to a symbol clash. We therefore run the PyTorch -> ONNX export in a
## separate Python process and only import ROOT afterwards.
##
## \macro_code
## \macro_output
## \author Lorenzo Moneta

import os
import sys
import subprocess

import numpy as np
import ROOT


# The PyTorch export, as a small standalone script run in its own process.
# It takes the model name as its only argument and writes <modelName>.onnx.
EXPORT_SCRIPT = r"""
import sys
import inspect
import warnings
import contextlib

import torch
import torch.nn as nn

modelName = sys.argv[1]


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


def CreateAndTrainModel(modelName):

    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2), nn.Softmax(dim=1))

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train model with the random data
    for i in range(500):
        x = torch.randn(2, 32)
        y = torch.randn(2, 2)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # *******************************************************
    ##  EXPORT to ONNX
    #
    #  need to evaluate the model before exporting to ONNX
    #  and to provide a dummy input tensor to set the input model shape
    model.eval()

    modelFile = modelName + ".onnx"
    dummy_x = torch.randn(1, 32)
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
        return modelFile
    except TypeError:
        print("Cannot export model from pytorch to ONNX - with version ", torch.__version__)
        # leave no .onnx behind: which the parent process treats as a RuntimeError
        sys.exit()

CreateAndTrainModel(modelName)
"""


def ParseModel(modelFile, verbose=False):

    parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
    model = parser.Parse(modelFile, verbose)
    #
    # print model weights
    if verbose:
        model.PrintInitializedTensors()
        data = model.GetTensorData["float"]("0weight")
        print("0weight", data)
        data = model.GetTensorData["float"]("2weight")
        print("2weight", data)

    # Generating inference code
    model.Generate()
    # generate header file (and .dat file) with modelName+.hxx
    model.OutputGenerated()
    if verbose:
        model.PrintGenerated()

    modelCode = modelFile.replace(".onnx", ".hxx")
    print("Generated model header file ", modelCode)
    return modelCode


###################################################################
## Step 1 : Create and train the model, export it to ONNX
##          (done in a separate process to avoid the protobuf clash)
###################################################################

# use an arbitrary modelName
modelName = "LinearModel"
modelFile = modelName + ".onnx"

subprocess.run([sys.executable, "-c", EXPORT_SCRIPT, modelName])
if not os.path.exists(modelFile):
    raise RuntimeError("ONNX model could not be exported")


###################################################################
## Step 2 : Parse model and generate inference code with SOFIE
###################################################################

modelCode = ParseModel(modelFile, False)

###################################################################
## Step 3 : Compile the generated C++ model code
###################################################################

ROOT.gInterpreter.Declare('#include "' + modelCode + '"')

###################################################################
## Step 4: Evaluate the model
###################################################################

# get first the SOFIE session namespace
sofie = getattr(ROOT, "TMVA_SOFIE_" + modelName)
session = sofie.Session()

x = np.random.normal(0, 1, (1, 32)).astype(np.float32)
print("\n************************************************************")
print("Running inference with SOFIE ")
print("\ninput to model is ", x)
y = session.infer(x)
# output shape is (1,2)
y_sofie = np.asarray(y.data())
print("-> output using SOFIE = ", y_sofie)

# check inference with onnx
try:
    import onnxruntime as ort

    # Load model
    print("Running inference with ONNXRuntime ")
    ort_session = ort.InferenceSession(modelFile)

    # Run inference
    outputs = ort_session.run(None, {"input": x})
    y_ort = outputs[0]
    print("-> output using ORT =", y_ort)

    testFailed = abs(y_sofie - y_ort) > 0.01
    if np.any(testFailed):
        raise RuntimeError("Result is different between SOFIE and ONNXRT")
    else:
        print("OK")

except ImportError:
    print("Missing ONNXRuntime: skipping comparison test")
