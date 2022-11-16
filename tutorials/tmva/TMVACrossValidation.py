### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
### This macro provides an example of how to use TMVA for k-folds cross evaluation.
###
### As input data is used a toy-MC sample consisting of two gaussian
### distributions.
###
### The output file "TMVA.root" can be analysed with the use of dedicated
### macros (simply say: root -l <macro.C>), which can be conveniently
### invoked through a GUI that will appear at the end of the run of this macro.
### Launch the GUI via the command:
###
### ```
### root -l -e 'TMVA::TMVAGui("TMVA.root")'
### ```
###
### ## Cross Evaluation
### Cross evaluation is a special case of k-folds cross validation where the
### splitting into k folds is computed deterministically. This ensures that the
### a given event will always end up in the same fold.
###
### In addition all resulting classifiers are saved and can be applied to new
### data using `MethodCrossValidation`. One requirement for this to work is a
### splitting function that is evaluated for each event to determine into what
### fold it goes (for training/evaluation) or to what classifier (for
### application).
###
### ## Split Expression
### Cross evaluation uses a deterministic split to partition the data into
### folds called the split expression. The expression can be any valid
### `TFormula` as long as all parts used are defined.
###
### For each event the split expression is evaluated to a number and the event
### is put in the fold corresponding to that number.
###
### It is recommended to always use `%int([NumFolds])` at the end of the
### expression.
###
### The split expression has access to all spectators and variables defined in
### the dataloader. Additionally, the number of folds in the split can be
### accessed with `NumFolds` (or `numFolds`).
###
### ### Example
###  ```
###  "int(fabs([eventID]))%int([NumFolds])"
###  ```
###
### - Project   : TMVA - a ROOT-integrated toolkit for multivariate data analysis
### - Package   : TMVA
### - Root Macro: TMVACrossValidation
###
### \macro_output
### \macro_code
### \author Harshal Shende


import ROOT


# Helper function to load data into TTrees.
def genTree(nPoints, offset, scale, seed=100):
    rng = ROOT.TRandom(seed)
    x = 0
    y = 0
    eventID = 0
    data = ROOT.TTree()
    data.Branch("x", x, "x/F")
    data.Branch("y", y, "y/F")
    data.Branch("eventID", eventID, "eventID/I")

    for _ in range(1, nPoints):
        x = rng.Gaus(offset, scale)
        y = rng.Gaus(offset, scale)
        # For our simple example it is enough that the id's are uniformly
        # distributed and independent of the data.
        eventID += 1
        data.Fill()

    # Important: Disconnects the tree from the memory locations of x and y.
    data.ResetBranchAddresses()
    return data


useRandomSplitting = False

# This loads the library
ROOT.TMVA.Tools.Instance()

# --------------------------------------------------------------------------

# Load the data into TTrees. If you load data from file you can use a
# variant of
# ```
# TString filename = "/path/to/file";
# TFile * input = TFile::Open( filename );
# TTree * signalTree = (TTree*)input->Get("TreeName");
# ```
sigTree = genTree(1000, 1.0, 1.0, 100)
bkgTree = genTree(1000, -1.0, 1.0, 101)

# Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outfileName = "TMVA.root"
outputFile = ROOT.TFile.Open(outfileName, "RECREATE")

# DataLoader definitions; We declare variables in the tree so that TMVA can
# find them. For more information see TMVAClassification tutorial.
dataloader = ROOT.TMVA.DataLoader("dataset")

# Data variables
dataloader.AddVariable("x", "F")
dataloader.AddVariable("y", "F")

# Spectator used for split
dataloader.AddSpectator("eventID", "I")

# NOTE: Currently TMVA treats all input variables, spectators etc as
#       floats. Thus, if the absolute value of the input is too large
#       there can be precision loss. This can especially be a problem for
#       cross validation with large event numbers.
#       A workaround is to define your splitting variable as:
#           `dataloader->AddSpectator("eventID := eventID % 4096", 'I');`
#       where 4096 should be a number much larger than the number of folds
#       you intend to run with.

# Attaches the trees so they can be read from
dataloader.AddSignalTree(sigTree, 1.0)
dataloader.AddBackgroundTree(bkgTree, 1.0)

# The CV mechanism of TMVA splits up the training set into several folds.
# The test set is currently left unused. The `nTest_ClassName=1` assigns
# one event to the test set for each class and puts the rest in the
# training set. A value of 0 is a special value and would split the
# datasets 50 / 50.
dataloader.PrepareTrainingAndTestTree(
    "",
    "",
    nTest_Signal=1,
    nTest_Background=1,
    SplitMode="Random",
    NormMode="NumEvents",
    V=False,
)

# --------------------------------------------------------------------------

#
# This sets up a CrossValidation class (which wraps a TMVA::Factory
# internally) for 2-fold cross validation.
#
# The split type can be "Random", "RandomStratified" or "Deterministic".
# For the last option, check the comment below. Random splitting randomises
# the order of events and distributes events as evenly as possible.
# RandomStratified applies the same logic but distributes events within a
# class as evenly as possible over the folds.
#
numFolds = 2
analysisType = "Classification"

splitType = "Random" if useRandomSplitting else "Deterministic"

#
# One can also use a custom splitting function for producing the folds.
# The example uses a dataset spectator `eventID`.
#
# The idea here is that eventID should be an event number that is integral,
# random and independent of the data, generated only once. This last
# property ensures that if a calibration is changed the same event will
# still be assigned the same fold.
#
# This can be used to use the cross validated classifiers in application,
# a technique that can simplify statistical analysis.
#
# If you want to run TMVACrossValidationApplication, make sure you have
# run this tutorial with Deterministic splitting type, i.e.
# with the option useRandomSPlitting = false
#

splitExpr = "int(fabs([eventID]))%int([NumFolds])" if not useRandomSplitting else ""


cv = ROOT.TMVA.CrossValidation(
    "TMVACrossValidationRegression",
    dataloader,
    outputFile,
    V=False,
    Silent=False,
    ModelPersistence=True,
    FoldFileOutput=False,
    AnalysisType=analysisType,
    SplitType=splitType,
    NumFolds=2,
    SplitExpr=splitExpr,
)

# --------------------------------------------------------------------------

#
# Books a method to use for evaluation
#
cv.BookMethod(
    ROOT.TMVA.Types.kBDT,
    "BDTG",
    H=False,
    V=False,
    NTrees=100,
    MinNodeSize="2.5%",
    BoostType="Grad",
    NegWeightTreatment="Pray",
    Shrinkage=0.10,
    nCuts=20,
    MaxDepth=2
)

cv.BookMethod(
    ROOT.TMVA.Types.kFisher,
    "Fisher",
    H=False,
    V=False,
    Fisher=True,
    VarTransform="None",
)

# --------------------------------------------------------------------------

#
# Train, test and evaluate the booked methods.
# Evaluates the booked methods once for each fold and aggregates the result
# in the specified output file.
#
cv.Evaluate()

# --------------------------------------------------------------------------

#
# Process some output programatically, printing the ROC score for each
# booked method.
#
iMethod = 0
for result in cv.GetResults():
    iMethod += 1
    print("Summary for method {}".format(cv.GetMethods()[iMethod].GetValue("MethodName")))
    for iFold in range(cv.GetNumFolds()):
        print("Fold {} :".format(iFold))
        print(
            "ROC int: {}".format(result.GetROCValues()[iFold]),
            ", BkgEff@SigEff=0.3: {}".format(result.GetEff30Values()[iFold]),
        )


# --------------------------------------------------------------------------

#
# Save the output
#
outputFile.Close()

print("==> Wrote root file: {}".format(outputFile.GetName()))
print("==> TMVACrossValidation is done!")

# --------------------------------------------------------------------------

#
# Launch the GUI for the root macros
#
if not ROOT.gROOT.IsBatch():
    # Draw cv-specific graphs
    cv.GetResults()[0].DrawAvgROCCurve(True, "Avg ROC for BDTG")
    cv.GetResults()[0].DrawAvgROCCurve(True, "Avg ROC for Fisher")

    # You can also use the classical gui
    ROOT.TMVA.TMVAGui(outfileName)
