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
### - Root Macro: TMVACrossValidationRegression
###
### \macro_output
### \macro_code
### \author Kim Albertsson (adapted from code originally by Andreas Hoecker)

import ROOT


def getDataFile(fname):
    if ROOT.gSystem.AccessPathName(fname):
        ROOT.TFile.SetCacheFileDir(".")
        input = ROOT.TFile.Open("http://root.cern.ch/files/tmva_reg_example.root", "CACHEREAD")
        if input is None:
            raise FileNotFoundError("Input file cannot be downloaded - exit")
    else:
        input = ROOT.TFile.Open(fname)
        if input is None:
            raise FileNotFoundError("ERROR: could not open data file ")

    return input


ROOT.TMVA.Tools.Instance()

# --------------------------------------------------------------------------

# Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outfileName = "TMVARegCv.root"
outputFile = ROOT.TFile.Open(outfileName, "RECREATE")

infileName = "tmva_reg_example.root"
inputFile = getDataFile(infileName)

dataloader = ROOT.TMVA.DataLoader("dataset")

dataloader.AddVariable("var1", "Variable 1", "units", "F")
dataloader.AddVariable("var2", "Variable 2", "units", "F")

# Add the variable carrying the regression target
dataloader.AddTarget("fvalue")

regTree = inputFile.Get("TreeR")
dataloader.AddRegressionTree(regTree, 1.0)

# Individual events can be weighted
# dataloader.SetWeightExpression("weight", "Regression")

print("--- TMVACrossValidationRegression: Using input file: {}".format(inputFile.GetName()))

# Bypasses the normal splitting mechanism, CV uses a new system for this.
# Unfortunately the old system is unhappy if we leave the test set empty so
# we ensure that there is at least one event by placing the first event in
# it.
# You can with the selection cut place a global cut on the defined
# variables. Only events passing the cut will be using in training/testing.
# Example: `TCut selectionCut = "var1 < 1"`
dataloader.PrepareTrainingAndTestTree("", nTest_Regression=1, SplitMode="Block", NormMode="NumEvents", V=False)

# --------------------------------------------------------------------------

#
# This sets up a CrossValidation class (which wraps a TMVA::Factory
# internally) for 2-fold cross validation. The data will be split into the
# two folds randomly if `splitExpr` is `""`.
#
# One can also give a deterministic split using spectator variables. An
# example would be e.g. `"int(fabs([spec1]))%int([NumFolds])"`.

numFolds = 2
analysisType = "Regression"
splitExpr = ""


cv = ROOT.TMVA.CrossValidation(
    "TMVACrossValidationRegression",
    dataloader,
    outputFile,
    V=False,
    Silent=False,
    ModelPersistence=True,
    FoldFileOutput=False,
    AnalysisType=analysisType,
    NumFolds=numFolds,
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
    NTrees=500,
    BoostType="Grad",
    Shrinkage=0.1,
    UseBaggedBoost=True,
    BaggedSampleFraction=0.5,
    nCuts=20,
    MaxDepth=3,
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
# Save the output
#
outputFile.Close()

print("==> Wrote root file: {}".format(outputFile.GetName()))
print("==> TMVACrossValidationRegression is done!")

# --------------------------------------------------------------------------

#
# Launch the GUI for the root macros
#
if ROOT.gROOT.IsBatch():
    ROOT.TMVA.TMVAGui(outfileName)
