## \file
## \ingroup tutorial_ml
## \notebook
## Classification example of TMVA based on public Higgs UCI dataset
##
##  The UCI data set is a public HIGGS data set , see http://archive.ics.uci.edu/ml/datasets/HIGGS
## used in this paper: Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics
##                     with Deep Learning.” Nature Communications 5 (July 2, 2014).
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Harshal Shende

## Declare Factory


## Create the Factory class. Later you can choose the methods
## whose performance you'd like to investigate.

## The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass

## - The first argument is the base of the name of all the output
## weightfiles in the directory weight/ that will be created with the
##    method parameters

## - The second argument is the output file for the training results

## - The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string

import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile

TMVA.Tools.Instance()

# options to control used methods
useLikelihood = True  # likelihood based discriminant
useLikelihoodKDE = False  # likelihood based discriminant
useFischer = True  # Fischer discriminant
useMLP = False  # Multi Layer Perceptron
useBDT = True  # Boosted Decision Tree

outputFile = TFile.Open("Higgs_ClassificationOutput.root", "RECREATE")
factory = TMVA.Factory(
    "TMVA_Higgs_Classification", outputFile, V=False, ROC=True, Silent=False, Color=True, AnalysisType="Classification"
)


## Setup Dataset(s)

# Define now input data file and signal and background trees

inputFileName = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"

inputFile = TFile.Open(inputFileName)
if inputFile is None:
   raise FileNotFoundError("Input file is not found - exit")


# --- Register the training and test trees
signalTree = inputFile.Get("sig_tree")
backgroundTree = inputFile.Get("bkg_tree")
signalTree.Print()

## Declare DataLoader(s)

# The next step is to declare the DataLoader class that deals with input variables
# Define the input variables that shall be used for the MVA training
# note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]
loader = TMVA.DataLoader("dataset")

loader.AddVariable("m_jj")
loader.AddVariable("m_jjj")
loader.AddVariable("m_lv")
loader.AddVariable("m_jlv")
loader.AddVariable("m_bb")
loader.AddVariable("m_wbb")
loader.AddVariable("m_wwbb")

# We set now the input data trees in the TMVA DataLoader class
# global event weights per tree (see below for setting event-wise weights)
signalWeight = 1.0
backgroundWeight = 1.0
# You can add an arbitrary number of signal or background trees
loader.AddSignalTree(signalTree, signalWeight)
loader.AddBackgroundTree(backgroundTree, backgroundWeight)

# Set individual event weights (the variables must exist in the original TTree)
#   for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
#   for background: factory->SetBackgroundWeightExpression("weight1*weight2");
#   loader->SetBackgroundWeightExpression( "weight" );


# Apply additional cuts on the signal and background samples (can be different)
mycuts = ROOT.TCut("")  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ROOT.TCut("")  # for example: TCut mycutb = "abs(var1)<0.5";

# Tell the factory how to use the training and testing events
#
# If no numbers of events are given, half of the events in the tree are used
# for training, and the other half for testing:
#    loader->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
# To also specify the number of testing events, use:

loader.PrepareTrainingAndTestTree(
    mycuts, mycutb, nTrain_Signal=7000, nTrain_Background=7000, SplitMode="Random", NormMode="NumEvents", V=False
)

## Booking Methods

# Here we book the TMVA methods. We book a Likelihood based on KDE (Kernel Density Estimation), a Fischer discriminant, a BDT
# and a shallow neural network
# Likelihood ("naive Bayes estimator")
if useLikelihood:
    factory.BookMethod(
        loader,
        TMVA.Types.kLikelihood,
        "Likelihood",
        H=True,
        V=False,
        TransformOutput=True,
        PDFInterpol="Spline2:NSmoothSig[0]=20:NSmoothBkg[0]=20:NSmoothBkg[1]=10",
        NSmooth=1,
        NAvEvtPerBin=50,
    )

# Use a kernel density estimator to approximate the PDFs
if useLikelihoodKDE:
    factory.BookMethod(
        loader,
        TMVA.Types.kLikelihood,
        "LikelihoodKDE",
        H=False,
        V=False,
        TransformOutput=False,
        PDFInterpol="KDE",
        KDEtype="Gauss",
        KDEiter="Adaptive",
        KDEFineFactor=0.3,
        KDEborder=None,
        NAvEvtPerBin=50,
    )

# Fisher discriminant (same as LD)
if useFischer:
    factory.BookMethod(
        loader,
        TMVA.Types.kFisher,
        "Fisher",
        H=True,
        V=False,
        Fisher=True,
        VarTransform=None,
        CreateMVAPdfs=True,
        PDFInterpolMVAPdf="Spline2",
        NbinsMVAPdf=50,
        NsmoothMVAPdf=10,
    )

# Boosted Decision Trees
if useBDT:
    factory.BookMethod(
        loader,
        TMVA.Types.kBDT,
        "BDT",
        V=False,
        NTrees=200,
        MinNodeSize="2.5%",
        MaxDepth=2,
        BoostType="AdaBoost",
        AdaBoostBeta=0.5,
        UseBaggedBoost=True,
        BaggedSampleFraction=0.5,
        SeparationType="GiniIndex",
        nCuts=20,
    )

# Multi-Layer Perceptron (Neural Network)
if useMLP:
    factory.BookMethod(
        loader,
        TMVA.Types.kMLP,
        "MLP",
        H=False,
        V=False,
        NeuronType="tanh",
        VarTransform="N",
        NCycles=100,
        HiddenLayers="N+5",
        TestRate=5,
        UseRegulator=False,
    )

## Train Methods

# Here we train all the previously booked methods.

factory.TrainAllMethods()
## Test  all methods

# Now we test and evaluate all methods using the test data set
factory.TestAllMethods()

factory.EvaluateAllMethods()

# after we get the ROC curve and we display

c1 = factory.GetROCCurve(loader)
c1.Draw()
# at the end we close the output file which contains the evaluation result of all methods and it can be used by TMVAGUI
# to display additional plots

outputFile.Close()
