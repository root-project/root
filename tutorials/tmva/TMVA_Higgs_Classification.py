## \file
## \ingroup tutorial_tmva
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
## For PYMVA methods
TMVA.PyMethodBase.PyInitialize()


# options to control used methods
useLikelihood = True  # likelihood based discriminant
useLikelihoodKDE = False  # likelihood based discriminant
useFischer = True  # Fischer discriminant
useMLP = False  # Multi Layer Perceptron (old TMVA NN implementation)
useBDT = True  # Boosted Decision Tree
useDL = True  # TMVA Deep learning ( CPU or GPU)

TMVA.Tools.Instance()
outputFile = TFile.Open("Higgs_ClassificationOutput.root", "RECREATE")
factory = TMVA.Factory(
    "TMVA_Higgs_Classification", outputFile, V=False, ROC=True, Silent=False, Color=True, AnalysisType="Classification"
)


## Setup Dataset(s)

# Define now input data file and signal and background trees

inputFileName = "Higgs_data.root"
inputFileLink = "http://root.cern.ch/files/" + inputFileName


if ROOT.gSystem.AccessPathName(inputFileName):
    # file exists
    ROOT.Info("TMVA_Higgs_Classification", "Download Higgs_data.root file")
    TFile.SetCacheFileDir(".")
    inputFile = TFile.Open(inputFileLink, "CACHEREAD")
    if inputFile is None:
        raise FileNotFoundError("Input file cannot be downloaded - exit")
else:
    inputFile = TFile.Open(inputFileName)


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

# Here we book the TMVA methods. We book first a Likelihood based on KDE (Kernel Density Estimation), a Fischer discriminant, a BDT
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

## Here we book the new DNN of TMVA if we have support in ROOT. We will use GPU version if ROOT is enabled with GPU


## Booking Deep Neural Network

# Here we define the option string for building the Deep Neural network model.

#### 1. Define DNN layout

# The DNN configuration is defined using a string. Note that whitespaces between characters are not allowed.

# We define first the DNN layout:

# - **input layout** :   this defines the input data format for the DNN as  ``input depth | height | width``.
# In case of a dense layer as first layer the input layout should be  ``1 | 1 | number of input variables`` (features)
# - **batch layout**  : this defines how are the input batch. It is related to input layout but not the same.
# If the first layer is dense it should be ``1 | batch size ! number of variables`` (features)

# *(note the use of the character `|` as  separator of  input parameters for DNN layout)*

# note that in case of only dense layer the input layout could be omitted but it is required when defining more
# complex architectures

# - **layer layout** string defining the layer architecture. The syntax is
# - layer type (e.g. DENSE, CONV, RNN)
# - layer parameters (e.g. number of units)
# - activation function (e.g  TANH, RELU,...)

# *the different layers are separated by the ``","`` *

#### 2. Define Training Strategy

# We define here the training strategy parameters for the DNN. The parameters are separated by the ``","`` separator.
# One can then concatenate different training strategy with different parameters. The training strategy are separated by
# the ``"|"`` separator.

# - Optimizer
# - Learning rate
# - Momentum (valid for SGD and RMSPROP)
# - Regularization and Weight Decay
# - Dropout
# - Max number of epochs
# - Convergence steps. if the test error will not decrease after that value the training will stop
# - Batch size (This value must be the same specified in the input layout)
# - Test Repetitions (the interval when the test error will be computed)


#### 3. Define general DNN options

# We define the general DNN options concatenating in the final string the previously defined layout and training strategy.
# Note we use the ``":"`` separator to separate the different higher level options, as in the other TMVA methods.
# In addition to input layout, batch layout and training strategy we add now:

# - Type of Loss function (e.g. CROSSENTROPY)
# - Weight Initizalization (e.g XAVIER, XAVIERUNIFORM, NORMAL )
# - Variable Transformation
# - Type of Architecture (e.g. CPU, GPU, Standard)

# We can then book the DL method using the built option string
if useDL:
    useDLGPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes"

    # Define DNN layout
    # Define Training strategies
    # one can catenate several training strategies
    training1 = ROOT.TString(
        "LearningRate=1e-3,Momentum=0.9,"
        "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
        "MaxEpochs=30,WeightDecay=1e-4,Regularization=None,"
        "Optimizer=ADAM,ADAM_beta1=0.9,ADAM_beta2=0.999,ADAM_eps=1.E-7,"  # ADAM default parameters
        "DropConfig=0.0+0.0+0.0+0."
    )
    #   training2 = ROOT.TString("LearningRate=1e-3,Momentum=0.9"
    #                      "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
    #                       "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
    #                       "Optimizer=SGD,DropConfig=0.0+0.0+0.0+0.")

    # General Options.
    dnnMethodName = ROOT.TString("DNN_CPU")

    if useDLGPU:
        arch = "GPU"
        dnnMethodName = "DNN_GPU"
    else:
        arch = "CPU"

    factory.BookMethod(
        loader,
        TMVA.Types.kDL,
        dnnMethodName,
        H=False,
        V=True,
        ErrorStrategy="CROSSENTROPY",
        VarTransform="G",
        WeightInitialization="XAVIER",
        InputLayout="1|1|7",
        BatchLayout="1|128|7",
        Layout="DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|1|LINEAR",
        TrainingStrategy=training1,
        Architecture=arch,
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
