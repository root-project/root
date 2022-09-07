## \file
## \ingroup tutorial_tmva
## \notebook
##  TMVA Classification Example Using a Convolutional Neural Network
##
## This is an example of using a CNN in TMVA. We do classification using a toy image data set
## that is generated when running the example macro
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Harshal Shende


# TMVA Classification Example Using a Convolutional Neural Network


## Helper function to create input images data
## we create a signal and background 2D histograms from 2d gaussians
## with a location (means in X and Y)  different for each event
## The difference between signal and background is in the gaussian width.
## The width for the background gaussian is slightly larger than the signal width by few % values


import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile


import os

os.environ["KERAS_BACKEND"] = "tensorflow"


TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()


def MakeImagesTree(n, nh, nw):
    # image size (nh x nw)
    ntot = nh * nw
    fileOutName = "images_data_16x16.root"
    nRndmEvts = 10000  # number of events we use to fill each image
    delta_sigma = 0.1  # 5% difference in the sigma
    pixelNoise = 5

    sX1 = 3
    sY1 = 3
    sX2 = sX1 + delta_sigma
    sY2 = sY1 - delta_sigma
    h1 = ROOT.TH2D("h1", "h1", nh, 0, 10, nw, 0, 10)
    h2 = ROOT.TH2D("h2", "h2", nh, 0, 10, nw, 0, 10)
    f1 = ROOT.TF2("f1", "xygaus")
    f2 = ROOT.TF2("f2", "xygaus")
    sgn = ROOT.TTree("sig_tree", "signal_tree")
    bkg = ROOT.TTree("bkg_tree", "background_tree")

    f = TFile(fileOutName, "RECREATE")
    x1 = ROOT.std.vector["float"](ntot)
    x2 = ROOT.std.vector["float"](ntot)

    # create signal and background trees with a single branch
    # an std::vector<float> of size nh x nw containing the image data
    bkg.Branch("vars", "std::vector<float>", x1)
    sgn.Branch("vars", "std::vector<float>", x2)

    sgn.SetDirectory(f)
    bkg.SetDirectory(f)

    f1.SetParameters(1, 5, sX1, 5, sY1)
    f2.SetParameters(1, 5, sX2, 5, sY2)
    ROOT.gRandom.SetSeed(0)
    ROOT.Info("TMVA_CNN_Classification", "Filling ROOT tree \n")
    for i in range(n):
        if i % 1000 == 0:
            print("Generating image event ...", i)

        h1.Reset()
        h2.Reset()
        # generate random means in range [3,7] to be not too much on the border
        f1.SetParameter(1, ROOT.gRandom.Uniform(3, 7))
        f1.SetParameter(3, ROOT.gRandom.Uniform(3, 7))
        f2.SetParameter(1, ROOT.gRandom.Uniform(3, 7))
        f2.SetParameter(3, ROOT.gRandom.Uniform(3, 7))

        h1.FillRandom("f1", nRndmEvts)
        h2.FillRandom("f2", nRndmEvts)

        for k in range(nh):
            for l in range(nw):
                m = k * nw + l
                # add some noise in each bin
                x1[m] = h1.GetBinContent(k + 1, l + 1) + ROOT.gRandom.Gaus(0, pixelNoise)
                x2[m] = h2.GetBinContent(k + 1, l + 1) + ROOT.gRandom.Gaus(0, pixelNoise)

        sgn.Fill()
        bkg.Fill()

    sgn.Write()
    bkg.Write()

    print("Signal and background tree with images data written to the file %s", f.GetName())
    sgn.Print()
    bkg.Print()
    f.Close()


opt = [1, 1, 1, 1, 1]
useTMVACNN = opt[0] if len(opt) > 0 or ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes" else False
useKerasCNN = opt[1] if len(opt) > 1 else False
useTMVADNN = opt[2] if len(opt) > 2 else False
useTMVABDT = opt[3] if len(opt) > 3 else False
usePyTorchCNN = opt[4] if len(opt) > 4 else False


if not useTMVACNN:
    ROOT.Warning(
        "TMVA_CNN_Classificaton",
        "TMVA is not build with GPU or CPU multi-thread support. Cannot use TMVA Deep Learning for CNN",
    )

writeOutputFile = True

num_threads = 0  # use default threads


# do enable MT running
if num_threads >= 0:
    ROOT.EnableImplicitMT(num_threads)
    if not num_threads:
        ROOT.gSystem.Setenv("OMP_NUM_THREADS", ROOT.TString.Format("%d", num_threads))
else:
    ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")

print("Running with nthreads  = ", ROOT.GetThreadPoolSize())


if ROOT.gSystem.GetFromPipe("root-config --has-tmva-pymva") == "yes":
    useKerasCNN = True
    usePyTorchCNN = True


outputFile = None
if writeOutputFile:
    outputFile = TFile.Open("TMVA_CNN_ClassificationOutput.root", "RECREATE")


## Create TMVA Factory

# Create the Factory class. Later you can choose the methods
# whose performance you'd like to investigate.

# The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass

# - The first argument is the base of the name of all the output
# weightfiles in the directory weight/ that will be created with the
#    method parameters

# - The second argument is the output file for the training results

# - The third argument is a string option defining some general configuration for the TMVA session.
# For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the
# option string

# - note that we disable any pre-transformation of the input variables and we avoid computing correlations between
# input variables


factory = TMVA.Factory(
    "TMVA_CNN_Classification",
    outputFile,
    V=False,
    ROC=True,
    Silent=False,
    Color=True,
    AnalysisType="Classification",
    Transformations=None,
    Correlations=False,
)


## Declare DataLoader(s)

# The next step is to declare the DataLoader class that deals with input variables

# Define the input variables that shall be used for the MVA training
# note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]

# In this case the input data consists of an image of 16x16 pixels. Each single pixel is a branch in a ROOT TTree

loader = TMVA.DataLoader("dataset")


## Setup Dataset(s)

# Define input data file and signal and background trees


imgSize = 16 * 16
inputFileName = "images_data_16x16.root"

# if file does not exists create it
if ROOT.gSystem.AccessPathName(inputFileName):
    MakeImagesTree(5000, 16, 16)

inputFile = TFile.Open(inputFileName)
if inputFile is None:
    ROOT.Warning("TMVA_CNN_Classification", "Error opening input file %s - exit", inputFileName.Data())


# inputFileName = "tmva_class_example.root"


# --- Register the training and test trees

signalTree = inputFile.Get("sig_tree")
backgroundTree = inputFile.Get("bkg_tree")

nEventsSig = signalTree.GetEntries()
nEventsBkg = backgroundTree.GetEntries()

# global event weights per tree (see below for setting event-wise weights)
signalWeight = 1.0
backgroundWeight = 1.0

# You can add an arbitrary number of signal or background trees
loader.AddSignalTree(signalTree, signalWeight)
loader.AddBackgroundTree(backgroundTree, backgroundWeight)

## add event variables (image)
## use new method (from ROOT 6.20 to add a variable array for all image data)
loader.AddVariablesArray("vars", imgSize)

# Set individual event weights (the variables must exist in the original TTree)
#    for signal    : factory->SetSignalWeightExpression    ("weight1*weight2");
#    for background: factory->SetBackgroundWeightExpression("weight1*weight2");
# loader->SetBackgroundWeightExpression( "weight" );

# Apply additional cuts on the signal and background samples (can be different)
mycuts = ""  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ""  # for example: TCut mycutb = "abs(var1)<0.5";

# Tell the factory how to use the training and testing events
# If no numbers of events are given, half of the events in the tree are used
# for training, and the other half for testing:
#    loader.PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
# It is possible also to specify the number of training and testing events,
# note we disable the computation of the correlation matrix of the input variables

nTrainSig = 0.8 * nEventsSig
nTrainBkg = 0.8 * nEventsBkg

# build the string options for DataLoader::PrepareTrainingAndTestTree

loader.PrepareTrainingAndTestTree(
    mycuts,
    mycutb,
    nTrain_Signal=nTrainSig,
    nTrain_Background=nTrainBkg,
    SplitMode="Random",
    SplitSeed=100,
    NormMode="NumEvents",
    V=False,
    CalcCorrelations=False,
)


# DataSetInfo              : [dataset] : Added class "Signal"
#    : Add Tree sig_tree of type Signal with 10000 events
#    DataSetInfo              : [dataset] : Added class "Background"
#        : Add Tree bkg_tree of type Background with 10000 events

# signalTree.Print();

# Booking Methods

# Here we book the TMVA methods. We book a Boosted Decision Tree method (BDT)


# Boosted Decision Trees
if useTMVABDT:
    factory.BookMethod(
        loader,
        TMVA.Types.kBDT,
        "BDT",
        V=False,
        NTrees=400,
        MinNodeSize="2.5%",
        MaxDepth=2,
        BoostType="AdaBoost",
        AdaBoostBeta=0.5,
        UseBaggedBoost=True,
        BaggedSampleFraction=0.5,
        SeparationType="GiniIndex",
        nCuts=20,
    )


#### Booking Deep Neural Network

# Here we book the DNN of TMVA. See the example TMVA_Higgs_Classification.C for a detailed description of the
# options

if useTMVADNN:
    layoutString = ROOT.TString(
        "DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,DENSE|1|LINEAR"
    )

# Training strategies
# one can catenate several training strings with different parameters (e.g. learning rates or regularizations
# parameters) The training string must be concatenates with the `|` delimiter
trainingString1 = ROOT.TString(
    "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
    "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"
    "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
    "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0."
)  # + "|" + trainingString2 + ....

# Build now the full DNN Option string
dnnMethodName = "TMVA_DNN_CPU"

# use GPU if available
if ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes":
    dnnOptions = "GPU"
    dnnMethodName = "TMVA_DNN_GPU"
elif ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes":
    dnnOptions = "CPU"


factory.BookMethod(
    loader,
    TMVA.Types.kDL,
    dnnMethodName,
    H=False,
    V=True,
    ErrorStrategy="CROSSENTROPY",
    VarTransform=None,
    WeightInitialization="XAVIER",
    Layout=layoutString,
    TrainingStrategy=trainingString1,
    Architecture=dnnOptions,
)


### Book Convolutional Neural Network in TMVA

# For building a CNN one needs to define

# -  Input Layout :  number of channels (in this case = 1)  | image height | image width
# -  Batch Layout :  batch size | number of channels | image size = (height*width)

# Then one add Convolutional layers and MaxPool layers.

# -  For Convolutional layer the option string has to be:
# - CONV | number of units | filter height | filter width | stride height | stride width | padding height | paddig
# width | activation function

# - note in this case we are using a filer 3x3 and padding=1 and stride=1 so we get the output dimension of the
# conv layer equal to the input

# - note we use after the first convolutional layer a batch normalization layer. This seems to help significantly the
# convergence

# - For the MaxPool layer:
# - MAXPOOL  | pool height | pool width | stride height | stride width

# The RESHAPE layer is needed to flatten the output before the Dense layer

# Note that to run the CNN is required to have CPU  or GPU support


if useTMVACNN:
    # Training strategies.
    trainingString1 = ROOT.TString(
        "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
        "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"
        "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
        "Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0"
    )

    ## New DL (CNN)
    cnnMethodName = "TMVA_CNN_CPU"
    cnnOptions = None
    # use GPU if available
    if ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes":
        cnnOptions = "GPU"
        cnnMethodName = "TMVA_CNN_GPU"
    elif ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes":
        cnnOptions = "CPU"

    cnnOptions += ROOT.cnnOptions
    cnnMethodName = ROOT.cnnMethodName

    factory.BookMethod(
        loader,
        TMVA.Types.kDL,
        cnnMethodName,
        H=False,
        V=True,
        ErrorStrategy="CROSSENTROPY",
        VarTransform=None,
        WeightInitialization="XAVIER",
        InputLayout="1|16|16",
        Layout="CONV|10|3|3|1|1|1|1|RELU,BNORM,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|100|RELU,DENSE|1|LINEAR",
        TrainingStrategy=trainingString1,
        Architecture=cnnOptions,
    )


### Book Convolutional Neural Network in Keras using a generated model


if useKerasCNN:
    ROOT.Info("TMVA_CNN_Classification", "Building convolutional keras model")
    # create python script which can be executed
    # create 2 conv2d layer + maxpool + dense
    import tensorflow
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    # from keras.initializers import TruncatedNormal
    # from keras import initializations
    from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape

    # from keras.callbacks import ReduceLROnPlateau
    model = Sequential()
    model.add(Reshape((16, 16, 1), input_shape=(256,)))
    model.add(Conv2D(10, kernel_size=(3, 3), kernel_initializer="TruncatedNormal", activation="relu", padding="same"))
    model.add(Conv2D(10, kernel_size=(3, 3), kernel_initializer="TruncatedNormal", activation="relu", padding="same"))
    # stride for maxpool is equal to pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="tanh"))
    # model.add(Dropout(0.2))
    model.add(Dense(2, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    model.save("model_cnn.h5")
    model.summary()

    if ROOT.gSystem.AccessPathName("model_cnn.h5"):
        raise FileNotFoundError("Error creating Keras model file - skip using Keras")
    else:
        # book PyKeras method only if Keras model could be created
        ROOT.Info("TMVA_CNN_Classification", "Building convolutional keras model")
        factory.BookMethod(
            loader,
            TMVA.Types.kPyKeras,
            "PyKeras",
            H=True,
            V=False,
            VarTransform=None,
            FilenameModel="model_cnn.h5",
            FilenameTrainedModel="trained_model_cnn.h5",
            NumEpochs=20,
            BatchSize=100,
            GpuOptions="allow_growth=True",
        )  # needed for RTX NVidia card and to avoid TF allocates all GPU memory


if usePyTorchCNN:
    ROOT.Info("TMVA_CNN_Classification", "Using Convolutional PyTorch Model")
    pyTorchFileName = ROOT.gROOT.GetTutorialDir() + "/tmva/PyTorch_Generate_CNN_Model.py"
    # check that pytorch can be imported and file defining the model and used later when booking the method is existing
    if ROOT.gSystem.Exec(str(TMVA.Python_Executable()) + "-c 'import torch'") or ROOT.gSystem.AccessPathName(
        pyTorchFileName
    ):
        ROOT.Warning(
            "TMVA_CNN_Classification",
            "PyTorch is not installed or model building file is not existing - skip using PyTorch",
        )

    else:
        ROOT.Info("TMVA_CNN_Classification", "Booking PyTorch CNN model")
        factory.BookMethod(
            loader,
            TMVA.Types.kPyTorch,
            "PyTorch",
            H=True,
            V=False,
            VarTransform=None,
            FilenameModel="PyTorchModelCNN.pt",
            FilenameTrainedModel="PyTorchTrainedModelCNN.pt",
            NumEpochs=20,
            BatchSize=100,
            UserCode=pyTorchFileName,
        )


## Train Methods

factory.TrainAllMethods()

## Test and Evaluate Methods

factory.TestAllMethods()

factory.EvaluateAllMethods()

## Plot ROC Curve

c1 = factory.GetROCCurve(loader)
c1.Draw()

# close outputfile to save output file
outputFile.Close()
