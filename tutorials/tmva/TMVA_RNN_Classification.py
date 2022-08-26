## \file
## \ingroup tutorial_tmva
## \notebook
##  TMVA Classification Example Using a Recurrent Neural Network
##
## This is an example of using a RNN in TMVA. We do classification using a toy time dependent data set
## that is generated when running this example macro
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Harshal Shende


# TMVA Classification Example Using a Recurrent Neural Network

# This is an example of using a RNN in TMVA.
# We do the classification using a toy data set containing a time series of data sample ntimes
# and with dimension ndim that is generated when running the provided function `MakeTimeData (nevents, ntime, ndim)`


import ROOT

TMVA = ROOT.TMVA
TFile = ROOT.TFile

import os

os.environ["KERAS_BACKEND"] = "tensorflow"


TMVA.Tools.Instance()
TMVA.Config.Instance()


##  Helper function to generate the time data set
##  make some time data but not of fixed length.
##  use a poisson with mu = 5 and truncated at 10


def MakeTimeData(n, ntime, ndim):
    # ntime = 10;
    # ndim = 30; // number of dim/time

    fname = "time_data_t" + str(ntime) + "_d" + str(ndim) + ".root"
    v1 = []
    v2 = []

    for i in range(ntime):
        v1.append(ROOT.TH1D("h1_" + str(i), "h1", ndim, 0, 10))
        v2.append(ROOT.TH1D("h2_" + str(i), "h2", ndim, 0, 10))

    f1 = ROOT.TF1("f1", "gaus")
    f2 = ROOT.TF1("f2", "gaus")

    sgn = ROOT.TTree("sgn", "sgn")
    bkg = ROOT.TTree("bkg", "bkg")
    f = TFile(fname, "RECREATE")

    x1 = []
    x2 = []

    for i in range(ntime):
        x1.append(ROOT.std.vector["float"](ndim))
        x2.append(ROOT.std.vector["float"](ndim))

    for i in range(ntime):
        bkg.Branch("vars_time" + str(i), "std::vector<float>", x1[i])
        sgn.Branch("vars_time" + str(i), "std::vector<float>", x2[i])

    sgn.SetDirectory(f)
    bkg.SetDirectory(f)
    ROOT.gRandom.SetSeed(0)

    mean1 = ROOT.std.vector["double"](ntime)
    mean2 = ROOT.std.vector["double"](ntime)
    sigma1 = ROOT.std.vector["double"](ntime)
    sigma2 = ROOT.std.vector["double"](ntime)

    for j in range(ntime):
        mean1[j] = 5.0 + 0.2 * ROOT.TMath.Sin(ROOT.TMath.Pi() * j / float(ntime))
        mean2[j] = 5.0 + 0.2 * ROOT.TMath.Cos(ROOT.TMath.Pi() * j / float(ntime))
        sigma1[j] = 4 + 0.3 * ROOT.TMath.Sin(ROOT.TMath.Pi() * j / float(ntime))
        sigma2[j] = 4 + 0.3 * ROOT.TMath.Cos(ROOT.TMath.Pi() * j / float(ntime))

    for i in range(n):
        if i % 1000 == 0:
            print("Generating  event ... %d", i)

        for j in range(ntime):
            h1 = v1[j]
            h2 = v2[j]
            h1.Reset()
            h2.Reset()

            f1.SetParameters(1, mean1[j], sigma1[j])
            f2.SetParameters(1, mean2[j], sigma2[j])

            h1.FillRandom("f1", 1000)
            h2.FillRandom("f2", 1000)

            for k in range(ntime):
                # std::cout << j*10+k << "   ";
                x1[j][k] = h1.GetBinContent(k + 1) + ROOT.gRandom.Gaus(0, 10)
                x2[j][k] = h2.GetBinContent(k + 1) + ROOT.gRandom.Gaus(0, 10)

        sgn.Fill()
        bkg.Fill()

        if n == 1:
            c1 = ROOT.TCanvas()
            c1.Divide(ntime, 2)
            for j in range(ntime):
                c1.cd(j + 1)
                v1[j].Draw()
            for j in range(ntime):
                c1.cd(ntime + j + 1)
                v2[j].Draw()

            ROOT.gPad.Update()

    if n > 1:
        sgn.Write()
        bkg.Write()
        sgn.Print()
        bkg.Print()
        f.Close()


## macro for performing a classification using a Recurrent Neural Network
## @param use_type
##    use_type = 0    use Simple RNN network
##    use_type = 1    use LSTM network
##    use_type = 2    use GRU
##    use_type = 3    build 3 different networks with RNN, LSTM and GRU


use_type = 1
ninput = 30
ntime = 10
batchSize = 100
maxepochs = 20

nTotEvts = 10000  # total events to be generated for signal or background

useKeras = True

useTMVA_RNN = True
useTMVA_DNN = True
useTMVA_BDT = False

rnn_types = ["RNN", "LSTM", "GRU"]
use_rnn_type = [1, 1, 1]

if 0 <= use_type < 3:
    use_rnn_type = [0, 0, 0]
    use_rnn_type[use_type] = 1

useGPU = True  # use GPU for TMVA if available

useGPU = ROOT.gSystem.GetFromPipe("root-config --has-tmva-gpu") == "yes"
useTMVA_RNN = ROOT.gSystem.GetFromPipe("root-config --has-tmva-cpu") == "yes"

if useTMVA_RNN:
    ROOT.Warning(
        "TMVA_RNN_Classification",
        "TMVA is not build with GPU or CPU multi-thread support. Cannot use TMVA Deep Learning for RNN",
    )

archString = "GPU" if useGPU else "CPU"

writeOutputFile = True

rnn_type = "RNN"

if ROOT.gSystem.GetFromPipe("root-config --has-tmva-pymva") == "yes":
    TMVA.PyMethodBase.PyInitialize()
else:
    useKeras = False

num_threads = 0  # use by default all threads
# do enable MT running
if num_threads >= 0:
    ROOT.EnableImplicitMT(num_threads)
    if num_threads > 0:
        ROOT.gSystem.Setenv("OMP_NUM_THREADS", ROOT.TString.Format("%d", num_threads))
else:
    ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")


print("Running with nthreads  = {}".format(ROOT.GetThreadPoolSize()))

inputFileName = "time_data_t10_d30.root"

fileDoesNotExist = ROOT.gSystem.AccessPathName(inputFileName)

# if file does not exists create it
if fileDoesNotExist:
    MakeTimeData(nTotEvts, ntime, ninput)


inputFile = TFile.Open(inputFileName)
if inputFile is None:
    raise ROOT.Error("Error opening input file %s - exit", inputFileName.Data())


print("--- RNNClassification  : Using input file: {}".format(inputFile.GetName()))

# Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outfileName = "data_RNN_" + archString + ".root"
outputFile = None


if writeOutputFile:
    outputFile = TFile.Open(outfileName, "RECREATE")


## Declare Factory

# Create the Factory class. Later you can choose the methods
# whose performance you'd like to investigate.

# The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to
# pass

# - The first argument is the base of the name of all the output
# weightfiles in the directory weight/ that will be created with the
#     method parameters

# - The second argument is the output file for the training results
#
# - The third argument is a string option defining some general configuration for the TMVA session.
# For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in
# the option string


# // Creating the factory object
factory = TMVA.Factory(
    "TMVAClassification",
    outputFile,
    V=False,
    Silent=False,
    Color=True,
    DrawProgressBar=True,
    Transformations=None,
    Correlations=False,
    AnalysisType="Classification",
    ModelPersistence=True,
)
dataloader = TMVA.DataLoader("dataset")

signalTree = inputFile.Get("sgn")
background = inputFile.Get("bkg")

nvar = ninput * ntime

## add variables - use new AddVariablesArray function
for i in range(ntime):
    dataloader.AddVariablesArray("vars_time" + str(i), ninput)


dataloader.AddSignalTree(signalTree, 1.0)
dataloader.AddBackgroundTree(background, 1.0)

# check given input
datainfo = dataloader.GetDataSetInfo()
vars = datainfo.GetListOfVariables()
print("number of variables is {}".format(vars.size()))


for v in vars:
    print(v)

nTrainSig = 0.8 * nTotEvts
nTrainBkg = 0.8 * nTotEvts

# Apply additional cuts on the signal and background samples (can be different)
mycuts = ""  # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ""

# build the string options for DataLoader::PrepareTrainingAndTestTree
dataloader.PrepareTrainingAndTestTree(
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

print("prepared DATA LOADER ")


## Book TMVA  recurrent models

# Book the different types of recurrent models in TMVA  (SimpleRNN, LSTM or GRU)


if useTMVA_RNN:
    for i in range(3):
        if not use_rnn_type[i]:
            continue

        rnn_type = rnn_types[i]

        ## Define RNN layer layout
        ##  it should be   LayerType (RNN or LSTM or GRU) |  number of units | number of inputs | time steps | remember output (typically no=0 | return full sequence
        rnnLayout = str(rnn_type) + "|10|" + str(ninput) + "|" + str(ntime) + "|0|1,RESHAPE|FLAT,DENSE|64|TANH,LINEAR"

        ## Defining Training strategies. Different training strings can be concatenate. Use however only one
        trainingString1 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,ConvergenceSteps=5,BatchSize=" + str(batchSize)
        trainingString1 += ",TestRepetitions=1,WeightDecay=1e-2,Regularization=None,MaxEpochs=" + str(maxepochs)
        trainingString1 += "Optimizer=ADAM,DropConfig=0.0+0.+0.+0."

        ## define the inputlayout string for RNN
        ## the input data should be organize as   following:
        ##/ input layout for RNN:    time x ndim
        ## add after RNN a reshape layer (needed top flatten the output) and a dense layer with 64 units and a last one
        ## Note the last layer is linear because  when using Crossentropy a Sigmoid is applied already
        ## Define the full RNN Noption string adding the final options for all network
        rnnName = "TMVA_" + str(rnn_type)
        factory.BookMethod(
            dataloader,
            TMVA.Types.kDL,
            rnnName,
            H=False,
            V=True,
            ErrorStrategy="CROSSENTROPY",
            VarTransform=None,
            WeightInitialization="XAVIERUNIFORM",
            ValidationSize=0.2,
            RandomSeed=1234,
            InputLayout=str(ntime) + "|" + str(ninput),
            Layout=rnnLayout,
            TrainingStrategy=trainingString1,
            Architecture=archString,
        )


## Book TMVA  fully connected dense layer  models
if useTMVA_DNN:
    # Method DL with Dense Layer
    # Training strategies.
    trainingString1 = ROOT.TString(
        "LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
        "ConvergenceSteps=10,BatchSize=256,TestRepetitions=1,"
        "WeightDecay=1e-4,Regularization=None,MaxEpochs=20"
        "DropConfig=0.0+0.+0.+0.,Optimizer=ADAM:"
    )  # + "|" + trainingString2
    # General Options.
    trainingString1.Append(archString)
    dnnName = "TMVA_DNN"
    factory.BookMethod(
        dataloader,
        TMVA.Types.kDL,
        dnnName,
        H=False,
        V=True,
        ErrorStrategy="CROSSENTROPY",
        VarTransform=None,
        WeightInitialization="XAVIER",
        RandomSeed=0,
        InputLayout="1|1|" + str(ntime * ninput),
        Layout="DENSE|64|TANH,DENSE|TANH|64,DENSE|TANH|64,LINEAR",
        TrainingStrategy=trainingString1,
    )


## Book Keras recurrent models

# Book the different types of recurrent models in Keras  (SimpleRNN, LSTM or GRU)


if useKeras:
    for i in range(3):
        if use_rnn_type[i]:
            modelName = "model_" + rnn_types[i] + ".h5"
            trainedModelName = "trained_" + modelName
            print("Building recurrent keras model using a", rnn_types[i], "layer")
            # create python script which can be executed
            # create 2 conv2d layer + maxpool + dense
            from keras.models import Sequential
            from keras.optimizers import Adam

            # from keras.initializers import TruncatedNormal
            # from keras import initializations
            from keras.layers import Input, Dense, Dropout, Flatten, SimpleRNN, GRU, LSTM, Reshape, BatchNormalization

            model = Sequential()
            model.add(Reshape((10, 30), input_shape=(10 * 30,)))
            # add recurrent neural network depending on type / Use option to return the full output
            if rnn_types[i] == "LSTM":
                model.add(LSTM(units=10, return_sequences=True))
            elif rnn_types[i] == "GRU":
                model.add(GRU(units=10, return_sequences=True))
            else:
                model.add(SimpleRNN(units=10, return_sequences=True))
                # m.AddLine("model.add(BatchNormalization())");
                model.add(Flatten())  # needed if returning the full time output sequence
                model.add(Dense(64, activation="tanh"))
                model.add(Dense(2, activation="sigmoid"))
                model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
                model.save(modelName)
                model.summary()

            if ROOT.gSystem.AccessPathName(modelName):
                useKeras = False
                raise FileNotFoundError("Error creating Keras recurrent model file - Skip using Keras")
            else:
                # book PyKeras method only if Keras model could be created
                print("Booking Keras  model ", rnn_types[i])
                factory.BookMethod(
                    dataloader,
                    TMVA.Types.kPyKeras,
                    "PyKeras_" + rnn_types[i],
                    H=True,
                    V=False,
                    VarTransform=None,
                    FilenameModel=modelName,
                    FilenameTrainedModel="trained_" + modelName,
                    NumEpochs=maxepochs,
                    BatchSize=batchSize,
                    GpuOptions="allow_growth=True",
                )


# use BDT in case not using Keras or TMVA DL
if not useKeras or not useTMVA_BDT:
    useTMVA_BDT = True


## Book TMVA BDT


if useTMVA_BDT:
    factory.BookMethod(
        dataloader,
        TMVA.Types.kBDT,
        "BDTG",
        H=True,
        V=False,
        NTrees=100,
        MinNodeSize="2.5%",
        BoostType="Grad",
        Shrinkage=0.10,
        UseBaggedBoost=True,
        BaggedSampleFraction=0.5,
        nCuts=20,
        MaxDepth=2,
    )


## Train all methods
factory.TrainAllMethods()

print("nthreads  = {}".format(ROOT.GetThreadPoolSize()))

# ---- Evaluate all MVAs using the set of test events
factory.TestAllMethods()

# ----- Evaluate and compare performance of all configured MVAs
factory.EvaluateAllMethods()

# check method

# plot ROC curve
c1 = factory.GetROCCurve(dataloader)
c1.Draw()

if outputFile:
    outputFile.Close()
