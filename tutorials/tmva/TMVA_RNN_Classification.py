
# TMVA Classification Example Using a Recurrent Neural Network
#  
# This is an example of using a RNN in TMVA. We do classification using a toy time dependent data set
# that is generated when running this example macro.
# 
# This is an example of using a RNN in TMVA. We do the classification using a toy data set containing a time series of data sample ntimes and with dimension ndim.

# First you need to run TMVA_RNN_Classification.C to generate time_data_t10_d30.root.

# Import the necessary modules

# We start with importing the necessary modules required for the tutorial. Here we imported ROOT and TMVA(Toolkit for Multivariate Data Analysis). If you want to know more about TMVA, you can refer the documentation.

import ROOT
from ROOT import TMVA

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization

ninput = 30

ntime = 10

batchSize = 100
maxepochs = 20

use_type = 1

nTotEvts = 10000 # total events to be generated for signal or background

useKeras = True


useTMVA_RNN = True
useTMVA_DNN = True
useTMVA_BDT = False

rnn_types = ["RNN", "LSTM", "GRU"]
use_rnn_type = [1, 1, 1]
if (use_type >=0 & use_type < 3):
      use_rnn_type = [0,0,0]
      use_rnn_type[use_type] = 1

archString = "CPU"
writeOutputFile = True

rnn_type = "RNN"


# ### Setting up TMVA
# 
# TMVA requires initialization the PyMVA to utilize PyTorch. PyMVA is the interface for third-party MVA tools based on Python. It is created to make powerful external libraries easily accessible with a direct integration into the TMVA workflow. All PyMVA methods provide the same plug-and-play mechanisms as the TMVA methods. Because the base method of PyMVA is inherited from the TMVA base method, all options of internal TMVA methods apply for PyMVA methods as well.

ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()

# Define the input files and the number of threads

num_threads = 0   # use by default all threads
#    do enable MT running
if (num_threads >= 0):
    ROOT.EnableImplicitMT(num_threads)
    if (num_threads > 0):
        ROOT.gSystem.Setenv("OMP_NUM_THREADS", num_threads)
    else:
      ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")


print("Running with nthreads  = " + str(ROOT.GetThreadPoolSize()) + "\n" )

inputFileName = "time_data_t10_d30.root"

fileExist = ROOT.gSystem.AccessPathName(inputFileName)

inputFile = ROOT.TFile.Open(inputFileName)
if (inputFile==None):
    Error("TMVA_RNN_Classification", "Error opening input file %s - exit", inputFileName.Data())


# ### Create an Output File and Declare Factory
# 
# Create the Factory class. Later you can choose the methods whose performance you'd like to investigate.
# 
# The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass
# 
# - The first argument is the base of the name of all the output weightfiles in the directory weight/ that will be created with the method parameters
# 
# - The second argument is the output file for the training results
# 
# - The third argument is a string option defining some general configuration for the TMVA session. 
# 
# For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string
# 

print("--- RNNClassification  : Using input file: " + inputFile.GetName()+"\n")

#   Create a ROOT output file where TMVA will store ntuples, histograms, etc.
outfileName = "data_RNN_"+ archString +".root"

if (writeOutputFile):
    outputFile = ROOT.TFile.Open(outfileName, "RECREATE")

#  Creating the factory object
factory = ROOT.TMVA.Factory("TMVAClassification", outputFile,"!V:!Silent:Color:DrawProgressBar:Transformations=None:!Correlations:"+"AnalysisType=Classification:ModelPersistence")


# ### Declare DataLoader(s)
# 
# The next step is to declare the DataLoader class that deals with input variables
# 
# Define the input variables that shall be used for the MVA training
# note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]


dataloader =TMVA.DataLoader("dataset")

signalTree = inputFile.Get("sgn")
background = inputFile.Get("bkg")

signalTree.Print()
nvar = ninput * ntime

# add variables - use new AddVariablesArray function
for i in range(ntime):
    varName = "vars_time"+str(i)
    dataloader.AddVariablesArray(varName,ninput,'F')

dataloader.AddSignalTree(signalTree, 1.0)
dataloader.AddBackgroundTree(background, 1.0)

# check given input
datainfo = dataloader.GetDataSetInfo()
vars = datainfo.GetListOfVariables()
print("number of variables is " + str(vars.size())+ "\n")
for v in vars:
    print(str(v)+"\n")

nTrainSig = 0.8 * nTotEvts
nTrainBkg = 0.8 *  nTotEvts

#build the string options for DataLoader::PrepareTrainingAndTestTree
prepareOptions = "nTrain_Signal="+str(nTrainSig)+":nTrain_Background="+str(nTrainBkg)+":SplitMode=Random:SplitSeed=100:NormMode=NumEvents:!V:!CalcCorrelations"


# ###  Tell the factory how to use the training and testing events

# Apply additional cuts on the signal and background samples (can be different)
mycuts = ROOT.TCut("")   ## for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ROOT.TCut("")   ## for example: TCut mycutb = "abs(var1)<0.5";

dataloader.PrepareTrainingAndTestTree(mycuts, mycutb, prepareOptions)

print("prepared DATA LOADER " )


# ### Book TMVA  recurrent models
# 
# Book the different types of recurrent models in TMVA  (SimpleRNN, LSTM or GRU)

if (useTMVA_RNN):
    for i in range(3):
        if (use_rnn_type[i]==None):
            continue
        rnn_type = str(rnn_types[i])

#          define the inputlayout string for RNN
#          the input data should be organize as   following:
#          input layout for RNN:    time x ndim

        inputLayoutString = "InputLayout="+str(ntime)+"|"+str(ninput)

        # Define RNN layer layout
        # it should be   LayerType (RNN or LSTM or GRU) |  number of units | number of inputs | time steps | remember output (typically no=0 | return full sequence
        rnnLayout = str(rnn_type) + "|10|"+ str(ninput) + "|" + str(ntime) + "|0|1"

        #        add after RNN a reshape layer (needed top flatten the output) and a dense layer with 64 units and a last one
        #        Note the last layer is linear because  when using Crossentropy a Sigmoid is applied already
        layoutString ="Layout=" + rnnLayout + ",RESHAPE|FLAT,DENSE|64|TANH,LINEAR"

        #Defining Training strategies. Different training strings can be concatenate. Use however only one
        trainingString1 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,"+"ConvergenceSteps=5,BatchSize="+str(batchSize)+",TestRepetitions=1,"+"WeightDecay=1e-2,Regularization=None,MaxEpochs="+str(maxepochs
        )+","+"Optimizer=ADAM,DropConfig=0.0+0.+0.+0."

        trainingStrategyString="TrainingStrategy="
        trainingStrategyString += trainingString1; # + "|" + trainingString2

        # Define the full RNN Noption string adding the final options for all network
        rnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"+"WeightInitialization=XAVIERUNIFORM:ValidationSize=0.2:RandomSeed=1234"
        rnnOptions +=  ":" + inputLayoutString
        rnnOptions +=  ":" + layoutString
        rnnOptions +=  ":" + trainingStrategyString
        rnnOptions +=  ":" + "Architecture=" + str(archString)

        rnnName = "TMVA_" + rnn_type
        factory.BookMethod(dataloader, TMVA.Types.kDL, rnnName, rnnOptions)


# Book TMVA  fully connected dense layer  models

if (useTMVA_DNN):
#    Method DL with Dense Layer
    inputLayoutString = "InputLayout=1|1|" + str(ntime * ninput)

    layoutString = "Layout=DENSE|64|TANH,DENSE|TANH|64,DENSE|TANH|64,LINEAR"
#   Training strategies.
    trainingString1 = "LearningRate=1e-3,Momentum=0.0,Repetitions=1,"+"ConvergenceSteps=10,BatchSize=256,TestRepetitions=1,"+"WeightDecay=1e-4,Regularization=None,MaxEpochs=20"+"DropConfig=0.0+0.+0.+0.,Optimizer=ADAM"
    trainingStrategyString = "TrainingStrategy="
    trainingStrategyString += trainingString1 # + "|" + trainingString2

      # General Options.
    dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"+"WeightInitialization=XAVIER:RandomSeed=0" 

    dnnOptions +=  ":" + inputLayoutString
    dnnOptions +=  ":" + layoutString
    dnnOptions +=  ":" + trainingStrategyString
    dnnOptions +=  ":" + "Architecture=" + str(archString)


    dnnName = "TMVA_DNN"
    factory.BookMethod(dataloader, TMVA.Types.kDL, dnnName, dnnOptions)
   


# Book Keras recurrent models
# 
# Book the different types of recurrent models in Keras  (SimpleRNN, LSTM or GRU)




if (useKeras):
    for i in range(3):
        if (use_rnn_type[i]):
            modelName = "model_" + str(rnn_types[i]) + ".h5"
            trainedModelName = "trained_model_"+ str(rnn_types[i]) + ".h5"

            ROOT.Info("TMVA_RNN_Classification", "Building recurrent keras model using a"+str(rnn_types[i])+" layer")
            # create python script which can be executed
            # create 2 conv2d layer + maxpool + dense
        
            
            
            model = Sequential()
            model.add(Reshape((10, 30), input_shape = (10*30, )))
            # add recurrent neural network depending on type / Use option to return the full output
            if (rnn_types[i] == "LSTM"):
               model.add(LSTM(units=10, return_sequences=True) )
            elif (rnn_types[i] == "GRU"):
               model.add(GRU(units=10, return_sequences=True) )
            else:
               model.add(SimpleRNN(units=10, return_sequences=True) )

            model.add(BatchNormalization())
            model.add(Flatten())# needed if returning the full time output sequen
            model.add(Dense(64, activation = 'tanh')) 
            model.add(Dense(2, activation = 'sigmoid')) 
            model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
            
            model.save(modelName)
            model.summary()

#             m.SaveSource("make_rnn_model.py");
#              execute
            ROOT.gSystem.Exec("python make_rnn_model.py")

            if (ROOT.gSystem.AccessPathName(modelName)):
               Warning("TMVA_RNN_Classification", "Error creating Keras recurrent model file - Skip using Keras")
               useKeras = False
            else:
               # book PyKeras method only if Keras model could be created
               ROOT.Info("TMVA_RNN_Classification", "Booking Keras" + str(rnn_types[i]) +  "model")
               factory.BookMethod(dataloader, TMVA.Types.kPyKeras,"PyKeras_"+ str(rnn_types[i]),"!H:!V:VarTransform=None:FilenameModel="+str(modelName)+":tf.keras:"+"FilenameTrainedModel="+str(trainedModelName)+":GpuOptions=allow_growth=True:"+"NumEpochs="+str(maxepochs)+":BatchSize="+str(batchSize))
                                                   


# Training All Methods
# 
# Here we train all the previously booked methods.

# Train all methods
factory.TrainAllMethods()


# Test all methods
# 
# Now we test  all methods using the test data set

print("nthreads  = "+ str(ROOT.GetThreadPoolSize()) + "\n")

# Evaluate all MVAs using the set of test events
factory.TestAllMethods()


# Evaluate all methods
# 
# Here we evaluate all methods and compare their performances, computing efficiencies, ROC curves etc.. using both training and tetsing data sets. Several histograms are produced which can be examined with the TMVAGui or directly using the output file

# Evaluate and compare performance of all configured MVAs
factory.EvaluateAllMethods()
#  check method


# Plot ROC Curve
# Here we plot the ROC curve and display the same.

#  plot ROC curve
c1 = factory.GetROCCurve(dataloader)
c1.Draw()  


# Close the Output File
# Close outputfile to save all output information (evaluation result of methods) and it can be used by TMVAGUI to display additional plots

if (outputFile):
    outputFile.Close()
