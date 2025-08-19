# TMVA Classification Example Using a Convolutional Neural Network

# This is an example of using a CNN in TMVA. We do classification using a toy image data set that is generated when running the example macro.

# Helper function to create input images data we create a signal and background 2D histograms from 2d gaussians with a location (means in X and Y)  different for each event The difference between signal and background is in the gaussian width. The width for the background gaussian is slightly larger than the signal width by few % values

# First you need to run TMVA_CNN_Classification.C to generate images_data_16x16.root.

import ROOT
from ROOT import TMVA 
import os
from array import array
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, BatchNormalization

# Setting up TMVA

ROOT.TMVA.Tools.Instance()
# TMVA requires initialization the PyMVA to utilize PyTorch. PyMVA is the interface for third-party MVA tools based on Python. It is created to make powerful external libraries easily accessible with a direct integration into the TMVA workflow. All PyMVA methods provide the same plug-and-play mechanisms as the TMVA methods. Because the base method of PyMVA is inherited from the TMVA base method, all options of internal TMVA methods apply for PyMVA methods as well.
# For PYMVA methods
TMVA.PyMethodBase.PyInitialize()

#  Create an Output File and Declare Factory
# 
# Create the Factory class. Later you can choose the methods whose performance you'd like to investigate.
# 
# The factory is the major TMVA object you have to interact with. Here is the list of parameters you need to pass
# 
# - The first argument is the base of the name of all the output weightfiles in the directory weight/ that will be created with the method parameters
# 
# - The second argument is the output file for the training results
# 
# - The third argument is a string option defining some general configuration for the TMVA session. For example all TMVA output can be suppressed by removing the "!" (not) in front of the "Silent" argument in the option string
# 

outputFile = ROOT.TFile.Open("CNN_ClassificationOutput.root", "RECREATE")

factory = ROOT.TMVA.Factory("TMVA_CNN_Classification", outputFile,
                      "!V:ROC:!Silent:Color:!DrawProgressBar:AnalysisType=Classification" )


# Define the Options and number of threads


opt=[1,1,1,1,1]
useTMVACNN = opt[0] if (len(opt) > 0) else False
useKerasCNN = opt[0] if (len(opt) > 1) else False
useTMVADNN = opt[0] if (len(opt) > 2) else False
useTMVABDT = opt[0] if (len(opt) > 3) else False
usePyTorchCNN = opt[0] if (len(opt) > 4) else False

writeOutputFile = True

num_threads = 0  # use default threads

# do enable MT running
if (num_threads >= 0):
  ROOT.EnableImplicitMT(num_threads)
  if (num_threads > 0):
     ROOT.gSystem.Setenv("OMP_NUM_THREADS", ROOT.TString.Format("%d",num_threads))

else:
  ROOT.gSystem.Setenv("OMP_NUM_THREADS", "1")

print("Running with nthreads  = " + str(ROOT.GetThreadPoolSize()) )


if __debug__:
    ROOT.gSystem.Setenv("KERAS_BACKEND", "tensorflow")
    # for using Keras
    # TMVA.PyMethodBase.PyInitialize()
else:
    useKerasCNN = False

factory =  ROOT.TMVA.Factory (
  "TMVA_CNN_Classification", outputFile,
  "!V:ROC:!Silent:Color:AnalysisType=Classification:Transformations=None:!Correlations")


#  Declare DataLoader(s)
# 
#   The next step is to declare the DataLoader class that deals with input variables
# 
#   Define the input variables that shall be used for the MVA training
#   note that you may also use variable expressions, which can be parsed by TTree::Draw( "expression" )]
# 
#   In this case the input data consists of an image of 16x16 pixels. Each single pixel is a branch in a ROOT TTree

loader = ROOT.TMVA.DataLoader("dataset")

# Setup Dataset(s)
# Define input data file and setup the signal and background trees

imgSize = 16 * 16
inputFileName = "images_data_16x16.root"
inputFile = ROOT.TFile.Open(inputFileName)
if (inputFile == None):
    Error("TMVA_CNN_Classification", "Error opening input file %s - exit", inputFileName.Data())
signalTree     = inputFile.Get("sig_tree")
backgroundTree = inputFile.Get("bkg_tree")

signalTree.Print()



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
 
loader.AddVariablesArray("vars",imgSize,'F')

# Set individual event weights (the variables must exist in the original TTree)
#    for signal    : factory.SetSignalWeightExpression    ("weight1*weight2")
#    for background: factory.SetBackgroundWeightExpression("weight1*weight2")
# loader.SetBackgroundWeightExpression( "weight" )


# Apply additional cuts on the signal and background samples (can be different)
mycuts = ROOT.TCut("") # for example: TCut mycuts = "abs(var1)<0.5 && abs(var2-0.5)<1";
mycutb = ROOT.TCut("") # for example: TCut mycutb = "abs(var1)<0.5";


# Tell the factory how to use the training and testing events

#  If no numbers of events are given, half of the events in the tree are used
#  for training, and the other half for testing:
#     loader.PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );
#  It is possible also to specify the number of training and testing events,
#  note we disable the computation of the correlation matrix of the input variables

nTrainSig = 0.8 * nEventsSig
nTrainBkg = 0.8 * nEventsBkg

#  build the string options for DataLoader::PrepareTrainingAndTestTree
prepareOptions = "nTrain_Signal="+str(nTrainSig)+":nTrain_Background="+str(nTrainBkg)+":SplitMode=Random:SplitSeed=100:NormMode=NumEvents:!V:!CalcCorrelations"
  
loader.PrepareTrainingAndTestTree(mycuts, mycutb, prepareOptions)


# Booking Methods
# 
# Here we book the TMVA methods. We book a Boosted Decision Tree method (BDT)

# Boosted Decision Trees
if (useTMVABDT):
  factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT","!V:NTrees=400:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:"+"UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")


#    ### Booking Deep Neural Network
# 
#    Here we define the option string for building the Deep Neural network model.
# 
#    #### 1. Define DNN layout
# 
#    The DNN configuration is defined using a string. Note that whitespaces between characters are not allowed.
# 
#    We define first the DNN layout:
# 
#    - **input layout** :   this defines the input data format for the DNN as  ``input depth | height | width``.
#       In case of a dense layer as first layer the input layout should be  ``1 | 1 | number of input variables`` (features)
#    - **batch layout**  : this defines how are the input batch. It is related to input layout but not the same.
#       If the first layer is dense it should be ``1 | batch size ! number of variables`` (features)
# 
#       *(note the use of the character `|` as  separator of  input parameters for DNN layout)*
# 
#    note that in case of only dense layer the input layout could be omitted but it is required when defining more
#    complex architectures
# 
#    - **layer layout** string defining the layer architecture. The syntax is
#       - layer type (e.g. DENSE, CONV, RNN)
#       - layer parameters (e.g. number of units)
#       - activation function (e.g  TANH, RELU,...)
# 
#       *the different layers are separated by the ``","`` *
# 
#    #### 2. Define Training Strategy
# 
#    We define here the training strategy parameters for the DNN. The parameters are separated by the ``","`` separator.
#    One can then concatenate different training strategy with different parameters. The training strategy are separated by
#    the ``"|"`` separator.
# 
#    - Optimizer
#    - Learning rate
#    - Momentum (valid for SGD and RMSPROP)
#    - Regularization and Weight Decay
#    - Dropout
#    - Max number of epochs
#    - Convergence steps. if the test error will not decrease after that value the training will stop
#    - Batch size (This value must be the same specified in the input layout)
#    - Test Repetitions (the interval when the test error will be computed)
# 
# 
#    #### 3. Define general DNN options
# 
#    We define the general DNN options concatenating in the final string the previously defined layout and training strategy.
#    Note we use the ``":"`` separator to separate the different higher level options, as in the other TMVA methods.
#    In addition to input layout, batch layout and training strategy we add now:
# 
#    - Type of Loss function (e.g. CROSSENTROPY)
#    - Weight Initizalization (e.g XAVIER, XAVIERUNIFORM, NORMAL )
#    - Variable Transformation
#    - Type of Architecture (e.g. CPU, GPU, Standard)
# 
#    We can then book the DL method using the built option string
# 

# Define the DNN layout
if (useTMVADNN):
  layoutString = "Layout=DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,BNORM,DENSE|100|RELU,DENSE|1|LINEAR"

  #  Training strategies
  #  one can catenate several training strings with different parameters (e.g. learning rates or regularizations
  #  parameters) The training string must be concatenates with the `|` delimiter
  trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"+ "ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"+"MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"+"Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0."
                          

  trainingStrategyString = "TrainingStrategy="
  trainingStrategyString += trainingString1 # + "|" + trainingString2 + ....

  # Build now the full DNN Option string

  dnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"+"WeightInitialization=XAVIER"
  dnnOptions+= ":"
  dnnOptions+= layoutString
  dnnOptions+= ":"
  dnnOptions+= trainingStrategyString

  dnnMethodName = "TMVA_DNN_CPU"

  dnnOptions += ":Architecture=CPU"


factory.BookMethod(loader, ROOT.TMVA.Types.kDL, dnnMethodName, dnnOptions)


# 
# ### Book Convolutional Neural Network in TMVA
# 
# For building a CNN one needs to define
# 
# -  Input Layout :  number of channels (in this case = 1)  | image height | image width
# -  Batch Layout :  batch size | number of channels | image size = (height*width)
# 
# Then one add Convolutional layers and MaxPool layers.
# 
# -  For Convolutional layer the option string has to be:
#    - CONV | number of units | filter height | filter width | stride height | stride width | padding height | paddig
# width | activation function
# 
#    - note in this case we are using a filer 3x3 and padding=1 and stride=1 so we get the output dimension of the
# conv layer equal to the input
# 
#   - note we use after the first convolutional layer a batch normalization layer. This seems to help significantly the
# convergence
# 
#  - For the MaxPool layer:
#     - MAXPOOL  | pool height | pool width | stride height | stride width
# 
# The RESHAPE layer is needed to flatten the output before the Dense layer
# 
# 
# Note that to run the CNN is required to have CPU  or GPU support
# 

inputLayoutString ="InputLayout=1|16|16"

#  Batch Layout
layoutString = "Layout=CONV|10|3|3|1|1|1|1|RELU,BNORM,CONV|10|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,"+"RESHAPE|FLAT,DENSE|100|RELU,DENSE|1|LINEAR"

#  Training strategies.
trainingString1 = "LearningRate=1e-3,Momentum=0.9,Repetitions=1,"+"ConvergenceSteps=5,BatchSize=100,TestRepetitions=1,"+"MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"+"Optimizer=ADAM,DropConfig=0.0+0.0+0.0+0.0"

trainingStrategyString = "TrainingStrategy="
trainingStrategyString += trainingString1 # + "|" + trainingString2 + "|" + trainingString3; for concatenating more training strings

# Build full CNN Options.


cnnOptions = "!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:" +"WeightInitialization=XAVIER::Architecture=CPU"

cnnOptions +=  ":" + inputLayoutString
cnnOptions +=  ":" + layoutString
cnnOptions +=  ":" + trainingStrategyString
  ## New DL (CNN)
cnnMethodName = "TMVA_CNN_CPU"
# use GPU if available


cnnOptions += ":Architecture=CPU"
cnnMethodName = "TMVA_CNN_CPU"


factory.BookMethod(loader, ROOT.TMVA.Types.kDL, cnnMethodName, cnnOptions)
# Book Convolutional Neural Network in Keras using a generated model
ROOT.Info("TMVA_CNN_Classification", "Building convolutional keras model")
#  create python script which can be executed
#  create 2 conv2d layer + maxpool + dense

model = Sequential()
model.add(Reshape((16, 16, 1), input_shape = (256, )))
model.add(Conv2D(10, kernel_size=(3,3), kernel_initializer='TruncatedNormal', activation='relu', padding='same' ) )
model.add(Conv2D(10, kernel_size=(3,3), kernel_initializer='glorot_normal', activation ='relu', padding = 'same') )
model.add(BatchNormalization())
model.add(Conv2D(10, kernel_size = (3,3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same') )
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2), strides = (1,1))) 
model.add(Flatten())
model.add(Dense(256, activation = 'relu')) 
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy'])
model.save('model_cnn.h5')
model.summary()



if (ROOT.gSystem.AccessPathName("model_cnn.h5")):
 Warning("TMVA_CNN_Classification", "Error creating Keras model file - skip using Keras")
else:
 #  book PyKeras method only if Keras model could be created
 ROOT.Info("TMVA_CNN_Classification", "Booking tf.Keras CNN model")


factory.BookMethod(loader, ROOT.TMVA.Types.kPyKeras, "PyKeras","H:!V:VarTransform=None:FilenameModel=model_cnn.h5:"+"FilenameTrainedModel=trained_model_cnn.h5:NumEpochs=20:BatchSize=128")

# Training All Methods

# Here we train all the previously booked methods.

# Train Methods

factory.TrainAllMethods()


# Test all methods
# Now we test  all methods using the test data set

factory.TestAllMethods()

#  Evaluate all methods
# 
# Here we evaluate all methods and compare their performances, computing efficiencies, ROC curves etc.. using both training and tetsing data sets. Several histograms are produced which can be examined with the TMVAGui or directly using the output file

factory.EvaluateAllMethods()

# Plot ROC Curve
# Here we plot the ROC curve and display the same.

## Plot ROC Curve

c1 = factory.GetROCCurve(loader)
c1.Draw()


# Close the Output File
# Close outputfile to save all output information (evaluation result of methods) and it can be used by TMVAGUI to display additional plots

# close outputfile to save output file
outputFile.Close()
