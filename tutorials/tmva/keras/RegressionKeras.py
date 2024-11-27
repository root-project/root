#!/usr/bin/env python
# \file
# \ingroup tutorial_tmva_keras
# \notebook -nodraw
# This tutorial shows how to do regression in TMVA with neural networks
# trained with keras.
#
# \macro_code
#
# \date 2017
# \author TMVA Team

from ROOT import TMVA, TFile, TCut
from subprocess import call
from os.path import isfile

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


def create_model():
    # Define model
    model = Sequential()
    model.add(Dense(64, activation='tanh', input_dim=2))
    model.add(Dense(1, activation='linear'))

    # Set loss and optimizer
    model.compile(loss='mean_squared_error', optimizer=SGD(
        learning_rate=0.01), weighted_metrics=[])

    # Store model to file
    model.save('modelRegression.h5')
    model.summary()


def run():

    with TFile.Open('TMVA_Regression_Keras.root', 'RECREATE') as output, TFile.Open('tmva_reg_example.root') as data:
        factory = TMVA.Factory('TMVARegression', output,
                               '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Regression')

        tree = data.Get('TreeR')

        dataloader = TMVA.DataLoader('dataset')
        for branch in tree.GetListOfBranches():
            name = branch.GetName()
            if name != 'fvalue':
                dataloader.AddVariable(name)
        dataloader.AddTarget('fvalue')

        dataloader.AddRegressionTree(tree, 1.0)
        # use only 1000 events since evaluation is very slow (especially on MacOS). Increase it to get meaningful results
        dataloader.PrepareTrainingAndTestTree(TCut(''),
                                              'nTrain_Regression=1000:SplitMode=Random:NormMode=NumEvents:!V')

        # Book methods
        factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                           'H:!V:VarTransform=D,G:FilenameModel=modelRegression.h5:FilenameTrainedModel=trainedModelRegression.h5:NumEpochs=20:BatchSize=32')
        factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDTG',
                           '!H:!V:VarTransform=D,G:NTrees=1000:BoostType=Grad:Shrinkage=0.1:UseBaggedBoost:BaggedSampleFraction=0.5:nCuts=20:MaxDepth=4')

        # Run TMVA
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()


if __name__ == "__main__":
    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    # Load data
    if not isfile('tmva_reg_example.root'):
        call(['curl', '-L', '-O', 'http://root.cern/files/tmva_reg_example.root'])

    # Generate model
    create_model()

    # Run TMVA
    run()
