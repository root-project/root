// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Method DL for Conv Net                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_DL_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_DL_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"

#include "TMVA/MethodDL.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"

#include <iostream>

/** Testing the entire pipeline of the Method DL, when only a Convolutional Net
 *  is constructed. */
//______________________________________________________________________________
void testMethodDL(TString architectureStr)
{
   // Load the photon input
   TFile *photonInput(0);
   TString photonFileName =
      "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/test/DNN/CNN/"
      "dataset/SinglePhotonPt50_FEVTDEBUG_n250k_IMG_CROPS32.root";
   photonInput = TFile::Open(photonFileName);

   // Load the electron input
   TFile *electronInput(0);
   TString electronFileName = "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/"
                              "test/DNN/CNN/dataset/SingleElectronPt50_FEVTDEBUG_n250k_IMG_CROPS32.root";
   electronInput = TFile::Open(electronFileName);

   // Get the trees
   TTree *signalTree = (TTree *)photonInput->Get("RHTree");
   TTree *background = (TTree *)electronInput->Get("RHTree");

   // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
   TString outfileName("TMVA_MethodDL.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   // global event weights per tree
   Double_t signalWeight = 1.0;
   Double_t backgroundWeight = 1.0;

   // Add signal and background trees
   dataloader->AddSignalTree(signalTree, signalWeight);
   dataloader->AddBackgroundTree(background, backgroundWeight);

   // dataloader->SetBackgroundWeightExpression("weight");

   TCut mycuts = "";
   TCut mycutb = "";

   dataloader->PrepareTrainingAndTestTree(
      mycuts, mycutb, "nTrain_Signal=10000:nTrain_Background=10000:SplitMode=Random:NormMode=NumEvents:!V");

   // Input Layout
   TString inputLayoutString("InputLayout=1|32|32");

   // Input Layout
   TString batchLayoutString("BatchLayout=256|1|1024");

   // General layout.
   TString layoutString("Layout=CONV|12|2|2|1|1|1|1|TANH,MAXPOOL|6|6|1|1,RESHAPE|1|1|9408|FLAT,DENSE|512|TANH,DENSE|32|"
                        "TANH,DENSE|2|LINEAR");

   // Training strategies.
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString trainingStrategyString("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.
   TString dnnOptions("!H:V:ErrorStrategy=CROSSENTROPY:"
                      "WeightInitialization=XAVIERUNIFORM");

   // Concatenate all option strings
   dnnOptions.Append(":");
   dnnOptions.Append(inputLayoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(batchLayoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(layoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(trainingStrategyString);

   dnnOptions.Append(":Architecture=");
   dnnOptions.Append(architectureStr);

   // Create a factory for booking the method
   TMVA::Factory *factory =
      new TMVA::Factory("TMVAClassification", outputFile,
                        "!V:!Silent:Color:DrawProgressBar:Transformations=I:AnalysisType=Classification");

   TString methodTitle = "DL_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kDL, methodTitle, dnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
}

/** Testing the entire pipeline of the Method DL, when only a Multilayer Percepton
 *  is constructed. */
//______________________________________________________________________________
void testMethodDL_DNN(TString architectureStr)
{
   TFile *input(0);
   TString fname = "/Users/vladimirilievski/Desktop/Vladimir/GSoC/ROOT-CI/common-version/root/tmva/tmva/test/DNN/CNN/"
                   "dataset/tmva_class_example.root";
   input = TFile::Open(fname);

   // Register the training and test trees
   TTree *signalTree = (TTree *)input->Get("TreeS");
   TTree *background = (TTree *)input->Get("TreeB");

   TString outfileName("TMVA_DNN.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable("myvar1 := var1+var2", 'F');
   dataloader->AddVariable("myvar2 := var1-var2", "Expression 2", "", 'F');
   dataloader->AddVariable("var3", "Variable 3", "units", 'F');
   dataloader->AddVariable("var4", "Variable 4", "units", 'F');
   dataloader->AddSpectator("spec1 := var1*2", "Spectator 1", "units", 'F');
   dataloader->AddSpectator("spec2 := var1*3", "Spectator 2", "units", 'F');

   // global event weights per tree
   Double_t signalWeight = 1.0;
   Double_t backgroundWeight = 1.0;

   // Add signal and background trees
   dataloader->AddSignalTree(signalTree, signalWeight);
   dataloader->AddBackgroundTree(background, backgroundWeight);

   dataloader->SetBackgroundWeightExpression("weight");

   TCut mycuts = "";
   TCut mycutb = "";

   dataloader->PrepareTrainingAndTestTree(
      mycuts, mycutb, "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V");

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|4");

   // Batch Layout
   TString batchLayoutString("BatchLayout=1|256|4");

   // General layout.
   TString layoutString("Layout=DENSE|128|LINEAR,DENSE|128|LINEAR,DENSE|128|LINEAR,DENSE|1|LINEAR");

   // Training strategies.
   TString training0("LearningRate=1e-1,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=1,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.5+0.5+0.5, Multithreading=True");
   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=1,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("LearningRate=1e-3,Momentum=0.0,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=1,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString trainingStrategyString("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.
   TString dnnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:"
                      "WeightInitialization=XAVIERUNIFORM");

   // Concatenate all option strings
   dnnOptions.Append(":");
   dnnOptions.Append(inputLayoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(batchLayoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(layoutString);

   dnnOptions.Append(":");
   dnnOptions.Append(trainingStrategyString);

   dnnOptions.Append(":Architecture=");
   dnnOptions.Append(architectureStr);

   // create factory
   TMVA::Factory *factory =
      new TMVA::Factory("TMVAClassification", outputFile,
                        "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification");

   TString methodTitle = "DL_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kDL, methodTitle, dnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
}

#endif
