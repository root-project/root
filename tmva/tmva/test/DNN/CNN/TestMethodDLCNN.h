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

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_DL_CNN_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_METHOD_DL_CNN_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/MethodDL.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Config.h"

#include "MakeImageData.h"

#include <iostream>

/** Testing the entire pipeline of the Method DL, when only a Convolutional Net
 *  is constructed. */
//______________________________________________________________________________
void testMethodDL_CNN(TString architectureStr)
{
   // Load the input data

   TString fname = "imagesData.root";

   // generate the files
   // 1000 for testing 1000 for training
   makeImages(2000,8,8);

   auto input = TFile::Open(fname);

   R__ASSERT(input);

   // Get the trees
   TTree *signalTree = (TTree *)input->Get("sgn");
   TTree *background = (TTree *)input->Get("bkg");

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

   for (int i = 0; i < 8; ++i) {
      for (int j = 0; j < 8; ++j) {
         int ivar = i * 8 + j;
         TString varName = TString::Format("var%d", ivar);
         dataloader->AddVariable(varName, 'F');
      }
   }

   dataloader->PrepareTrainingAndTestTree(
      mycuts, mycutb, "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V");

   // Input Layout
   TString inputLayoutString("InputLayout=1|8|8");

   // Input Layout
   TString batchLayoutString("BatchLayout=256|1|64");

   // General layout.
   TString layoutString("Layout=CONV|6|3|3|1|1|0|0|TANH,MAXPOOL|2|2|2|2,RESHAPE|FLAT,DENSE|10|TANH,"
                        "DENSE|2|LINEAR");

   // Training strategies.
   TString training0("LearningRate=1e-1,Optimizer=SGD,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,MaxEpochs=40"
                     "DropConfig=0.0+0.5+0.5+0.5");
   TString training1("LearningRate=1e-2,Optimizer=SGD,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=5,"
                     "WeightDecay=1e-4,Regularization=L2,MaxEpochs=20"
                     "DropConfig=0.0+0.0+0.0+0.0");
   TString training2("LearningRate=1e-3,Optimizer=SGD,Momentum=0.0,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=5,"
                     "WeightDecay=1e-4,Regularization=L2,MaxEpochs=20"
                     "DropConfig=0.0+0.0+0.0+0.0");
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

   TString methodTitle = "DL_CNN_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kDL, methodTitle, dnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // Train MVAs using the set of training events
   factory->TestAllMethods();

   // Train MVAs using the set of training events
   factory->EvaluateAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVAClassification is done!" << std::endl;

   delete factory;
   delete dataloader;
}


#endif
