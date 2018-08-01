// @(#)root/tmva/tmva/cnn:$Id$
// Author: Siddhartha Rao Kamalakara

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Method DL for Regression                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Siddhartha Rao Kamalakara       <srk97c@gmail.com>   - CERN, Switzerland  *
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

#ifndef TMVA_TEST_DNN_TEST_REGRESSION_TEST_METHOD_DL_H
#define TMVA_TEST_DNN_TEST_REGRESSION_TEST_METHOD_DL_H

#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TROOT.h"

#include "TMVA/MethodDL.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/Config.h"


#include <iostream>

TTree *genTree(Int_t nPoints, UInt_t seed = 100)
{
   TRandom3 rng(seed);
   Float_t u_a = 0;
   Float_t u_b = 0;
   Float_t a_b_add = 0;
   Float_t a_b_sub = 0;

   auto data = new TTree();
   data->Branch("uniform1", &u_a, "uniform1/F");
   data->Branch("uniform2", &u_b, "uniform2/F");
   data->Branch("uniform_add", &a_b_add, "uniform_add/F");
   data->Branch("uniform_sub", &a_b_sub, "uniform_sub/F");

   for (Int_t n = 0; n < nPoints; ++n) {
      u_a = rng.Uniform();
      u_b = rng.Uniform();
      a_b_add = u_a + u_b;
      a_b_sub = u_a - u_b;

      data->Fill();
   }

   // Important: Disconnects the tree from the memory locations of x and y.
   data->ResetBranchAddresses();
   return data;
}

/** Testing the entire pipeline of the Method DL, when only a Multilayer Percepton
 *  is constructed. */
//______________________________________________________________________________
int main()
{

   TString architectureStr("CPU");
   ROOT::EnableImplicitMT(1);
   TMVA::Config::Instance();
   
   //TFile *input(0);

   //input = TFile::Open("~/Documents/gsoc/root/tree.root", "CACHEREAD");

   TString outfileName("TMVA_DNN_Regression.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TTree *dataTree = genTree(10000);

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable( "uniform1", "Variable 1", "units", 'F' );
   dataloader->AddVariable( "uniform2", "Variable 2", "units", 'F' );
   dataloader->AddVariable( "uniform_add", "Variable 3", "units", 'F' );
   dataloader->AddVariable( "uniform_sub", "Variable 4", "units", 'F' );

   dataloader->AddTarget("uniform1");
   dataloader->AddTarget("uniform2");
   dataloader->AddTarget("uniform_add");
   dataloader->AddTarget("uniform_sub");

   Double_t regWeight  = 1.0;

   dataloader->AddRegressionTree( dataTree, regWeight );

   TCut mycut = "";

   dataloader->PrepareTrainingAndTestTree( mycut,"nTrain_Regression=9000:nTest_Regression=1000:SplitMode=Random:NormMode=NumEvents:!V" );

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|4");

   // Batch Layout
   TString batchLayoutString("BatchLayout=16|1|4");

   // General layout.
   TString layoutString("Layout=RESHAPE|1|1|4|FLAT,DENSE|2|SIGMOID,DENSE|4|LINEAR");

   // Training strategies.
   TString training0("LearningRate=1e-2,Momentum=0.0,Repetitions=1000,"
                     "ConvergenceSteps=100,BatchSize=16,TestRepetitions=10,"
                     "WeightDecay=0,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training1("LearningRate=1e-2,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString training2("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,"
                     "WeightDecay=1e-4,Regularization=L2,"
                     "DropConfig=0.0+0.0+0.0+0.0, Multithreading=True");
   TString trainingStrategyString("TrainingStrategy=");
   trainingStrategyString += training0 + "|" + training1 + "|" + training2;

   // General Options.
   TString dnnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:""WeightInitialization=XAVIERUNIFORM");
   
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
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,"!V:!Silent:Color:DrawProgressBar:AnalysisType=Regression" );

   TString methodTitle = "DL_" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kDL, methodTitle, dnnOptions);

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   factory->TestAllMethods();

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;

   delete factory;
   delete dataloader;
}

#endif
