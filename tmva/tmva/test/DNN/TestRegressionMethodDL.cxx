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
#include "TH1.h"
#include "TCanvas.h"

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

   TString outfileName("TMVA_DNN_MultiRegression.root");
   TFile *outputFile = TFile::Open(outfileName, "RECREATE");

   TTree *dataTree = genTree(50000);

   TMVA::DataLoader *dataloader = new TMVA::DataLoader("dataset");

   dataloader->AddVariable( "uniform1", "Variable 1", "units", 'F' );
   dataloader->AddVariable( "uniform2", "Variable 2", "units", 'F' );

   dataloader->AddTarget("uniform_add");
   dataloader->AddTarget("uniform_sub");

   Double_t regWeight  = 1.0;

   dataloader->AddRegressionTree( dataTree, regWeight );

   TCut mycut = "";

   dataloader->PrepareTrainingAndTestTree( mycut,"nTrain_Regression=40000:nTest_Regression=10000:SplitMode=Random:NormMode=NumEvents:!V" );

   // Input Layout
   TString inputLayoutString("InputLayout=1|1|2");

   // Batch Layout
   TString batchLayoutString("BatchLayout=128|1|2");

   // General layout.
   TString layoutString("Layout=RESHAPE|1|1|2|FLAT,DENSE|2|SIGMOID,DENSE|2|LINEAR");

   // Training strategy
   TString training0("LearningRate=1e-3,Momentum=0.0,Repetitions=10,"
                     "ConvergenceSteps=10,BatchSize=128,TestRepetitions=10,MaxEpochs=100"
                     "WeightDecay=0,Regularization=None,Optimizer=ADAM,"
                     "DropConfig=0.0+0.0+0.0+0.0");
   TString trainingStrategyString("TrainingStrategy=");
   trainingStrategyString += training0;

   // General Options.
   TString dnnOptions("!H:V:ErrorStrategy=SUMOFSQUARES:WeightInitialization=XAVIERUNIFORM");
   
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
   TMVA::Factory *factory = new TMVA::Factory( "TMVARegression", outputFile,"!V:!Silent:Color:!DrawProgressBar:AnalysisType=Regression" );

   TString methodTitle = "DL_MultiReg" + architectureStr;
   factory->BookMethod(dataloader, TMVA::Types::kDL, methodTitle, dnnOptions);

   // // Boosted Decision Trees 
   // factory->BookMethod(dataloader,TMVA::Types::kBDT, "BDTG_LS",
   //                    TString("!H:!V:NTrees=64::BoostType=Grad:Shrinkage=0.3:nCuts=20:MaxDepth=4:")+
   //                    TString("RegressionLossFunctionBDTG=LeastSquares"));
   

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   factory->TestAllMethods();

   // ebaluate all method to save result in outpt file
   factory->EvaluateAllMethods();
   

   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;

   delete factory;
   delete dataloader;

   // check the resulted target value   
   auto resultsFile = TFile::Open(outfileName);
   if (!resultsFile) {
      std::cout << "ERROR in output file " << std::endl;
      return -1; 
   }
   auto resultsTree = (TTree*) resultsFile->Get("dataset/TestTree"); 
   if (!resultsTree) {
      std::cout << "ERROR in reading output tree " << std::endl;
      return -1; 
   }
   auto h1 = new TH1D("h1","Reg var1 - true target1",100,1,0);
   auto h2 = new TH1D("h2","Reg var2 - true target2",100,1,0);
   resultsTree->Draw("DL_MultiRegCPU.uniform_add-uniform_add >> h1"); 
   resultsTree->Draw("DL_MultiRegCPU.uniform_sub-uniform_sub >> h2");
   auto c1 = new TCanvas();
   c1->Divide(1,2);
   c1->cd(1);
   h1->Draw();
   c1->cd(2);
   h2->Draw(); 
   c1->SaveAs("DL_MultiRegCPU.pdf");

   if (std::abs( h1->GetMean() / h1->GetMeanError() ) > 10. )  
      std::cout << "ERROR in reconstructing target 1 " << std::endl;
   if (std::abs( h2->GetMean() / h2->GetMeanError() ) > 10.  ) 
      std::cout << "ERROR in reconstructing target 2 " << std::endl;

   std::cout << "< Deviation target 1> = " << h1->GetMean()  << " +/- " << h1->GetMeanError() << std::endl;
   std::cout << "< Deviation target 2> = " << h2->GetMean()  << " +/- " << h2->GetMeanError() << std::endl;
}

#endif
