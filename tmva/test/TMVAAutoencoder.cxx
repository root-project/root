// @(#)root/tmva $Id$
/**********************************************************************************
 * Project   : TMVA - a Root-integrated toolkit for multivariate data analysis    *
 * Package   : TMVA                                                               *
 * Exectuable: TMVARegression                                                     *
 *                                                                                *
 * This executable provides examples for the training and testing of the          *
 * TMVA classifiers.                                                              *
 *                                                                                *
 * As input data is used a toy-MC sample consisting of four Gaussian-distributed  *
 * and linearly correlated input variables.                                       *
 *                                                                                *
 * The methods to be used can be switched on and off by means of booleans.        *
 *                                                                                *
 * Compile and run the example with the following commands                        *
 *                                                                                *
 *    make                                                                        *
 *    ./TMVAAutoencoder                                                           *
 *                                                                                *
 *                                                                                *
 * The output file "TMVAReg.root" can be analysed with the use of dedicated       *
 * macros (simply say: root -l <../macros/macro.C>), which can be conveniently    *
 * invoked through a GUI launched by the command                                  *
 *                                                                                *
 *    root -l ../macros/TMVAGui.C                                                 *
 **********************************************************************************/

#include <cstdlib>
#include <iostream> 
#include <map>
#include <string>
#include <algorithm>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"

#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodMLP.h"



int factory() 
{
   // The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
   // if you use your private .rootrc, or run from a different directory, please copy the 
   // corresponding lines from .rootrc

   // methods to be processed can be given as an argument; use format:
   //

   std::cout << std::endl;
   std::cout << "==> Start TMVAAutoencoder" << std::endl;

   // --------------------------------------------------------------------------------------------------

   // --- Here the preparation phase begins

   // Create a new root output file
   TString outfileName( "TMVAReg.root" );
   TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

   // Create the factory object. Later you can choose the methods
   // whose performance you'd like to investigate. The factory will
   // then run the performance analysis for you.
   //
   // The first argument is the base of the name of all the
   // weightfiles in the directory weight/ 
   //
   // The second argument is the output file for the training results
   // All TMVA output can be suppressed by removing the "!" (not) in 
   // front of the "Silent" argument in the option string
   TMVA::Factory *factory = new TMVA::Factory( "TMVAAutoencoder", outputFile, 
                                               "!V:!Silent:Color:DrawProgressBar" );

   // If you wish to modify default settings 
   // (please check "src/Config.h" to see all available global options)
   //    (TMVA::gConfig().GetVariablePlotting()).fTimesRMS = 8.0;
   //    (TMVA::gConfig().GetIONames()).fWeightFileDir = "myWeightDirectory";

   // Define the input variables that shall be used for the MVA training
   // note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
   // [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
   factory->AddVariable( "var1", "Variable 1", "units", 'F' );
   factory->AddVariable( "var2", "Variable 2", "units", 'F' );
   factory->AddVariable( "var3:=var1+var2", "Variable 3", "units", 'F' );
   factory->AddVariable( "var4:=var2*var1", "Variable 4", "units", 'F' );

   // Add the variable carrying the regression target
   factory->AddTarget( "var1" ); 
   factory->AddTarget( "var2" ); 
   factory->AddTarget( "var1+var2" ); 
   factory->AddTarget( "var2*var1" ); 

   // Read training and test data (see TMVAClassification for reading ASCII files)
   // load the signal and background event samples from ROOT trees
   TFile *input(0);
   TString fname = "./tmva_reg_example.root";
   if (!gSystem->AccessPathName( fname )) 
   input = TFile::Open( fname ); // check if file in local directory exists
   else 
      input = TFile::Open( "http://root.cern.ch/files/tmva_reg_example.root" ); // if not: download from ROOT server
   
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVARegression           : Using input file: " << input->GetName() << std::endl;

   // --- Register the regression tree

   TTree *regTree = (TTree*)input->Get("TreeR");

   // global event weights per tree (see below for setting event-wise weights)
   Double_t regWeight  = 1.0;   

   // You can add an arbitrary number of regression trees
   factory->AddRegressionTree( regTree, regWeight );

   // This would set individual event weights (the variables defined in the 
   // expression need to exist in the original TTree)
//   factory->SetWeightExpression( "var1", "Regression" );

   // Apply additional cuts on the signal and background samples (can be different)
   TCut mycut = ""; // for example: TCut mycut = "abs(var1)<0.5 && abs(var2-0.5)<1";

   // tell the factory to use all remaining events in the trees after training for testing:
   factory->PrepareTrainingAndTestTree( mycut, 
                                        "nTrain_Regression=0:nTest_Regression=0:SplitMode=Random:NormMode=NumEvents:!V" );

   // If no numbers of events are given, half of the events in the tree are used 
   // for training, and the other half for testing:
   //    factory->PrepareTrainingAndTestTree( mycut, "SplitMode=random:!V" );  

   // ---- Book MVA methods
   // Neural network (MLP)
   factory->BookMethod( TMVA::Types::kMLP, "MLP_4", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+2,4,N+2:TestRate=6:TrainingMethod=BFGS:Sampling=0.4:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=3:!UseRegulator:LearningRate=0.001" );

   factory->BookMethod( TMVA::Types::kMLP, "MLP_3", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+2,3,N+2:TestRate=6:TrainingMethod=BFGS:Sampling=0.4:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=3:!UseRegulator:LearningRate=0.001" );

   factory->BookMethod( TMVA::Types::kMLP, "MLP_2", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+2,2,N+2:TestRate=6:TrainingMethod=BFGS:Sampling=0.4:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=10:!UseRegulator:LearningRate=0.001" );

   factory->BookMethod( TMVA::Types::kMLP, "MLP_1", "!H:!V:VarTransform=Norm:NeuronType=tanh:NCycles=20000:HiddenLayers=N+2,1,N+2:TestRate=6:TrainingMethod=BFGS:Sampling=0.4:SamplingEpoch=0.8:ConvergenceImprove=1e-6:ConvergenceTests=10:!UseRegulator:LearningRate=0.001" );


   // --------------------------------------------------------------------------------------------------

   // ---- Now you can tell the factory to train, test, and evaluate the MVAs

   // Train MVAs using the set of training events
   factory->TrainAllMethods();

   // ---- Evaluate all MVAs using the set of test events
   factory->TestAllMethods();

   // ----- Evaluate and compare performance of all configured MVAs
   factory->EvaluateAllMethods();    

   // --------------------------------------------------------------
   
   // Save the output
   outputFile->Close();

   std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
   std::cout << "==> TMVARegression is done!" << std::endl;      

   delete factory;

   std::cout << std::endl;
   std::cout << "==> Too view the results, launch the GUI: \"root -l TMVARegGui.C\"" << std::endl;
   std::cout << std::endl;
}






void reader () 
{
   // ---------------------------------------------------------------

   std::cout << std::endl;
   std::cout << "==> Start TMVARegressionApplication" << std::endl;


   // --- Create the Reader object

   TMVA::Reader *reader = new TMVA::Reader( "!Color:!Silent" );    

   // Create a set of variables and declare them to the reader
   // - the variable names MUST corresponds in name and type to those given in the weight file(s) used
   Float_t var1, var2, var3, var4;
   reader->AddVariable( "var1", &var1 );
   reader->AddVariable( "var2", &var2 );
   reader->AddVariable( "var3", &var3 );
   reader->AddVariable( "var4", &var4 );


   // --- Book the MVA methods

   TString dir    = "weights/";
   TString prefix = "TMVAAutoencoder";

   TString weightfile = dir + prefix + "_" + "MLP_4" + ".weights.xml";
   TMVA::IMethod* iMlp4 = reader->BookMVA( TString("MLP_4 method"), weightfile ); 
   weightfile = dir + prefix + "_" + "MLP_3" + ".weights.xml";
   TMVA::IMethod* iMlp3 = reader->BookMVA( TString("MLP_3 method"), weightfile ); 
   weightfile = dir + prefix + "_" + "MLP_2" + ".weights.xml";
   TMVA::IMethod* iMlp2 = reader->BookMVA( TString("MLP_2 method"), weightfile ); 
   weightfile = dir + prefix + "_" + "MLP_1" + ".weights.xml";
   TMVA::IMethod* iMlp1 = reader->BookMVA( TString("MLP_1 method"), weightfile ); 
   
   TMVA::MethodMLP* mlp4 = dynamic_cast<TMVA::MethodMLP*>(iMlp4);
   TMVA::MethodMLP* mlp3 = dynamic_cast<TMVA::MethodMLP*>(iMlp3);
   TMVA::MethodMLP* mlp2 = dynamic_cast<TMVA::MethodMLP*>(iMlp2);
   TMVA::MethodMLP* mlp1 = dynamic_cast<TMVA::MethodMLP*>(iMlp1);
   
   TFile *input(0);
   TString fname = "./tmva_reg_example.root";
   if (!gSystem->AccessPathName( fname )) {
      input = TFile::Open( fname ); // check if file in local directory exists
   } 
   else { 
      input = TFile::Open( "http://root.cern.ch/files/tmva_reg_example.root" ); // if not: download from ROOT server
   }
   
   if (!input) {
      std::cout << "ERROR: could not open data file" << std::endl;
      exit(1);
   }
   std::cout << "--- TMVARegressionApp        : Using input file: " << input->GetName() << std::endl;

   // --- Event loop

   // Prepare the tree
   // - here the variable names have to corresponds to your tree
   // - you can use the same variables as above which is slightly faster,
   //   but of course you can use different ones and copy the values inside the event loop
   //
   TTree* theTree = (TTree*)input->Get("TreeR");
   std::cout << "--- Select signal sample" << std::endl;
   theTree->SetBranchAddress( "var1", &var1 );
   theTree->SetBranchAddress( "var2", &var2 );

   
   TFile *target  = new TFile( "TMVAAutoApp.root","RECREATE" );
   TTree* outTree = new TTree( "aenc", "Auto encoder");

   float enc1[1];
   float enc2[2];
   float enc3[3];
   float enc4[4];
   // reduced dimensions
   // enc1[0] --> reduced to 1 node
   // enc2[0...2] --> reduced to 2 nodes
   // enc3[0...3] --> reduced to 3 nodes
   // enc4[0...4] --> reduced to 4 nodes
   outTree->Branch ("enc1", enc1, "enc1[1]/F" );
   outTree->Branch ("enc2", enc2, "enc2[2]/F" );
   outTree->Branch ("enc3", enc3, "enc3[3]/F" );
   outTree->Branch ("enc4", enc4, "enc4[4]/F" );

   // reduced dimensions
   // var1, var2, var3, var4 --> input variables
   outTree->Branch ("var1", &var1, "var1/F" );
   outTree->Branch ("var2", &var2, "var2/F" );
   outTree->Branch ("var3", &var3, "var3/F" );
   outTree->Branch ("var4", &var4, "var4/F" );


   float r1[4];
   float r2[4];
   float r3[4];
   float r4[4];
   // r1, r2, r3, r4 --> target variables which should be as close as possible to the input variables
   // the deviation of r1,2,3,4 from var1,2,3,4 is a measure for the error made by the autoencoder
   // r1[0...4]
   // r2[0...4]
   // r3[0...4]
   // r4[0...4]
   outTree->Branch ("r1", r1, "r1[4]/F" );
   outTree->Branch ("r2", r2, "r2[4]/F" );
   outTree->Branch ("r3", r3, "r3[4]/F" );
   outTree->Branch ("r4", r4, "r4[4]/F" );

   std::cout << "--- Processing: " << theTree->GetEntries() << " events" << std::endl;
   TStopwatch sw;
   sw.Start();
   for (Long64_t ievt=0; ievt<theTree->GetEntries();ievt++) {

      if (ievt%1000 == 0) {
         std::cout << "--- ... Processing event: " << ievt << std::endl;
      }

      theTree->GetEntry(ievt);
      var3 = var1+var2;
      var4 = var2*var1;

      // Retrieve the MVA target values (regression outputs) and fill into histograms
      // NOTE: EvaluateRegression(..) returns a vector for multi-target regression

      // retrieve as well the values of the nodes of the second layer which is the smallest
      // layer of the network
      {
	  const std::vector<Float_t>&  output = (reader->EvaluateRegression( TString("MLP_4 method") ));
	  std::copy (output.begin(), output.end(), r4);
	  mlp4->GetLayerActivation (2, enc4);
      }
      {
	  const std::vector<Float_t>&  output = (reader->EvaluateRegression( TString("MLP_3 method") ));
	  std::copy (output.begin(), output.end(), r3);
	  mlp3->GetLayerActivation (2, enc3);
      }
      {
	  const std::vector<Float_t>&  output = (reader->EvaluateRegression( TString("MLP_2 method") ));
	  std::copy (output.begin(), output.end(), r2);
	  mlp2->GetLayerActivation (2, enc2);
      }
      {
	  const std::vector<Float_t>&  output = (reader->EvaluateRegression( TString("MLP_1 method") ));
	  std::copy (output.begin(), output.end(), r1);
	  mlp1->GetLayerActivation (2, enc1);
      }
      
      outTree->Fill ();
   }
   sw.Stop();
   std::cout << "--- End of event loop: "; sw.Print();

   // --- Write histograms
   TH1F h4("quality4","quality4",100,0,15);
   TH1F h3("quality3","quality3",100,0,15);
   TH1F h2("quality2","quality2",100,0,15);
   TH1F h1("quality1","quality1",100,0,15);
   outTree->Draw ("pow(var1-r4[0],2)+pow(var2-r4[1],2)+pow(var3-r4[2],2)+pow(var4-r4[3],2)>>quality4","","");
   outTree->Draw ("pow(var1-r3[0],2)+pow(var2-r3[1],2)+pow(var3-r3[2],2)+pow(var4-r3[3],2)>>quality3","","");
   outTree->Draw ("pow(var1-r2[0],2)+pow(var2-r2[1],2)+pow(var3-r2[2],2)+pow(var4-r2[3],2)>>quality2","","");
   outTree->Draw ("pow(var1-r1[0],2)+pow(var2-r1[1],2)+pow(var3-r1[2],2)+pow(var4-r1[3],2)>>quality1","","");
   h4.SetLineColor(kBlue);
   h3.SetLineColor(kRed);
   h2.SetLineColor(kGreen);
   h1.SetLineColor(kMagenta);

   outTree->Write ();
   h4.Write();
   h3.Write();
   h2.Write();
   h1.Write();

   std::cout << "--- Created root file: \"" << target->GetName() 
             << "\" containing the MVA output histograms" << std::endl;
  
   delete reader;
    
   std::cout << "==> TMVAAutoencoderApplication is done!" << std::endl << std::endl;



   // reduced dimensions
   // enc1[0] --> reduced to 1 node
   // enc2[0...2] --> reduced to 2 nodes
   // enc3[0...3] --> reduced to 3 nodes
   // enc4[0...4] --> reduced to 4 nodes

   // reduced dimensions
   // var1, var2, var3, var4 --> input variables

   // r1, r2, r3, r4 --> target variables which should be as close as possible to the input variables
   // the deviation of r1,2,3,4 from var1,2,3,4 is a measure for the error made by the autoencoder
   // r1[0...4]
   // r2[0...4]
   // r3[0...4]
   // r4[0...4]

   // if the number of nodes in the smallest layer is sufficient and the training is sufficient
   // then the rX[0] to rX[4] should have the same values as var1 ... var4

}





int main ()
{
    factory ();
    reader ();
}


