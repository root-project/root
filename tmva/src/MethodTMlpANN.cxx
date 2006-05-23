// @(#)root/tmva $Id: MethodTMlpANN.cxx,v 1.9 2006/05/23 09:53:10 stelzer Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

// @(#)root/tmva $Id: MethodTMlpANN.cxx,v 1.9 2006/05/23 09:53:10 stelzer Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodTMlpANN                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
// Begin_Html
/*
  This is the TMVA TMultiLayerPerceptron interface class. It provides the 
  training and testing the ROOT internal MLP class in the TMVA framework.<be>

  Available learning methods:<br>
  <ul>
  <li>TMultiLayerPerceptron::kStochastic      </li> 
  <li>TMultiLayerPerceptron::kBatch           </li>
  <li>TMultiLayerPerceptron::kSteepestDescent </li>
  <li>TMultiLayerPerceptron::kRibierePolak    </li>
  <li>TMultiLayerPerceptron::kFletcherReeves  </li>
  <li>TMultiLayerPerceptron::kBFGS            </li>
  </ul>
  See the 
  <a href="http://root.cern.ch/root/html/TMultiLayerPerceptron.html>TMultiLayerPerceptron class description</a> 
  for details on this ANN.
*/
// End_Html
//_______________________________________________________________________

#include "TMVA/MethodTMlpANN.h"
#include <stdlib.h>
#include "Riostream.h"
#include "TMultiLayerPerceptron.h"
#include "TLeaf.h"
#include "TEventList.h"
#include "TObjString.h"
#include "TMVA/Tools.h"

// some additional TMlpANN options
const Bool_t EnforceNormalization__=kTRUE;
const TMultiLayerPerceptron::LearningMethod LearningMethod__=TMultiLayerPerceptron::kStochastic;

ClassImp(TMVA::MethodTMlpANN)


//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( TString jobName, std::vector<TString>* theVariables,  
                                    TTree* theTree, TString theOption, TDirectory* theTargetDir)
   : TMVA::MethodBase(jobName, theVariables, theTree, theOption, theTargetDir  )
{
   // standard constructor 
   // option string is interpreted in base class: MethodANNBase

   InitTMlpANN();

   if (fOptions.Sizeof()<2){
      fOptions = "3000:N-1:N-2";
      cout << "--- " << GetName() << ": using default options= "<< fOptions << endl;
   }

   // create full configuration string
   CreateMLPOptions();
  
   cout << "--- " << GetName() << ": use " << fNcycles << " training cycles" << endl;
   cout << "--- " << GetName() << ": use configuration (nodes per hidden layer): " 
        << fHiddenLayer << endl;  
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( vector<TString> *theVariables, 
                                    TString theWeightFile,  
                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
   // constructor to calculate the TMlpANN-MVA from previously generatad 
   // weigths (weight file)
   InitTMlpANN();
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::InitTMlpANN( void )
{
   // default initialisations
   fMethodName = "TMlpANN";
   fMethod     = TMVA::Types::TMlpANN;
   fTestvar    = fTestvarPrefix+GetMethodName();
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::~MethodTMlpANN( void )
{
   // destructor
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::CreateMLPOptions( void )
{
   // translates options from option string into TMlpANN language

   // parse the option string
   vector<Int_t>* nodes = ParseOptionString( fOptions, fNvar, new vector<Int_t>() );
   fNcycles = (*nodes)[0];
   fHiddenLayer = ":";
   for (UInt_t i=1; i<nodes->size(); i++) 
      fHiddenLayer = Form( "%s%i:", (const char*)fHiddenLayer, (*nodes)[i] );

   // set input vars
   vector<TString>::iterator itrVar    = (*fInputVars).begin();
   vector<TString>::iterator itrVarEnd = (*fInputVars).end();
   fOptions="";
   for (; itrVar != itrVarEnd; itrVar++) {
      if (EnforceNormalization__) fOptions += "@";
      TString myVar = *itrVar; ;
      fOptions += myVar;
      fOptions += ",";
   }
   fOptions.Chop(); // remove last ","

   // prepare final options for MLP kernel
   fOptions += fHiddenLayer;
   fOptions += "type";

   delete nodes;
}


//_______________________________________________________________________
void  TMVA::MethodTMlpANN::Train( void )
{
   // performs TMlpANN training
   // available learning methods:
   //
   //       TMultiLayerPerceptron::kStochastic      
   //       TMultiLayerPerceptron::kBatch           
   //       TMultiLayerPerceptron::kSteepestDescent 
   //       TMultiLayerPerceptron::kRibierePolak    
   //       TMultiLayerPerceptron::kFletcherReeves  
   //       TMultiLayerPerceptron::kBFGS            
   //
   if (!CheckSanity()) {
      cout << "--- " << GetName() << ": Error: sanity check failed ==> exit(1)" << endl;
      exit(1);
   }
  
   if (Verbose()) 
      cout << "--- " << GetName() << " <verbose>: option string: " << fOptions << endl;

   // TMultiLayerPerceptron wants test and training tree at once
   // so merge the training and testing trees from the MVA factory first:

   Double_t v[100];
   Int_t type;  
   TTree *localTrainingTree  = new TTree("localTrainingTree","Merged fTraining + fTestTree");
   localTrainingTree->Branch("type",&type,"type/I");
   for(Int_t ivar=0; ivar<fNvar; ivar++) {
      if (!(*fInputVars)[ivar].Contains("type")) {
         localTrainingTree->Branch( (*fInputVars)[ivar], &v[ivar], (*fInputVars)[ivar] + "/D" );
      }
   }

   // loop over training tree and fill local training tree
   for (Int_t ievt=0;ievt<fTrainingTree->GetEntries(); ievt++) {
      type = (Int_t)TMVA::Tools::GetValue( fTrainingTree, ievt, "type" );
      for (Int_t ivar=0; ivar<fNvar; ivar++) {
         if (!(*fInputVars)[ivar].Contains("type")) {
            v[ivar] =  TMVA::Tools::GetValue( fTrainingTree, ievt, (*fInputVars)[ivar] );
         }
      }
      localTrainingTree->Fill();        
   }

   // loop over test tree and fill local training tree
   for (Int_t ievt=0;ievt<fTestTree->GetEntries(); ievt++) {
      type = (Int_t)TMVA::Tools::GetValue( fTestTree, ievt, "type" );
      for(Int_t ivar=0; ivar<fNvar; ivar++) {
         if (!(*fInputVars)[ivar].Contains("type")) {
            v[ivar]= TMVA::Tools::GetValue( fTestTree, ievt, (*fInputVars)[ivar] );
         }
      }
      localTrainingTree->Fill();        
   }
  
   // These are the event lists for the mlp train method
   // first events in the tree are for training
   // the rest for internal testing...
   TString trainList = "Entry$<";
   trainList += (Int_t)fTrainingTree->GetEntries();
   TString testList  = "Entry$>=";
   testList  += (Int_t)fTrainingTree->GetEntries();

   // create NN 
   TMultiLayerPerceptron *mlp = new TMultiLayerPerceptron( fOptions, 
                                                           localTrainingTree,
                                                           trainList,
                                                           testList );
  
   // set learning method
   mlp->SetLearningMethod( LearningMethod__ );

   // train NN
   mlp->Train(fNcycles, "text,update=200");

   // write weights to File;
   // this is not nice, but mlp gets deleted at the end of Train()
   mlp->DumpWeights(GetWeightFileName());
   WriteWeightsToFile();

   localTrainingTree->Delete();
   delete mlp;
}

//_______________________________________________________________________
void  TMVA::MethodTMlpANN::WriteWeightsToFile( void )
{
   // write weights to file
   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": creating weight file: " << fname << endl;

   //assume that the weights are already written
   // this method just adds the net structure to the weights file
   ofstream fout( fname , ios::out | ios::app);
   if (!fout.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
           << "unable to open output  weight file: " << fname << endl;
      exit(1);
   }
   fout << fOptions;
   fout.close();  
}
  
//_______________________________________________________________________
void  TMVA::MethodTMlpANN::ReadWeightsFromFile( void )
{
   // read weights from file
   TString fname = GetWeightFileName();
   cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
   ifstream fin( fname );

   if (!fin.good( )) { // file not found --> Error
      cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
           << "unable to open input file: " << fname << endl;
      exit(1);
   }

   while (!fin.eof()) fin >> fOptions;

   fin.close();  
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::PrepareEvaluationTree( TTree* testTree )
{
   // evaluate method
   
   // A new branch is added to the TestTree which contains the outout of the neural network
   if (Verbose()) cout << "--- " << GetName() << " <verbose>: begin testing" << endl;
    
   Double_t v[100];
   Int_t type;  

   TTree *localTestTree  = new TTree("localTestTree","copy of testTree");
   localTestTree->Branch("type",&type,"type/I",128000);
   for (Int_t ivar=0; ivar<fNvar; ivar++) {
      if (!(*fInputVars)[ivar].Contains("type")) {
         localTestTree->Branch( (*fInputVars)[ivar], &v[ivar], (*fInputVars)[ivar] + "/D",128000 );
      }
   }
  
   // loop over training tree and fill local training tree
   for (Int_t ievt=0;ievt<testTree->GetEntries(); ievt++) {
      type = (Int_t)TMVA::Tools::GetValue( testTree, ievt, "type" );
      for (Int_t ivar=0; ivar<fNvar; ivar++) {
         if (!(*fInputVars)[ivar].Contains("type")) {
            v[ivar] =  TMVA::Tools::GetValue( testTree, ievt, (*fInputVars)[ivar] );
         }
      }
      localTestTree->Fill();        
   }
  
   // get net structure from weights file
   ReadWeightsFromFile();

   // create Net
   TMultiLayerPerceptron *mlp = new TMultiLayerPerceptron( fOptions, localTestTree );
   mlp->LoadWeights(GetWeightFileName());
  
   // add new branch to testTree
   Double_t myMVA;
   TBranch *newBranch = testTree->Branch( fTestvar, &myMVA, fTestvar + "/D" );
   
   // loop over testTree
   for (Int_t i=0; i< (testTree->GetEntries()) ; i++) {
      myMVA=mlp->Result(i);
      newBranch->Fill();
   }
   
   localTestTree->Delete();
   delete mlp;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::SetTestTree( TTree* testTree )
{
   // set the test tree
   fTestTree = testTree;
}

//_______________________________________________________________________
void  TMVA::MethodTMlpANN::WriteHistosToFile( void )
{
   // write special monitoring histograms to file - not implemented for TMlpANN
   cout << "--- " << GetName() << ": write " << GetName() 
        << " special histos to file: " << fBaseDir->GetPath() << endl;
}
