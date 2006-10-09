// @(#)root/tmva $Id: MethodTMlpANN.cxx,v 1.28 2006/10/01 16:13:55 andreas.hoecker Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodTMlpANN                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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
const TMultiLayerPerceptron::LearningMethod LearningMethod__= TMultiLayerPerceptron::kStochastic;
// const TMultiLayerPerceptron::LearningMethod LearningMethod__= TMultiLayerPerceptron::kBatch;

ClassImp(TMVA::MethodTMlpANN)

//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir)
   : TMVA::MethodBase(jobName, methodTitle, theData, theOption, theTargetDir  )
   , fMLP(0)
{
   // standard constructor 
   InitTMlpANN();
   
   DeclareOptions();

   ParseOptions();

   ProcessOptions();  
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( DataSet& theData, 
                                    TString theWeightFile,  
                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
   , fMLP(0)
{
   // constructor to calculate the TMlpANN-MVA from previously generatad 
   // weigths (weight file)
   InitTMlpANN();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::InitTMlpANN( void )
{
   // default initialisations
   SetMethodName( "TMlpANN" );
   SetMethodType( TMVA::Types::TMlpANN );
   SetTestvarName();
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::~MethodTMlpANN( void )
{
   // destructor
   if(fMLP!=0) delete fMLP;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::CreateMLPOptions( TString layerSpec )
{
   // translates options from option string into TMlpANN language

   fHiddenLayer = ":";

   while(layerSpec.Length()>0) {
      TString sToAdd="";
      if(layerSpec.First(',')<0) {
         sToAdd = layerSpec;
         layerSpec = "";
      } else {
         sToAdd = layerSpec(0,layerSpec.First(','));
         layerSpec = layerSpec(layerSpec.First(',')+1,layerSpec.Length());
      }
      int nNodes = 0;
      if(sToAdd.BeginsWith("N")) { sToAdd.Remove(0,1); nNodes = GetNvar(); }
      nNodes += atoi(sToAdd);
      fHiddenLayer = Form( "%s%i:", (const char*)fHiddenLayer, nNodes );
   }


   // set input vars
   vector<TString>::iterator itrVar    = (*fInputVars).begin();
   vector<TString>::iterator itrVarEnd = (*fInputVars).end();
   fMLPBuildOptions="";
   for (; itrVar != itrVarEnd; itrVar++) {
      if (EnforceNormalization__) fMLPBuildOptions += "@";
      TString myVar = *itrVar; ;
      fMLPBuildOptions += myVar;
      fMLPBuildOptions += ",";
   }
   fMLPBuildOptions.Chop(); // remove last ","

   // prepare final options for MLP kernel
   fMLPBuildOptions += fHiddenLayer;
   fMLPBuildOptions += "type";

   cout << "--- " << GetName() << ": use " << fNcycles << " training cycles" << endl;
   cout << "--- " << GetName() << ": use configuration (nodes per hidden layer): " 
        << fHiddenLayer << endl;  
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::DeclareOptions() 
{
   DeclareOptionRef(fNcycles=3000,"NCycles","Number of training cycles");
   DeclareOptionRef(fLayerSpec="N-1,N-2","HiddenLayers","Specification of the hidden layers");
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::ProcessOptions() 
{
   CreateMLPOptions(fLayerSpec);

   // Here we create a dummy tree necessary to create 
   // a minimal NN
   // this NN gets recreated before training
   // but can be used for testing (see method GetMvaVal() )
   static Double_t* d = new Double_t[Data().GetNVariables()] ;
   static Int_t   type;

   gROOT->cd();
   TTree * dummyTree = new TTree("dummy","Empty dummy tree", 1);
   for(UInt_t ivar = 0; ivar<Data().GetNVariables(); ivar++) {
      TString vn = Data().GetInternalVarName(ivar);
      dummyTree->Branch(Form("%s",vn.Data()), d+ivar, Form("%s/D",vn.Data()));
   }
   dummyTree->Branch("type", &type, "type/I");

   if(fMLP!=0) delete fMLP;
   fMLP = new TMultiLayerPerceptron( fMLPBuildOptions.Data(), dummyTree );
}

//_______________________________________________________________________
Double_t TMVA::MethodTMlpANN::GetMvaValue()
{
   static Double_t* d = new Double_t[Data().GetNVariables()];
   for(UInt_t ivar = 0; ivar<Data().GetNVariables(); ivar++) {
      d[ivar] = (Double_t)Data().Event().GetVal(ivar);
   }
   Double_t mvaVal = fMLP->Evaluate(0,d);
   return mvaVal;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::Train( void )
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
      cout << "--- " << GetName() << " <verbose>: option string: " << GetOptions() << endl;

   // TMultiLayerPerceptron wants test and training tree at once
   // so merge the training and testing trees from the MVA factory first:

   TTree *localTrainingTree  = Data().GetTrainingTree()->CloneTree();
   localTrainingTree->CopyEntries(GetTestTree());
  
   // These are the event lists for the mlp train method
   // first events in the tree are for training
   // the rest for internal testing...
   TString trainList = "Entry$<";
   trainList += (Int_t)Data().GetNEvtTrain();
   TString testList  = "Entry$>=";
   testList  += (Int_t)Data().GetNEvtTrain();

   // create NN 
   if(fMLP!=0) delete fMLP;
   fMLP = new TMultiLayerPerceptron( fMLPBuildOptions.Data(), 
                                     localTrainingTree,
                                     trainList,
                                     testList );
  
   // set learning method
   fMLP->SetLearningMethod( LearningMethod__ );

   // train NN
   fMLP->Train(fNcycles, "text,update=200");

   // write weights to File;
   // this is not nice, but fMLP gets deleted at the end of Train()
   localTrainingTree->Delete();
}

//_______________________________________________________________________
void  TMVA::MethodTMlpANN::WriteWeightsToStream( ostream & o ) const
{
   // write weights to stream

   // since the MLP can not write to stream and provides no access to its content
   // except through DumpWeights(filename), we 
   // 1st: dump the weights
   fMLP->DumpWeights("weights/TMlp.nn.weights.temp");
   // 2nd: read them back
   ifstream inf("weights/TMlp.nn.weights.temp");
   // 3rd: write them to the stream
   o << inf.rdbuf();
   inf.close();
   // here we can delete the temporary file
   // how?
}
  
//_______________________________________________________________________
void  TMVA::MethodTMlpANN::ReadWeightsFromStream( istream & istr )
{
   // read weights from stream
   // since the MLP can not read from the stream, we
   // 1st: write the weights to temporary file
   ofstream fout("weights/TMlp.nn.weights.temp");
   fout << istr.rdbuf();
   fout.close();
   // 2nd: load the weights from the temporary file into the MLP
   // the MLP is already build
   cout << "Load TMLP weights" << endl;
   fMLP->LoadWeights("weights/TMlp.nn.weights.temp");
   // here we can delete the temporary file
   // how?
}

//_______________________________________________________________________
void  TMVA::MethodTMlpANN::WriteHistosToFile( void ) const
{
   // write special monitoring histograms to file - not implemented for TMlpANN
   cout << "--- " << GetName() << ": write " << GetName() 
        << " special histos to file: " << BaseDir()->GetPath() << endl;
}
