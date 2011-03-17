// @(#)root/tmva $Id$
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
/* Begin_Html

  This is the TMVA TMultiLayerPerceptron interface class. It provides the
  training and testing the ROOT internal MLP class in the TMVA framework.<be>

  Available learning methods:<br>
  <ul>
  <li>Stochastic      </li>
  <li>Batch           </li>
  <li>SteepestDescent </li>
  <li>RibierePolak    </li>
  <li>FletcherReeves  </li>
  <li>BFGS            </li>
  </ul>
End_Html */
//
//  See the TMultiLayerPerceptron class description
//  for details on this ANN.
//
//_______________________________________________________________________

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "Riostream.h"
#include "TLeaf.h"
#include "TEventList.h"
#include "TObjString.h"
#include "TROOT.h"
#include "TMultiLayerPerceptron.h"

#include "TMVA/Config.h"
#include "TMVA/MethodTMlpANN.h"

#include "TMVA/ClassifierFactory.h"
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif

// some additional TMlpANN options
const Bool_t EnforceNormalization__=kTRUE;
#if ROOT_VERSION_CODE > ROOT_VERSION(5,13,06)
const TMultiLayerPerceptron::ELearningMethod LearningMethod__= TMultiLayerPerceptron::kStochastic;
// const TMultiLayerPerceptron::ELearningMethod LearningMethod__= TMultiLayerPerceptron::kBatch;
#else
const TMultiLayerPerceptron::LearningMethod LearningMethod__= TMultiLayerPerceptron::kStochastic;
#endif

REGISTER_METHOD(TMlpANN)

ClassImp(TMVA::MethodTMlpANN)

//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( const TString& jobName,
                                    const TString& methodTitle,
                                    DataSetInfo& theData,
                                    const TString& theOption,
                                    TDirectory* theTargetDir) :
   TMVA::MethodBase( jobName, Types::kTMlpANN, methodTitle, theData, theOption, theTargetDir ),
   fMLP(0),
   fNcycles(100),
   fValidationFraction(0.5),
   fLearningMethod( "" )
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::MethodTMlpANN( DataSetInfo& theData,
                                    const TString& theWeightFile,
                                    TDirectory* theTargetDir ) :
   TMVA::MethodBase( Types::kTMlpANN, theData, theWeightFile, theTargetDir ),
   fMLP(0),
   fNcycles(100),
   fValidationFraction(0.5),
   fLearningMethod( "" )
{
   // constructor from weight file
}

//_______________________________________________________________________
Bool_t TMVA::MethodTMlpANN::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses,
                                             UInt_t /*numberTargets*/ )
{
   // TMlpANN can handle classification with 2 classes
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodTMlpANN::Init( void )
{
   // default initialisations
}

//_______________________________________________________________________
TMVA::MethodTMlpANN::~MethodTMlpANN( void )
{
   // destructor 
   if (fMLP) delete fMLP;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::CreateMLPOptions( TString layerSpec )
{
   // translates options from option string into TMlpANN language

   fHiddenLayer = ":";

   while (layerSpec.Length()>0) {
      TString sToAdd="";
      if (layerSpec.First(',')<0) {
         sToAdd = layerSpec;
         layerSpec = "";
      }
      else {
         sToAdd = layerSpec(0,layerSpec.First(','));
         layerSpec = layerSpec(layerSpec.First(',')+1,layerSpec.Length());
      }
      int nNodes = 0;
      if (sToAdd.BeginsWith("N")) { sToAdd.Remove(0,1); nNodes = GetNvar(); }
      nNodes += atoi(sToAdd);
      fHiddenLayer = Form( "%s%i:", (const char*)fHiddenLayer, nNodes );
   }

   // set input vars
   std::vector<TString>::iterator itrVar    = (*fInputVars).begin();
   std::vector<TString>::iterator itrVarEnd = (*fInputVars).end();
   fMLPBuildOptions = "";
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

   Log() << kINFO << "Use " << fNcycles << " training cycles" << Endl;
   Log() << kINFO << "Use configuration (nodes per hidden layer): " << fHiddenLayer << Endl;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // know options:
   // NCycles       <integer>    Number of training cycles (too many cycles could overtrain the network)
   // HiddenLayers  <string>     Layout of the hidden layers (nodes per layer)
   //   * specifiactions for each hidden layer are separated by commata
   //   * for each layer the number of nodes can be either absolut (simply a number)
   //        or relative to the number of input nodes to the neural net (N)
   //   * there is always a single node in the output layer
   //   example: a net with 6 input nodes and "Hiddenlayers=N-1,N-2" has 6,5,4,1 nodes in the
   //   layers 1,2,3,4, repectively
   DeclareOptionRef( fNcycles    = 200,       "NCycles",      "Number of training cycles" );
   DeclareOptionRef( fLayerSpec  = "N,N-1",   "HiddenLayers", "Specification of hidden layer architecture (N stands for number of variables; any integers may also be used)" );

   DeclareOptionRef( fValidationFraction = 0.5, "ValidationFraction",
                     "Fraction of events in training tree used for cross validation" );

   DeclareOptionRef( fLearningMethod = "Stochastic", "LearningMethod", "Learning method" );
   AddPreDefVal( TString("Stochastic") );
   AddPreDefVal( TString("Batch") );
   AddPreDefVal( TString("SteepestDescent") );
   AddPreDefVal( TString("RibierePolak") );
   AddPreDefVal( TString("FletcherReeves") );
   AddPreDefVal( TString("BFGS") );
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::ProcessOptions()
{
   // builds the neural network as specified by the user
   CreateMLPOptions(fLayerSpec);

   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kFATAL << "Mechanism to ignore events with negative weights in training not available for method"
            << GetMethodTypeName()
            << " --> please remove \"IgnoreNegWeightsInTraining\" option from booking string."
            << Endl;
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodTMlpANN::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // calculate the value of the neural net for the current event
   const Event* ev = GetEvent();
   static Double_t* d = new Double_t[Data()->GetNVariables()];
   for (UInt_t ivar = 0; ivar<Data()->GetNVariables(); ivar++) {
      d[ivar] = (Double_t)ev->GetValue(ivar);
   }
   Double_t mvaVal = fMLP->Evaluate(0,d);

   // cannot determine error
   NoErrorCalc(err, errUpper);

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
   // TMultiLayerPerceptron wants test and training tree at once
   // so merge the training and testing trees from the MVA factory first:

   Int_t type;
   Float_t weight;
   const Long_t basketsize = 128000;
   Float_t* vArr = new Float_t[GetNvar()]; 

   TTree *localTrainingTree = new TTree( "TMLPtrain", "Local training tree for TMlpANN" );
   localTrainingTree->Branch( "type",       &type,        "type/I",        basketsize );
   localTrainingTree->Branch( "weight",     &weight,      "weight/F",      basketsize );
   
   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {
      const char* myVar = GetInternalVarName(ivar).Data();
      localTrainingTree->Branch( myVar, &vArr[ivar], Form("Var%02i/F", ivar), basketsize );
   }
   
   for (UInt_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {
      const Event *ev = GetEvent(ievt);
      for (UInt_t i=0; i<GetNvar(); i++) {
         vArr[i] = ev->GetValue( i );
      }
      type   = DataInfo().IsSignal( ev ) ? 1 : 0;
      weight = ev->GetWeight();
      localTrainingTree->Fill();
   }

   // These are the event lists for the mlp train method
   // first events in the tree are for training
   // the rest for internal testing (cross validation)...
   // NOTE: the training events are ordered: first part is signal, second part background
   TString trainList = "Entry$<";
   trainList += 1.0-fValidationFraction;
   trainList += "*";
   trainList += (Int_t)Data()->GetNEvtSigTrain();
   trainList += " || (Entry$>";
   trainList += (Int_t)Data()->GetNEvtSigTrain();
   trainList += " && Entry$<";
   trainList += (Int_t)(Data()->GetNEvtSigTrain() + (1.0 - fValidationFraction)*Data()->GetNEvtBkgdTrain());
   trainList += ")";
   TString testList  = TString("!(") + trainList + ")";

   // print the requirements
   Log() << kINFO << "Requirement for training   events: \"" << trainList << "\"" << Endl;
   Log() << kINFO << "Requirement for validation events: \"" << testList << "\"" << Endl;

   // localTrainingTree->Print();

   // create NN
   if (fMLP != 0) { delete fMLP; fMLP = 0; }
   fMLP = new TMultiLayerPerceptron( fMLPBuildOptions.Data(),
                                     localTrainingTree,
                                     trainList,
                                     testList );
   fMLP->SetEventWeight( "weight" );

   // set learning method
#if ROOT_VERSION_CODE > ROOT_VERSION(5,13,06)
   TMultiLayerPerceptron::ELearningMethod learningMethod = TMultiLayerPerceptron::kStochastic;
#else
   TMultiLayerPerceptron::LearningMethod  learningMethod = TMultiLayerPerceptron::kStochastic;
#endif

   fLearningMethod.ToLower();
   if      (fLearningMethod == "stochastic"      ) learningMethod = TMultiLayerPerceptron::kStochastic;
   else if (fLearningMethod == "batch"           ) learningMethod = TMultiLayerPerceptron::kBatch;
   else if (fLearningMethod == "steepestdescent" ) learningMethod = TMultiLayerPerceptron::kSteepestDescent;
   else if (fLearningMethod == "ribierepolak"    ) learningMethod = TMultiLayerPerceptron::kRibierePolak;
   else if (fLearningMethod == "fletcherreeves"  ) learningMethod = TMultiLayerPerceptron::kFletcherReeves;
   else if (fLearningMethod == "bfgs"            ) learningMethod = TMultiLayerPerceptron::kBFGS;
   else {
      Log() << kFATAL << "Unknown Learning Method: \"" << fLearningMethod << "\"" << Endl;
   }
   fMLP->SetLearningMethod( learningMethod );

   // train NN
   fMLP->Train(fNcycles, "text,update=50" );

   // write weights to File;
   // this is not nice, but fMLP gets deleted at the end of Train()
   delete localTrainingTree;
   delete [] vArr;
}


//_______________________________________________________________________
void TMVA::MethodTMlpANN::AddWeightsXMLTo( void* parent ) const
{
   // write weights to xml file

   // first the architecture
   void *wght = gTools().AddChild(parent, "Weights");
   void* arch = gTools().AddChild( wght, "Architecture" );
   gTools().AddAttr( arch, "BuildOptions", fMLPBuildOptions.Data() );

   // dump weights first in temporary txt file, read from there into xml
   fMLP->DumpWeights( "weights/TMlp.nn.weights.temp" );
   std::ifstream inf( "weights/TMlp.nn.weights.temp" );
   char temp[256];
   TString data("");
   void *ch=NULL;
   while (inf.getline(temp,256)) {
      TString dummy(temp);
      //std::cout << dummy << std::endl; // remove annoying debug printout with std::cout
      if (dummy.BeginsWith('#')) {
         if (ch!=0) gTools().AddRawLine( ch, data.Data() );
         dummy = dummy.Strip(TString::kLeading, '#');
         dummy = dummy(0,dummy.First(' '));
         ch = gTools().AddChild(wght, dummy);
         data.Resize(0);
         continue;
      }
      data += (dummy + " ");
   }
   if (ch != 0) gTools().AddRawLine( ch, data.Data() );

   inf.close();
}

//_______________________________________________________________________
void  TMVA::MethodTMlpANN::ReadWeightsFromXML( void* wghtnode )
{
   // rebuild temporary textfile from xml weightfile and load this
   // file into MLP
   void* ch = gTools().GetChild(wghtnode);
   gTools().ReadAttr( ch, "BuildOptions", fMLPBuildOptions );

   ch = gTools().GetNextChild(ch);
   const char* fname = "weights/TMlp.nn.weights.temp";
   std::ofstream fout( fname );
   double temp1=0,temp2=0;
   while (ch) {
      const char* nodecontent = gTools().GetContent(ch);
      std::stringstream content(nodecontent);
      if (strcmp(gTools().GetName(ch),"input")==0) {
         fout << "#input normalization" << std::endl;
         while ((content >> temp1) &&(content >> temp2)) {
            fout << temp1 << " " << temp2 << std::endl;
         }
      }
      if (strcmp(gTools().GetName(ch),"output")==0) {
         fout << "#output normalization" << std::endl;
         while ((content >> temp1) &&(content >> temp2)) {
            fout << temp1 << " " << temp2 << std::endl;
         }
      }
      if (strcmp(gTools().GetName(ch),"neurons")==0) {
         fout << "#neurons weights" << std::endl;         
         while (content >> temp1) {
            fout << temp1 << std::endl;
         }
      }
      if (strcmp(gTools().GetName(ch),"synapses")==0) {
         fout << "#synapses weights" ;         
         while (content >> temp1) {
            fout << std::endl << temp1 ;                
         }
      }
      ch = gTools().GetNextChild(ch);
   }
   fout.close();;

   // Here we create a dummy tree necessary to create a minimal NN
   // to be used for testing, evaluation and application
   static Double_t* d = new Double_t[Data()->GetNVariables()] ;
   static Int_t type;

   gROOT->cd();
   TTree * dummyTree = new TTree("dummy","Empty dummy tree", 1);
   for (UInt_t ivar = 0; ivar<Data()->GetNVariables(); ivar++) {
      TString vn = DataInfo().GetVariableInfo(ivar).GetInternalName();
      dummyTree->Branch(Form("%s",vn.Data()), d+ivar, Form("%s/D",vn.Data()));
   }
   dummyTree->Branch("type", &type, "type/I");

   if (fMLP != 0) { delete fMLP; fMLP = 0; }
   fMLP = new TMultiLayerPerceptron( fMLPBuildOptions.Data(), dummyTree );
   fMLP->LoadWeights( fname );
}
 
//_______________________________________________________________________
void  TMVA::MethodTMlpANN::ReadWeightsFromStream( istream& istr )
{
   // read weights from stream
   // since the MLP can not read from the stream, we
   // 1st: write the weights to temporary file
   std::ofstream fout( "./TMlp.nn.weights.temp" );
   fout << istr.rdbuf();
   fout.close();
   // 2nd: load the weights from the temporary file into the MLP
   // the MLP is already build
   Log() << kINFO << "Load TMLP weights into " << fMLP << Endl;

   Double_t* d = new Double_t[Data()->GetNVariables()] ; 
   static Int_t type;
   gROOT->cd();
   TTree * dummyTree = new TTree("dummy","Empty dummy tree", 1);
   for (UInt_t ivar = 0; ivar<Data()->GetNVariables(); ivar++) {
      TString vn = DataInfo().GetVariableInfo(ivar).GetLabel();
      dummyTree->Branch(Form("%s",vn.Data()), d+ivar, Form("%s/D",vn.Data()));
   }
   dummyTree->Branch("type", &type, "type/I");

   if (fMLP != 0) { delete fMLP; fMLP = 0; }
   fMLP = new TMultiLayerPerceptron( fMLPBuildOptions.Data(), dummyTree );

   fMLP->LoadWeights( "./TMlp.nn.weights.temp" );
   // here we can delete the temporary file
   // how?
   delete [] d;
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::MakeClass( const TString& theClassFileName ) const
{
   // create reader class for classifier -> overwrites base class function
   // create specific class for TMultiLayerPerceptron

   // the default consists of
   TString classFileName = "";
   if (theClassFileName == "")
      classFileName = GetWeightFileDir() + "/" + GetJobName() + "_" + GetMethodName() + ".class";
   else
      classFileName = theClassFileName;

   classFileName.ReplaceAll(".class","");
   Log() << kINFO << "Creating specific (TMultiLayerPerceptron) standalone response class: " << classFileName << Endl;
   fMLP->Export( classFileName.Data() );
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::MakeClassSpecific( std::ostream& /*fout*/, const TString& /*className*/ ) const
{
   // write specific classifier response
   // nothing to do here - all taken care of by TMultiLayerPerceptron
}

//_______________________________________________________________________
void TMVA::MethodTMlpANN::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "This feed-forward multilayer perceptron neural network is the " << Endl;
   Log() << "standard implementation distributed with ROOT (class TMultiLayerPerceptron)." << Endl;
   Log() << Endl;
   Log() << "Detailed information is available here:" << Endl;
   if (gConfig().WriteOptionsReference()) {
      Log() << "<a href=\"http://root.cern.ch/root/html/TMultiLayerPerceptron.html\">";
      Log() << "http://root.cern.ch/root/html/TMultiLayerPerceptron.html</a>" << Endl;
   }
   else Log() << "http://root.cern.ch/root/html/TMultiLayerPerceptron.html" << Endl;
   Log() << Endl;
}
