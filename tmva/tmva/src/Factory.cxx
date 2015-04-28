// @(#)Root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <stelzer@cern.ch>        - DESY, Germany                  *
 *      Peter Speckmayer <peter.speckmayer@cern.ch> - CERN, Switzerland           *
 *      Jan Therhaag          <Jan.Therhaag@cern.ch>   - U of Bonn, Germany       *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// This is the main MVA steering class: it creates all MVA methods,
// and guides them through the training, testing and evaluation
// phases
//_______________________________________________________________________


#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TEventList.h"
#include "TH2.h"
#include "TText.h"
#include "TStyle.h"
#include "TMatrixF.h"
#include "TMatrixDSym.h"
#include "TPaletteAxis.h"
#include "TPrincipal.h"
#include "TMath.h"
#include "TObjString.h"

#include "TMVA/Factory.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/DataSet.h"
#include "TMVA/IMethod.h"
#include "TMVA/MethodBase.h"
#include "TMVA/DataInputHandler.h"
#include "TMVA/DataSetManager.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/MethodBoost.h"
#include "TMVA/MethodCategory.h"

#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"

#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsRegression.h"
#include "TMVA/ResultsMulticlass.h"

const Int_t  MinNoTrainingEvents = 10;
//const Int_t  MinNoTestEvents     = 1;
TFile* TMVA::Factory::fgTargetFile = 0;

ClassImp(TMVA::Factory)

#define RECREATE_METHODS kTRUE
#define READXML          kTRUE

//_______________________________________________________________________
TMVA::Factory::Factory( TString jobName, TFile* theTargetFile, TString theOption )
: Configurable          ( theOption ),
   fDataSetManager       ( NULL ), //DSMTEST
   fDataInputHandler     ( new DataInputHandler ),
   fTransformations      ( "I" ),
   fVerbose              ( kFALSE ),
   fJobName              ( jobName ),
   fDataAssignType       ( kAssignEvents ),
   fATreeEvent           ( NULL ),
   fAnalysisType         ( Types::kClassification )
{
   // standard constructor
   //   jobname       : this name will appear in all weight file names produced by the MVAs
   //   theTargetFile : output ROOT file; the test tree and all evaluation plots
   //                   will be stored here
   //   theOption     : option string; currently: "V" for verbose

   fgTargetFile = theTargetFile;

   //   DataSetManager::CreateInstance(*fDataInputHandler); // DSMTEST removed
   fDataSetManager = new DataSetManager( *fDataInputHandler ); // DSMTEST


   // render silent
   if (gTools().CheckForSilentOption( GetOptions() )) Log().InhibitOutput(); // make sure is silent if wanted to


   // init configurable
   SetConfigDescription( "Configuration options for Factory running" );
   SetConfigName( GetName() );

   // histograms are not automatically associated with the current
   // directory and hence don't go out of scope when closing the file
   // TH1::AddDirectory(kFALSE);
   Bool_t silent          = kFALSE;
#ifdef WIN32
   // under Windows, switch progress bar and color off by default, as the typical windows shell doesn't handle these (would need different sequences..)
   Bool_t color           = kFALSE;
   Bool_t drawProgressBar = kFALSE;
#else
   Bool_t color           = !gROOT->IsBatch();
   Bool_t drawProgressBar = kTRUE;
#endif
   DeclareOptionRef( fVerbose, "V", "Verbose flag" );
   DeclareOptionRef( color,    "Color", "Flag for coloured screen output (default: True, if in batch mode: False)" );
   DeclareOptionRef( fTransformations, "Transformations", "List of transformations to test; formatting example: \"Transformations=I;D;P;U;G,D\", for identity, decorrelation, PCA, Uniform and Gaussianisation followed by decorrelation transformations" );
   DeclareOptionRef( silent,   "Silent", "Batch mode: boolean silent flag inhibiting any output from TMVA after the creation of the factory class object (default: False)" );
   DeclareOptionRef( drawProgressBar,
                     "DrawProgressBar", "Draw progress bar to display training, testing and evaluation schedule (default: True)" );

   TString analysisType("Auto");
   DeclareOptionRef( analysisType,
                     "AnalysisType", "Set the analysis type (Classification, Regression, Multiclass, Auto) (default: Auto)" );
   AddPreDefVal(TString("Classification"));
   AddPreDefVal(TString("Regression"));
   AddPreDefVal(TString("Multiclass"));
   AddPreDefVal(TString("Auto"));

   ParseOptions();
   CheckForUnusedOptions();

   if (Verbose()) Log().SetMinType( kVERBOSE );

   // global settings
   gConfig().SetUseColor( color );
   gConfig().SetSilent( silent );
   gConfig().SetDrawProgressBar( drawProgressBar );

   analysisType.ToLower();
   if     ( analysisType == "classification" ) fAnalysisType = Types::kClassification;
   else if( analysisType == "regression" )     fAnalysisType = Types::kRegression;
   else if( analysisType == "multiclass" )     fAnalysisType = Types::kMulticlass;
   else if( analysisType == "auto" )           fAnalysisType = Types::kNoAnalysisType;

   Greetings();
}

//_______________________________________________________________________
void TMVA::Factory::Greetings()
{
   // print welcome message
   // options are: kLogoWelcomeMsg, kIsometricWelcomeMsg, kLeanWelcomeMsg

   gTools().ROOTVersionMessage( Log() );
   gTools().TMVAWelcomeMessage( Log(), gTools().kLogoWelcomeMsg );
   gTools().TMVAVersionMessage( Log() ); Log() << Endl;
}

//_______________________________________________________________________
TMVA::Factory::~Factory( void )
{
   // destructor
   //   delete fATreeEvent;

   std::vector<TMVA::VariableTransformBase*>::iterator trfIt = fDefaultTrfs.begin();
   for (;trfIt != fDefaultTrfs.end(); trfIt++) delete (*trfIt);

   this->DeleteAllMethods();
   delete fDataInputHandler;

   // destroy singletons
   //   DataSetManager::DestroyInstance(); // DSMTEST replaced by following line
   delete fDataSetManager; // DSMTEST

   // problem with call of REGISTER_METHOD macro ...
   //   ClassifierFactory::DestroyInstance();
   //   Types::DestroyInstance();
   Tools::DestroyInstance();
   Config::DestroyInstance();
}

//_______________________________________________________________________
void TMVA::Factory::DeleteAllMethods( void )
{
   // delete methods
   MVector::iterator itrMethod = fMethods.begin();
   for (; itrMethod != fMethods.end(); itrMethod++) {
      Log() << kDEBUG << "Delete method: " << (*itrMethod)->GetName() << Endl;
      delete (*itrMethod);
   }
   fMethods.clear();
}

//_______________________________________________________________________
void TMVA::Factory::SetVerbose( Bool_t v )
{
   fVerbose = v;
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::Factory::AddDataSet( DataSetInfo &dsi )
{
   return fDataSetManager->AddDataSetInfo(dsi); // DSMTEST
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::Factory::AddDataSet( const TString& dsiName )
{
   DataSetInfo* dsi = fDataSetManager->GetDataSetInfo(dsiName); // DSMTEST

   if (dsi!=0) return *dsi;
   
   return fDataSetManager->AddDataSetInfo(*(new DataSetInfo(dsiName))); // DSMTEST
}

// ________________________________________________
// the next functions are to assign events directly 

//_______________________________________________________________________
TTree* TMVA::Factory::CreateEventAssignTrees( const TString& name )
{
   // create the data assignment tree (for event-wise data assignment by user)
   TTree * assignTree = new TTree( name, name );
   assignTree->SetDirectory(0);
   assignTree->Branch( "type",   &fATreeType,   "ATreeType/I" );
   assignTree->Branch( "weight", &fATreeWeight, "ATreeWeight/F" );

   std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();
   std::vector<VariableInfo>& tgts = DefaultDataSetInfo().GetTargetInfos();
   std::vector<VariableInfo>& spec = DefaultDataSetInfo().GetSpectatorInfos();

   if (!fATreeEvent) fATreeEvent = new Float_t[vars.size()+tgts.size()+spec.size()];
   // add variables
   for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
      TString vname = vars[ivar].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[ivar]), vname + "/F" );
   }
   // add targets
   for (UInt_t itgt=0; itgt<tgts.size(); itgt++) {
      TString vname = tgts[itgt].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[vars.size()+itgt]), vname + "/F" );
   }
   // add spectators
   for (UInt_t ispc=0; ispc<spec.size(); ispc++) {
      TString vname = spec[ispc].GetExpression();
      assignTree->Branch( vname, &(fATreeEvent[vars.size()+tgts.size()+ispc]), vname + "/F" );
   }
   return assignTree;
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Signal", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal testing event
   AddEvent( "Signal", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddTrainingEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( className, Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddTestEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal test event
   AddEvent( className, Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddEvent( const TString& className, Types::ETreeType tt,
                              const std::vector<Double_t>& event, Double_t weight ) 
{
   // add event
   // vector event : the order of values is: variables + targets + spectators
   ClassInfo* theClass = DefaultDataSetInfo().AddClass(className); // returns class (creates it if necessary)
   UInt_t clIndex = theClass->GetNumber();


   // set analysistype to "kMulticlass" if more than two classes and analysistype == kNoAnalysisType
   if( fAnalysisType == Types::kNoAnalysisType && DefaultDataSetInfo().GetNClasses() > 2 )
      fAnalysisType = Types::kMulticlass;

   
   if (clIndex>=fTrainAssignTree.size()) {
      fTrainAssignTree.resize(clIndex+1, 0);
      fTestAssignTree.resize(clIndex+1, 0);
   }

   if (fTrainAssignTree[clIndex]==0) { // does not exist yet
      fTrainAssignTree[clIndex] = CreateEventAssignTrees( Form("TrainAssignTree_%s", className.Data()) );
      fTestAssignTree[clIndex]  = CreateEventAssignTrees( Form("TestAssignTree_%s",  className.Data()) );
   }
   
   fATreeType   = clIndex;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];

   if(tt==Types::kTraining) fTrainAssignTree[clIndex]->Fill();
   else                     fTestAssignTree[clIndex]->Fill();

}

//_______________________________________________________________________
Bool_t TMVA::Factory::UserAssignEvents(UInt_t clIndex) 
{
   // 
   return fTrainAssignTree[clIndex]!=0;
}

//_______________________________________________________________________
void TMVA::Factory::SetInputTreesFromEventAssignTrees()
{
   // assign event-wise local trees to data set
   UInt_t size = fTrainAssignTree.size();
   for(UInt_t i=0; i<size; i++) {
      if(!UserAssignEvents(i)) continue;
      const TString& className = DefaultDataSetInfo().GetClassInfo(i)->GetName();
      SetWeightExpression( "weight", className );
      AddTree(fTrainAssignTree[i], className, 1.0, TCut(""), Types::kTraining );
      AddTree(fTestAssignTree[i], className, 1.0, TCut(""), Types::kTesting );
   }
}

//_______________________________________________________________________
void TMVA::Factory::AddTree( TTree* tree, const TString& className, Double_t weight, 
                             const TCut& cut, const TString& treetype )
{
   // number of signal events (used to compute significance)
   Types::ETreeType tt = Types::kMaxTreeType;
   TString tmpTreeType = treetype; tmpTreeType.ToLower();
   if      (tmpTreeType.Contains( "train" ) && tmpTreeType.Contains( "test" )) tt = Types::kMaxTreeType;
   else if (tmpTreeType.Contains( "train" ))                                   tt = Types::kTraining;
   else if (tmpTreeType.Contains( "test" ))                                    tt = Types::kTesting;
   else {
      Log() << kFATAL << "<AddTree> cannot interpret tree type: \"" << treetype 
            << "\" should be \"Training\" or \"Test\" or \"Training and Testing\"" << Endl;
   }
   AddTree( tree, className, weight, cut, tt );
}

//_______________________________________________________________________
void TMVA::Factory::AddTree( TTree* tree, const TString& className, Double_t weight, 
                             const TCut& cut, Types::ETreeType tt )
{
   if(!tree)
      Log() << kFATAL << "Tree does not exist (empty pointer)." << Endl;

   DefaultDataSetInfo().AddClass( className );

   // set analysistype to "kMulticlass" if more than two classes and analysistype == kNoAnalysisType
   if( fAnalysisType == Types::kNoAnalysisType && DefaultDataSetInfo().GetNClasses() > 2 )
      fAnalysisType = Types::kMulticlass;

   Log() << kINFO << "Add Tree " << tree->GetName() << " of type " << className 
         << " with " << tree->GetEntries() << " events" << Endl;
   DataInput().AddTree( tree, className, weight, cut, tt );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTree( TString datFileS, Double_t weight, Types::ETreeType treetype )
{
   // add signal tree from text file

   // create trees from these ascii files
   TTree* signalTree = new TTree( "TreeS", "Tree (S)" );
   signalTree->ReadFile( datFileS );
 
   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Signal file    : \""
         << datFileS << Endl;
  
   // number of signal events (used to compute significance)
   AddTree( signalTree, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Background", weight, TCut(""), treetype );
}
//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTree( TString datFileB, Double_t weight, Types::ETreeType treetype )
{
   // add background tree from text file

   // create trees from these ascii files
   TTree* bkgTree = new TTree( "TreeB", "Tree (B)" );
   bkgTree->ReadFile( datFileB );
 
   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Background file    : \""
         << datFileB << Endl;
  
   // number of signal events (used to compute significance)
   AddTree( bkgTree, "Background", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Background", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::Factory::SetSignalTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Signal", weight );
}

//_______________________________________________________________________
void TMVA::Factory::SetBackgroundTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Background", weight );
}

//_______________________________________________________________________
void TMVA::Factory::SetTree( TTree* tree, const TString& className, Double_t weight )
{
   // set background tree
   AddTree( tree, className, weight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void  TMVA::Factory::SetInputTrees( TTree* signal, TTree* background, 
                                    Double_t signalWeight, Double_t backgroundWeight )
{
   // define the input trees for signal and background; no cuts are applied
   AddTree( signal,     "Signal",     signalWeight,     TCut(""), Types::kMaxTreeType );
   AddTree( background, "Background", backgroundWeight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::Factory::SetInputTrees( const TString& datFileS, const TString& datFileB, 
                                   Double_t signalWeight, Double_t backgroundWeight )
{
   DataInput().AddTree( datFileS, "Signal", signalWeight );
   DataInput().AddTree( datFileB, "Background", backgroundWeight );
}

//_______________________________________________________________________
void TMVA::Factory::SetInputTrees( TTree* inputTree, const TCut& SigCut, const TCut& BgCut )
{
   // define the input trees for signal and background from single input tree,
   // containing both signal and background events distinguished by the type 
   // identifiers: SigCut and BgCut
   AddTree( inputTree, "Signal",     1.0, SigCut, Types::kMaxTreeType );
   AddTree( inputTree, "Background", 1.0, BgCut , Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::Factory::AddVariable( const TString& expression, const TString& title, const TString& unit, 
                                 char type, Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, title, unit, min, max, type ); 
}

//_______________________________________________________________________
void TMVA::Factory::AddVariable( const TString& expression, char type,
                                 Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, "", "", min, max, type ); 
}

//_______________________________________________________________________
void TMVA::Factory::AddTarget( const TString& expression, const TString& title, const TString& unit, 
                               Double_t min, Double_t max )
{
   // user inserts target in data set info

   if( fAnalysisType == Types::kNoAnalysisType )
      fAnalysisType = Types::kRegression;

   DefaultDataSetInfo().AddTarget( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
void TMVA::Factory::AddSpectator( const TString& expression, const TString& title, const TString& unit, 
                                  Double_t min, Double_t max )
{
   // user inserts target in data set info
   DefaultDataSetInfo().AddSpectator( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::Factory::DefaultDataSetInfo() 
{ 
   // default creation
   return AddDataSet( "Default" );
}

//_______________________________________________________________________
void TMVA::Factory::SetInputVariables( std::vector<TString>* theVariables ) 
{ 
   // fill input variables in data set
   for (std::vector<TString>::iterator it=theVariables->begin();
        it!=theVariables->end(); it++) AddVariable(*it);
}

//_______________________________________________________________________
void TMVA::Factory::SetSignalWeightExpression( const TString& variable)  
{ 
   DefaultDataSetInfo().SetWeightExpression(variable, "Signal"); 
}

//_______________________________________________________________________
void TMVA::Factory::SetBackgroundWeightExpression( const TString& variable) 
{
   DefaultDataSetInfo().SetWeightExpression(variable, "Background");
}

//_______________________________________________________________________
void TMVA::Factory::SetWeightExpression( const TString& variable, const TString& className )  
{
   //Log() << kWarning << DefaultDataSetInfo().GetNClasses() /*fClasses.size()*/ << Endl;
   if (className=="") {
      SetSignalWeightExpression(variable);
      SetBackgroundWeightExpression(variable);
   } 
   else  DefaultDataSetInfo().SetWeightExpression( variable, className );
}

//_______________________________________________________________________
void TMVA::Factory::SetCut( const TString& cut, const TString& className ) {
   SetCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::Factory::SetCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().SetCut( cut, className );
}

//_______________________________________________________________________
void TMVA::Factory::AddCut( const TString& cut, const TString& className ) 
{
   AddCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::Factory::AddCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().AddCut( cut, className );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, 
                                                Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest,
                                                const TString& otherOpt )
{
   // prepare the training and test trees
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:%s", 
                                              NsigTrain, NbkgTrain, NsigTest, NbkgTest, otherOpt.Data()) );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, Int_t Ntrain, Int_t Ntest )
{
   // prepare the training and test trees 
   // kept for backward compatibility
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:SplitMode=Random:EqualTrainSample:!V", 
                                              Ntrain, Ntrain, Ntest, Ntest) );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, const TString& opt )
{
   // prepare the training and test trees
   // -> same cuts for signal and background
   SetInputTreesFromEventAssignTrees();

   DefaultDataSetInfo().PrintClasses();
   AddCut( cut );
   DefaultDataSetInfo().SetSplitOptions( opt );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( TCut sigcut, TCut bkgcut, const TString& splitOpt )
{
   // prepare the training and test trees

   // if event-wise data assignment, add local trees to dataset first
   SetInputTreesFromEventAssignTrees();

   Log() << kINFO << "Preparing trees for training and testing..." << Endl;
   AddCut( sigcut, "Signal"  );
   AddCut( bkgcut, "Background" );

   DefaultDataSetInfo().SetSplitOptions( splitOpt );
}

//_______________________________________________________________________
TMVA::MethodBase* TMVA::Factory::BookMethod( TString theMethodName, TString methodTitle, TString theOption )
{
   // Book a classifier or regression method

   if( fAnalysisType == Types::kNoAnalysisType ){
      if( DefaultDataSetInfo().GetNClasses()==2
          && DefaultDataSetInfo().GetClassInfo("Signal") != NULL
          && DefaultDataSetInfo().GetClassInfo("Background") != NULL
          ){
         fAnalysisType = Types::kClassification; // default is classification
      } else if( DefaultDataSetInfo().GetNClasses() >= 2 ){
         fAnalysisType = Types::kMulticlass;    // if two classes, but not named "Signal" and "Background"
      } else
         Log() << kFATAL << "No analysis type for " << DefaultDataSetInfo().GetNClasses() << " classes and "
               << DefaultDataSetInfo().GetNTargets() << " regression targets." << Endl;
   }

   // booking via name; the names are translated into enums and the
   // corresponding overloaded BookMethod is called
   if (GetMethod( methodTitle ) != 0) {
      Log() << kFATAL << "Booking failed since method with title <"
            << methodTitle <<"> already exists"
            << Endl;
   }

   Log() << kINFO << "Booking method: " << gTools().Color("bold") << methodTitle 
         << gTools().Color("reset") << Endl;

   // interpret option string with respect to a request for boosting (i.e., BostNum > 0)
   Int_t    boostNum = 0;
   TMVA::Configurable* conf = new TMVA::Configurable( theOption );
   conf->DeclareOptionRef( boostNum = 0, "Boost_num",
                           "Number of times the classifier will be boosted" );
   conf->ParseOptions();
   delete conf;

   // initialize methods
   IMethod* im;
   if (!boostNum) {
      im = ClassifierFactory::Instance().Create( std::string(theMethodName),
                                                 fJobName,
                                                 methodTitle,
                                                 DefaultDataSetInfo(),
                                                 theOption );
   }
   else {
      // boosted classifier, requires a specific definition, making it transparent for the user
      Log() << "Boost Number is " << boostNum << " > 0: train boosted classifier" << Endl;
      im = ClassifierFactory::Instance().Create( std::string("Boost"),
                                                 fJobName,
                                                 methodTitle,
                                                 DefaultDataSetInfo(),
                                                 theOption );
      MethodBoost* methBoost = dynamic_cast<MethodBoost*>(im); // DSMTEST divided into two lines
      if (!methBoost) // DSMTEST
         Log() << kFATAL << "Method with type kBoost cannot be casted to MethodCategory. /Factory" << Endl; // DSMTEST
      methBoost->SetBoostedMethodName( theMethodName ); // DSMTEST divided into two lines
      methBoost->fDataSetManager = fDataSetManager; // DSMTEST

   }

   MethodBase *method = dynamic_cast<MethodBase*>(im);
   if (method==0) return 0; // could not create method

   // set fDataSetManager if MethodCategory (to enable Category to create datasetinfo objects) // DSMTEST
   if (method->GetMethodType() == Types::kCategory) { // DSMTEST
      MethodCategory *methCat = (dynamic_cast<MethodCategory*>(im)); // DSMTEST
      if (!methCat) // DSMTEST
         Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory. /Factory" << Endl; // DSMTEST
      methCat->fDataSetManager = fDataSetManager; // DSMTEST
   } // DSMTEST


   if (!method->HasAnalysisType( fAnalysisType,
                                 DefaultDataSetInfo().GetNClasses(),
                                 DefaultDataSetInfo().GetNTargets() )) {
      Log() << kWARNING << "Method " << method->GetMethodTypeName() << " is not capable of handling " ;
      if (fAnalysisType == Types::kRegression) {
         Log() << "regression with " << DefaultDataSetInfo().GetNTargets() << " targets." << Endl;
      } 
      else if (fAnalysisType == Types::kMulticlass ) {
         Log() << "multiclass classification with " << DefaultDataSetInfo().GetNClasses() << " classes." << Endl;
      } 
      else {
         Log() << "classification with " << DefaultDataSetInfo().GetNClasses() << " classes." << Endl;
      }
      return 0;
   }


   method->SetAnalysisType( fAnalysisType );
   method->SetupMethod();
   method->ParseOptions();
   method->ProcessSetup();

   // check-for-unused-options is performed; may be overridden by derived classes
   method->CheckSetup();

   fMethods.push_back( method );

   return method;
}

//_______________________________________________________________________
TMVA::MethodBase* TMVA::Factory::BookMethod( Types::EMVA theMethod, TString methodTitle, TString theOption )
{
   // books MVA method; the option configuration string is custom for each MVA
   // the TString field "theNameAppendix" serves to define (and distinguish)
   // several instances of a given MVA, eg, when one wants to compare the
   // performance of various configurations
   return BookMethod( Types::Instance().GetMethodName( theMethod ), methodTitle, theOption );
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Factory::GetMethod( const TString &methodTitle ) const
{
   // returns pointer to MVA that corresponds to given method title
   MVector::const_iterator itrMethod    = fMethods.begin();
   MVector::const_iterator itrMethodEnd = fMethods.end();
   //
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      MethodBase* mva = dynamic_cast<MethodBase*>(*itrMethod);
      if ( (mva->GetMethodName())==methodTitle ) return mva;
   }
   return 0;
}

//_______________________________________________________________________
void TMVA::Factory::WriteDataInformation()
{
   // put correlations of input data and a few (default + user
   // selected) transformations into the root file

   RootBaseDir()->cd();

   DefaultDataSetInfo().GetDataSet(); // builds dataset (including calculation of correlation matrix)


   // correlation matrix of the default DS
   const TMatrixD* m(0);
   const TH2* h(0);
   
   if(fAnalysisType == Types::kMulticlass){
      for (UInt_t cls = 0; cls < DefaultDataSetInfo().GetNClasses() ; cls++) {
         m = DefaultDataSetInfo().CorrelationMatrix(DefaultDataSetInfo().GetClassInfo(cls)->GetName());
         h = DefaultDataSetInfo().CreateCorrelationMatrixHist(m, TString("CorrelationMatrix")+DefaultDataSetInfo().GetClassInfo(cls)->GetName(),
                                                              "Correlation Matrix ("+ DefaultDataSetInfo().GetClassInfo(cls)->GetName() +TString(")"));
         if (h!=0) {
            h->Write();
            delete h;
         }
      }
   }
   else{
      m = DefaultDataSetInfo().CorrelationMatrix( "Signal" );
      h = DefaultDataSetInfo().CreateCorrelationMatrixHist(m, "CorrelationMatrixS", "Correlation Matrix (signal)");
      if (h!=0) {
         h->Write();
         delete h;
      }
      
      m = DefaultDataSetInfo().CorrelationMatrix( "Background" );
      h = DefaultDataSetInfo().CreateCorrelationMatrixHist(m, "CorrelationMatrixB", "Correlation Matrix (background)");
      if (h!=0) {
         h->Write();
         delete h;
      }
      
      m = DefaultDataSetInfo().CorrelationMatrix( "Regression" );
      h = DefaultDataSetInfo().CreateCorrelationMatrixHist(m, "CorrelationMatrix", "Correlation Matrix");
      if (h!=0) { 
         h->Write();
         delete h;
      }
   }
   
   // some default transformations to evaluate
   // NOTE: all transformations are destroyed after this test
   TString processTrfs = "I"; //"I;N;D;P;U;G,D;"

   // plus some user defined transformations
   processTrfs = fTransformations;

   // remove any trace of identity transform - if given (avoid to apply it twice)
   std::vector<TMVA::TransformationHandler*> trfs;
   TransformationHandler* identityTrHandler = 0;

   std::vector<TString> trfsDef = gTools().SplitString(processTrfs,';');
   std::vector<TString>::iterator trfsDefIt = trfsDef.begin();
   for (; trfsDefIt!=trfsDef.end(); trfsDefIt++) {
      trfs.push_back(new TMVA::TransformationHandler(DefaultDataSetInfo(), "Factory"));
      TString trfS = (*trfsDefIt);

      Log() << kINFO << Endl;
      Log() << kINFO << "current transformation string: '" << trfS.Data() << "'" << Endl;
      TMVA::MethodBase::CreateVariableTransforms( trfS, 
                                                  DefaultDataSetInfo(),
                                                  *(trfs.back()),
                                                  Log() );

      if (trfS.BeginsWith('I')) identityTrHandler = trfs.back();
   }

   const std::vector<Event*>& inputEvents = DefaultDataSetInfo().GetDataSet()->GetEventCollection();

   // apply all transformations
   std::vector<TMVA::TransformationHandler*>::iterator trfIt = trfs.begin();

   for (;trfIt != trfs.end(); trfIt++) {
      // setting a Root dir causes the variables distributions to be saved to the root file
      (*trfIt)->SetRootDir(RootBaseDir());
      (*trfIt)->CalcTransformations(inputEvents);      
   }
   if(identityTrHandler) identityTrHandler->PrintVariableRanking();

   // clean up
   for (trfIt = trfs.begin(); trfIt != trfs.end(); trfIt++) delete *trfIt;
}

//_______________________________________________________________________
void TMVA::Factory::OptimizeAllMethods(TString fomType, TString fitType) 
{
   // iterates through all booked methods and sees if they use parameter tuning and if so..
   // does just that  i.e. calls "Method::Train()" for different parameter setttings and
   // keeps in mind the "optimal one"... and that's the one that will later on be used
   // in the main training loop.

 
   MVector::iterator itrMethod;

   // iterate over methods and optimize
   for( itrMethod = fMethods.begin(); itrMethod != fMethods.end(); ++itrMethod ) {
      Event::SetIsTraining(kTRUE);
      MethodBase* mva = dynamic_cast<MethodBase*>(*itrMethod);
      if (!mva) {
         Log() << kFATAL << "Dynamic cast to MethodBase failed" <<Endl;
         return;
      }

      if (mva->Data()->GetNTrainingEvents() < MinNoTrainingEvents) {
         Log() << kWARNING << "Method " << mva->GetMethodName() 
               << " not trained (training tree has less entries ["
               << mva->Data()->GetNTrainingEvents() 
               << "] than required [" << MinNoTrainingEvents << "]" << Endl; 
         continue;
      }

      Log() << kINFO << "Optimize method: " << mva->GetMethodName() << " for " 
            << (fAnalysisType == Types::kRegression ? "Regression" : 
                (fAnalysisType == Types::kMulticlass ? "Multiclass classification" : "Classification")) << Endl;
      
      mva->OptimizeTuningParameters(fomType,fitType);
      Log() << kINFO << "Optimization of tuning paremters finished for Method:"<<mva->GetName() << Endl;
   }
}

//_______________________________________________________________________
void TMVA::Factory::TrainAllMethods() 
{  
   // iterates through all booked methods and calls training

   if(fDataInputHandler->GetEntries() <=1) { // 0 entries --> 0 events, 1 entry --> dynamical dataset (or one entry)
      Log() << kFATAL << "No input data for the training provided!" << Endl;
   }
   
   if(fAnalysisType == Types::kRegression && DefaultDataSetInfo().GetNTargets() < 1 )
      Log() << kFATAL << "You want to do regression training without specifying a target." << Endl;
   else if( (fAnalysisType == Types::kMulticlass || fAnalysisType == Types::kClassification) 
            && DefaultDataSetInfo().GetNClasses() < 2 ) 
      Log() << kFATAL << "You want to do classification training, but specified less than two classes." << Endl;

   // iterates over all MVAs that have been booked, and calls their training methods

   // first print some information about the default dataset
   WriteDataInformation();

   // don't do anything if no method booked
   if (fMethods.empty()) {
      Log() << kINFO << "...nothing found to train" << Endl;
      return;
   }

   // here the training starts
   Log() << kINFO << " " << Endl;
   Log() << kINFO << "Train all methods for " 
         << (fAnalysisType == Types::kRegression ? "Regression" : 
             (fAnalysisType == Types::kMulticlass ? "Multiclass" : "Classification") ) << " ..." << Endl;

   MVector::iterator itrMethod;

   // iterate over methods and train
   for( itrMethod = fMethods.begin(); itrMethod != fMethods.end(); ++itrMethod ) {
      Event::SetIsTraining(kTRUE);
      MethodBase* mva = dynamic_cast<MethodBase*>(*itrMethod);
      if(mva==0) continue;

      if (mva->Data()->GetNTrainingEvents() < MinNoTrainingEvents) {
         Log() << kWARNING << "Method " << mva->GetMethodName()
               << " not trained (training tree has less entries ["
               << mva->Data()->GetNTrainingEvents()
               << "] than required [" << MinNoTrainingEvents << "]" << Endl;
         continue;
      }

      Log() << kINFO << "Train method: " << mva->GetMethodName() << " for "
            << (fAnalysisType == Types::kRegression ? "Regression" :
                (fAnalysisType == Types::kMulticlass ? "Multiclass classification" : "Classification")) << Endl;
      mva->TrainMethod();
      Log() << kINFO << "Training finished" << Endl;
   }

   if (fAnalysisType != Types::kRegression) {

      // variable ranking
      Log() << Endl;
      Log() << kINFO << "Ranking input variables (method specific)..." << Endl;
      for (itrMethod = fMethods.begin(); itrMethod != fMethods.end(); itrMethod++) {
         MethodBase* mva = dynamic_cast<MethodBase*>(*itrMethod);
         if (mva && mva->Data()->GetNTrainingEvents() >= MinNoTrainingEvents) {

            // create and print ranking
            const Ranking* ranking = (*itrMethod)->CreateRanking();
            if (ranking != 0) ranking->Print();
            else Log() << kINFO << "No variable ranking supplied by classifier: "
                       << dynamic_cast<MethodBase*>(*itrMethod)->GetMethodName() << Endl;
         }
      }
   }

   // delete all methods and recreate them from weight file - this ensures that the application
   // of the methods (in TMVAClassificationApplication) is consistent with the results obtained
   // in the testing
   Log() << Endl;
   if (RECREATE_METHODS) {

      Log() << kINFO << "=== Destroy and recreate all methods via weight files for testing ===" << Endl << Endl;

      RootBaseDir()->cd();

      // iterate through all booked methods
      for (UInt_t i=0; i<fMethods.size(); i++) {

         MethodBase* m = dynamic_cast<MethodBase*>(fMethods[i]);
         if(m==0) continue;

         TMVA::Types::EMVA methodType = m->GetMethodType();
         TString           weightfile = m->GetWeightFileName();

         // decide if .txt or .xml file should be read:
         if (READXML) weightfile.ReplaceAll(".txt",".xml");

         DataSetInfo& dataSetInfo = m->DataInfo();
         TString      testvarName = m->GetTestvarName();
         delete m; //itrMethod[i];

         // recreate
         m = dynamic_cast<MethodBase*>( ClassifierFactory::Instance()
                                        .Create( std::string(Types::Instance().GetMethodName(methodType)), 
                                                 dataSetInfo, weightfile ) );
         if( m->GetMethodType() == Types::kCategory ){ 
            MethodCategory *methCat = (dynamic_cast<MethodCategory*>(m));
            if( !methCat ) Log() << kFATAL << "Method with type kCategory cannot be casted to MethodCategory. /Factory" << Endl; 
            else methCat->fDataSetManager = fDataSetManager;
         }
         //ToDo, Do we need to fill the DataSetManager of MethodBoost here too?
	 
         m->SetAnalysisType(fAnalysisType);
         m->SetupMethod();
         m->ReadStateFromFile();
         m->SetTestvarName(testvarName);

         // replace trained method by newly created one (from weight file) in methods vector
         fMethods[i] = m;
      }
   }
}

//_______________________________________________________________________
void TMVA::Factory::TestAllMethods()
{
   Log() << kINFO << "Test all methods..." << Endl;

   // don't do anything if no method booked
   if (fMethods.empty()) {
      Log() << kINFO << "...nothing found to test" << Endl;
      return;
   }

   // iterates over all MVAs that have been booked, and calls their testing methods
   // iterate over methods and test
   MVector::iterator itrMethod    = fMethods.begin();
   MVector::iterator itrMethodEnd = fMethods.end();
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      Event::SetIsTraining(kFALSE);
      MethodBase* mva = dynamic_cast<MethodBase*>(*itrMethod);
      if(mva==0) continue;
      Types::EAnalysisType analysisType = mva->GetAnalysisType();
      Log() << kINFO << "Test method: " << mva->GetMethodName() << " for "
            << (analysisType == Types::kRegression ? "Regression" :
                (analysisType == Types::kMulticlass ? "Multiclass classification" : "Classification")) << " performance" << Endl;
      mva->AddOutput( Types::kTesting, analysisType );
   }
}

//_______________________________________________________________________
void TMVA::Factory::MakeClass( const TString& methodTitle ) const
{
   // Print predefined help message of classifier
   // iterate over methods and test
   if (methodTitle != "") {
      IMethod* method = GetMethod( methodTitle );
      if (method) method->MakeClass();
      else {
         Log() << kWARNING << "<MakeClass> Could not find classifier \"" << methodTitle 
               << "\" in list" << Endl;
      }
   }
   else {

      // no classifier specified, print all hepl messages
      MVector::const_iterator itrMethod    = fMethods.begin();
      MVector::const_iterator itrMethodEnd = fMethods.end();
      for (; itrMethod != itrMethodEnd; itrMethod++) {
         MethodBase* method = dynamic_cast<MethodBase*>(*itrMethod);
         if(method==0) continue;
         Log() << kINFO << "Make response class for classifier: " << method->GetMethodName() << Endl;
         method->MakeClass();
      }
   }
}

//_______________________________________________________________________
void TMVA::Factory::PrintHelpMessage( const TString& methodTitle ) const
{
   // Print predefined help message of classifier
   // iterate over methods and test
   if (methodTitle != "") {
      IMethod* method = GetMethod( methodTitle );
      if (method) method->PrintHelpMessage();
      else {
         Log() << kWARNING << "<PrintHelpMessage> Could not find classifier \"" << methodTitle 
               << "\" in list" << Endl;
      }
   }
   else {

      // no classifier specified, print all hepl messages
      MVector::const_iterator itrMethod    = fMethods.begin();
      MVector::const_iterator itrMethodEnd = fMethods.end();
      for (; itrMethod != itrMethodEnd; itrMethod++) {
         MethodBase* method = dynamic_cast<MethodBase*>(*itrMethod);
         if(method==0) continue;
         Log() << kINFO << "Print help message for classifier: " << method->GetMethodName() << Endl;
         method->PrintHelpMessage();
      }
   }
}

//_______________________________________________________________________
void TMVA::Factory::EvaluateAllVariables( TString options )
{
   // iterates over all MVA input varables and evaluates them
   Log() << kINFO << "Evaluating all variables..." << Endl;
   Event::SetIsTraining(kFALSE);

   for (UInt_t i=0; i<DefaultDataSetInfo().GetNVariables(); i++) {
      TString s = DefaultDataSetInfo().GetVariableInfo(i).GetLabel();
      if (options.Contains("V")) s += ":V";
      this->BookMethod( "Variable", s );
   }
}

//_______________________________________________________________________
void TMVA::Factory::EvaluateAllMethods( void )
{
   // iterates over all MVAs that have been booked, and calls their evaluation methods
   Log() << kINFO << "Evaluate all methods..." << Endl;

   // don't do anything if no method booked
   if (fMethods.empty()) {
      Log() << kINFO << "...nothing found to evaluate" << Endl;
      return;
   }

   // -----------------------------------------------------------------------
   // First part of evaluation process
   // --> compute efficiencies, and other separation estimators
   // -----------------------------------------------------------------------

   // although equal, we now want to seperate the outpuf for the variables
   // and the real methods
   Int_t isel;                  // will be 0 for a Method; 1 for a Variable
   Int_t nmeth_used[2] = {0,0}; // 0 Method; 1 Variable

   std::vector<std::vector<TString> >  mname(2);
   std::vector<std::vector<Double_t> > sig(2), sep(2), roc(2);
   std::vector<std::vector<Double_t> > eff01(2), eff10(2), eff30(2), effArea(2);
   std::vector<std::vector<Double_t> > eff01err(2), eff10err(2), eff30err(2);
   std::vector<std::vector<Double_t> > trainEff01(2), trainEff10(2), trainEff30(2);

   std::vector<std::vector<Float_t> > multiclass_testEff;
   std::vector<std::vector<Float_t> > multiclass_trainEff;
   std::vector<std::vector<Float_t> > multiclass_testPur;
   std::vector<std::vector<Float_t> > multiclass_trainPur;

   std::vector<std::vector<Double_t> > biastrain(1);  // "bias" of the regression on the training data
   std::vector<std::vector<Double_t> > biastest(1);   // "bias" of the regression on test data
   std::vector<std::vector<Double_t> > devtrain(1);   // "dev" of the regression on the training data
   std::vector<std::vector<Double_t> > devtest(1);    // "dev" of the regression on test data
   std::vector<std::vector<Double_t> > rmstrain(1);   // "rms" of the regression on the training data
   std::vector<std::vector<Double_t> > rmstest(1);    // "rms" of the regression on test data
   std::vector<std::vector<Double_t> > minftrain(1);  // "minf" of the regression on the training data
   std::vector<std::vector<Double_t> > minftest(1);   // "minf" of the regression on test data
   std::vector<std::vector<Double_t> > rhotrain(1);   // correlation of the regression on the training data
   std::vector<std::vector<Double_t> > rhotest(1);    // correlation of the regression on test data

   // same as above but for 'truncated' quantities (computed for events within 2sigma of RMS)
   std::vector<std::vector<Double_t> > biastrainT(1);
   std::vector<std::vector<Double_t> > biastestT(1);
   std::vector<std::vector<Double_t> > devtrainT(1);
   std::vector<std::vector<Double_t> > devtestT(1);
   std::vector<std::vector<Double_t> > rmstrainT(1);
   std::vector<std::vector<Double_t> > rmstestT(1);
   std::vector<std::vector<Double_t> > minftrainT(1);
   std::vector<std::vector<Double_t> > minftestT(1);

   // following vector contains all methods - with the exception of Cuts, which are special
   MVector methodsNoCuts;

   Bool_t doRegression = kFALSE;
   Bool_t doMulticlass = kFALSE;

   // iterate over methods and evaluate
   MVector::iterator itrMethod    = fMethods.begin();
   MVector::iterator itrMethodEnd = fMethods.end();
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      Event::SetIsTraining(kFALSE);
      MethodBase* theMethod = dynamic_cast<MethodBase*>(*itrMethod);
      if(theMethod==0) continue;
      if (theMethod->GetMethodType() != Types::kCuts) methodsNoCuts.push_back( *itrMethod );

      if (theMethod->DoRegression()) {
         doRegression = kTRUE;

         Log() << kINFO << "Evaluate regression method: " << theMethod->GetMethodName() << Endl;
         Double_t bias, dev, rms, mInf;
         Double_t biasT, devT, rmsT, mInfT;
         Double_t rho;

         theMethod->TestRegression( bias, biasT, dev, devT, rms, rmsT, mInf, mInfT, rho, TMVA::Types::kTesting  );
         biastest[0]  .push_back( bias );
         devtest[0]   .push_back( dev );
         rmstest[0]   .push_back( rms );
         minftest[0]  .push_back( mInf );
         rhotest[0]   .push_back( rho );
         biastestT[0] .push_back( biasT );
         devtestT[0]  .push_back( devT );
         rmstestT[0]  .push_back( rmsT );
         minftestT[0] .push_back( mInfT );

         theMethod->TestRegression( bias, biasT, dev, devT, rms, rmsT, mInf, mInfT, rho, TMVA::Types::kTraining  );
         biastrain[0] .push_back( bias );
         devtrain[0]  .push_back( dev );
         rmstrain[0]  .push_back( rms );
         minftrain[0] .push_back( mInf );
         rhotrain[0]  .push_back( rho );
         biastrainT[0].push_back( biasT );
         devtrainT[0] .push_back( devT );
         rmstrainT[0] .push_back( rmsT );
         minftrainT[0].push_back( mInfT );

         mname[0].push_back( theMethod->GetMethodName() );
         nmeth_used[0]++;

         Log() << kINFO << "Write evaluation histograms to file" << Endl;
         theMethod->WriteEvaluationHistosToFile(Types::kTesting);
         theMethod->WriteEvaluationHistosToFile(Types::kTraining);
      } 
      else if (theMethod->DoMulticlass()) {
         doMulticlass = kTRUE;
         Log() << kINFO << "Evaluate multiclass classification method: " << theMethod->GetMethodName() << Endl;
         Log() << kINFO << "Write evaluation histograms to file" << Endl;
         theMethod->WriteEvaluationHistosToFile(Types::kTesting);
         theMethod->WriteEvaluationHistosToFile(Types::kTraining);
         
         theMethod->TestMulticlass();
         multiclass_testEff.push_back(theMethod->GetMulticlassEfficiency(multiclass_testPur));

         nmeth_used[0]++;
         mname[0].push_back( theMethod->GetMethodName() );
      } 
      else {
         
         Log() << kINFO << "Evaluate classifier: " << theMethod->GetMethodName() << Endl;
         isel = (theMethod->GetMethodTypeName().Contains("Variable")) ? 1 : 0;
      
         // perform the evaluation
         theMethod->TestClassification();
         
         // evaluate the classifier
         mname[isel].push_back( theMethod->GetMethodName() );
         sig[isel].push_back  ( theMethod->GetSignificance() );
         sep[isel].push_back  ( theMethod->GetSeparation() );
         roc[isel].push_back  ( theMethod->GetROCIntegral() );

         Double_t err;
         eff01[isel].push_back( theMethod->GetEfficiency("Efficiency:0.01", Types::kTesting, err) );
         eff01err[isel].push_back( err );
         eff10[isel].push_back( theMethod->GetEfficiency("Efficiency:0.10", Types::kTesting, err) );
         eff10err[isel].push_back( err );
         eff30[isel].push_back( theMethod->GetEfficiency("Efficiency:0.30", Types::kTesting, err) );
         eff30err[isel].push_back( err );
         effArea[isel].push_back( theMethod->GetEfficiency("",              Types::kTesting, err)  ); // computes the area (average)

         trainEff01[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.01") ); // the first pass takes longer
         trainEff10[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.10") );
         trainEff30[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.30") );

         nmeth_used[isel]++;

         Log() << kINFO << "Write evaluation histograms to file" << Endl;
         theMethod->WriteEvaluationHistosToFile(Types::kTesting);
         theMethod->WriteEvaluationHistosToFile(Types::kTraining);
      }
   }
   if (doRegression) {

      std::vector<TString> vtemps = mname[0];
      std::vector< std::vector<Double_t> > vtmp;
      vtmp.push_back( devtest[0]   );  // this is the vector that is ranked
      vtmp.push_back( devtrain[0]  );
      vtmp.push_back( biastest[0]  );
      vtmp.push_back( biastrain[0] );
      vtmp.push_back( rmstest[0]   );
      vtmp.push_back( rmstrain[0]  );
      vtmp.push_back( minftest[0]  );
      vtmp.push_back( minftrain[0] );
      vtmp.push_back( rhotest[0]   );
      vtmp.push_back( rhotrain[0]  );
      vtmp.push_back( devtestT[0]  );  // this is the vector that is ranked
      vtmp.push_back( devtrainT[0] );
      vtmp.push_back( biastestT[0] );
      vtmp.push_back( biastrainT[0]);
      vtmp.push_back( rmstestT[0]  );
      vtmp.push_back( rmstrainT[0] );
      vtmp.push_back( minftestT[0] );
      vtmp.push_back( minftrainT[0]);
      gTools().UsefulSortAscending( vtmp, &vtemps );
      mname[0]      = vtemps;
      devtest[0]    = vtmp[0];
      devtrain[0]   = vtmp[1];
      biastest[0]   = vtmp[2];
      biastrain[0]  = vtmp[3];
      rmstest[0]    = vtmp[4];
      rmstrain[0]   = vtmp[5];
      minftest[0]   = vtmp[6];
      minftrain[0]  = vtmp[7];
      rhotest[0]    = vtmp[8];
      rhotrain[0]   = vtmp[9];
      devtestT[0]   = vtmp[10];
      devtrainT[0]  = vtmp[11];
      biastestT[0]  = vtmp[12];
      biastrainT[0] = vtmp[13];
      rmstestT[0]   = vtmp[14];
      rmstrainT[0]  = vtmp[15];
      minftestT[0]  = vtmp[16];
      minftrainT[0] = vtmp[17];
   } 
   else if (doMulticlass) {
      // TODO: fill in something meaningfull
      
   }  
   else {
      // now sort the variables according to the best 'eff at Beff=0.10'
      for (Int_t k=0; k<2; k++) {
         std::vector< std::vector<Double_t> > vtemp;
         vtemp.push_back( effArea[k] );  // this is the vector that is ranked
         vtemp.push_back( eff10[k] );
         vtemp.push_back( eff01[k] );
         vtemp.push_back( eff30[k] );
         vtemp.push_back( eff10err[k] ); 
         vtemp.push_back( eff01err[k] );
         vtemp.push_back( eff30err[k] );
         vtemp.push_back( trainEff10[k] );
         vtemp.push_back( trainEff01[k] );
         vtemp.push_back( trainEff30[k] );
         vtemp.push_back( sig[k] );
         vtemp.push_back( sep[k] );
         vtemp.push_back( roc[k] );
         std::vector<TString> vtemps = mname[k];
         gTools().UsefulSortDescending( vtemp, &vtemps );
         effArea[k]    = vtemp[0];
         eff10[k]      = vtemp[1];
         eff01[k]      = vtemp[2];
         eff30[k]      = vtemp[3];
         eff10err[k]   = vtemp[4];
         eff01err[k]   = vtemp[5];
         eff30err[k]   = vtemp[6];
         trainEff10[k] = vtemp[7];
         trainEff01[k] = vtemp[8];
         trainEff30[k] = vtemp[9];
         sig[k]        = vtemp[10];
         sep[k]        = vtemp[11];
         roc[k]        = vtemp[12];
         mname[k]      = vtemps;
      }
   }

   // -----------------------------------------------------------------------
   // Second part of evaluation process
   // --> compute correlations among MVAs
   // --> compute correlations between input variables and MVA (determines importsance)
   // --> count overlaps
   // -----------------------------------------------------------------------
   
   const Int_t nmeth = methodsNoCuts.size();
   const Int_t nvar  = DefaultDataSetInfo().GetNVariables();
   if (!doRegression && !doMulticlass ) {

      if (nmeth > 0) {

         // needed for correlations
         Double_t *dvec = new Double_t[nmeth+nvar];
         std::vector<Double_t> rvec;

         // for correlations
         TPrincipal* tpSig = new TPrincipal( nmeth+nvar, "" );   
         TPrincipal* tpBkg = new TPrincipal( nmeth+nvar, "" );   

         // set required tree branch references
         Int_t ivar = 0;
         std::vector<TString>* theVars = new std::vector<TString>;
         std::vector<ResultsClassification*> mvaRes;
         for (itrMethod = methodsNoCuts.begin(); itrMethod != methodsNoCuts.end(); itrMethod++, ivar++) {
            MethodBase* m = dynamic_cast<MethodBase*>(*itrMethod);
            if(m==0) continue;
            theVars->push_back( m->GetTestvarName() );
            rvec.push_back( m->GetSignalReferenceCut() );
            theVars->back().ReplaceAll( "MVA_", "" );
            mvaRes.push_back( dynamic_cast<ResultsClassification*>( m->Data()->GetResults( m->GetMethodName(), 
                                                                                           Types::kTesting, 
                                                                                           Types::kMaxAnalysisType) ) );
         }

         // for overlap study
         TMatrixD* overlapS = new TMatrixD( nmeth, nmeth );
         TMatrixD* overlapB = new TMatrixD( nmeth, nmeth );
         (*overlapS) *= 0; // init...
         (*overlapB) *= 0; // init...

         // loop over test tree
         DataSet* defDs = DefaultDataSetInfo().GetDataSet();
         defDs->SetCurrentType(Types::kTesting);
         for (Int_t ievt=0; ievt<defDs->GetNEvents(); ievt++) {
            const Event* ev = defDs->GetEvent(ievt);

            // for correlations
            TMatrixD* theMat = 0;
            for (Int_t im=0; im<nmeth; im++) {
               // check for NaN value
               Double_t retval = (Double_t)(*mvaRes[im])[ievt][0];
               if (TMath::IsNaN(retval)) {
                  Log() << kWARNING << "Found NaN return value in event: " << ievt
                        << " for method \"" << methodsNoCuts[im]->GetName() << "\"" << Endl;
                  dvec[im] = 0;
               }
               else dvec[im] = retval;
            }
            for (Int_t iv=0; iv<nvar;  iv++) dvec[iv+nmeth]  = (Double_t)ev->GetValue(iv);
            if (DefaultDataSetInfo().IsSignal(ev)) { tpSig->AddRow( dvec ); theMat = overlapS; }
            else                                   { tpBkg->AddRow( dvec ); theMat = overlapB; }

            // count overlaps
            for (Int_t im=0; im<nmeth; im++) {
               for (Int_t jm=im; jm<nmeth; jm++) {
                  if ((dvec[im] - rvec[im])*(dvec[jm] - rvec[jm]) > 0) {
                     (*theMat)(im,jm)++;
                     if (im != jm) (*theMat)(jm,im)++;
                  }
               }
            }
         }

         // renormalise overlap matrix
         (*overlapS) *= (1.0/defDs->GetNEvtSigTest());  // init...
         (*overlapB) *= (1.0/defDs->GetNEvtBkgdTest()); // init...

         tpSig->MakePrincipals();
         tpBkg->MakePrincipals();

         const TMatrixD* covMatS = tpSig->GetCovarianceMatrix();
         const TMatrixD* covMatB = tpBkg->GetCovarianceMatrix();
   
         const TMatrixD* corrMatS = gTools().GetCorrelationMatrix( covMatS );
         const TMatrixD* corrMatB = gTools().GetCorrelationMatrix( covMatB );

         // print correlation matrices
         if (corrMatS != 0 && corrMatB != 0) {

            // extract MVA matrix
            TMatrixD mvaMatS(nmeth,nmeth);
            TMatrixD mvaMatB(nmeth,nmeth);
            for (Int_t im=0; im<nmeth; im++) {
               for (Int_t jm=0; jm<nmeth; jm++) {
                  mvaMatS(im,jm) = (*corrMatS)(im,jm);
                  mvaMatB(im,jm) = (*corrMatB)(im,jm);
               }
            }
         
            // extract variables - to MVA matrix
            std::vector<TString> theInputVars;
            TMatrixD varmvaMatS(nvar,nmeth);
            TMatrixD varmvaMatB(nvar,nmeth);
            for (Int_t iv=0; iv<nvar; iv++) {
               theInputVars.push_back( DefaultDataSetInfo().GetVariableInfo( iv ).GetLabel() );
               for (Int_t jm=0; jm<nmeth; jm++) {
                  varmvaMatS(iv,jm) = (*corrMatS)(nmeth+iv,jm);
                  varmvaMatB(iv,jm) = (*corrMatB)(nmeth+iv,jm);
               }
            }

            if (nmeth > 1) {
               Log() << kINFO << Endl;
               Log() << kINFO << "Inter-MVA correlation matrix (signal):" << Endl;
               gTools().FormattedOutput( mvaMatS, *theVars, Log() );
               Log() << kINFO << Endl;

               Log() << kINFO << "Inter-MVA correlation matrix (background):" << Endl;
               gTools().FormattedOutput( mvaMatB, *theVars, Log() );
               Log() << kINFO << Endl;   
            }

            Log() << kINFO << "Correlations between input variables and MVA response (signal):" << Endl;
            gTools().FormattedOutput( varmvaMatS, theInputVars, *theVars, Log() );
            Log() << kINFO << Endl;

            Log() << kINFO << "Correlations between input variables and MVA response (background):" << Endl;
            gTools().FormattedOutput( varmvaMatB, theInputVars, *theVars, Log() );
            Log() << kINFO << Endl;
         }
         else Log() << kWARNING << "<TestAllMethods> cannot compute correlation matrices" << Endl;

         // print overlap matrices
         Log() << kINFO << "The following \"overlap\" matrices contain the fraction of events for which " << Endl;
         Log() << kINFO << "the MVAs 'i' and 'j' have returned conform answers about \"signal-likeness\"" << Endl;
         Log() << kINFO << "An event is signal-like, if its MVA output exceeds the following value:" << Endl;
         gTools().FormattedOutput( rvec, *theVars, "Method" , "Cut value", Log() );
         Log() << kINFO << "which correspond to the working point: eff(signal) = 1 - eff(background)" << Endl;

         // give notice that cut method has been excluded from this test
         if (nmeth != (Int_t)fMethods.size()) 
            Log() << kINFO << "Note: no correlations and overlap with cut method are provided at present" << Endl;

         if (nmeth > 1) {
            Log() << kINFO << Endl;
            Log() << kINFO << "Inter-MVA overlap matrix (signal):" << Endl;
            gTools().FormattedOutput( *overlapS, *theVars, Log() );
            Log() << kINFO << Endl;
      
            Log() << kINFO << "Inter-MVA overlap matrix (background):" << Endl;
            gTools().FormattedOutput( *overlapB, *theVars, Log() );
         }

         // cleanup
         delete tpSig;
         delete tpBkg;
         delete corrMatS;
         delete corrMatB;
         delete theVars;
         delete overlapS;
         delete overlapB;
         delete [] dvec;
      }
   }

   // -----------------------------------------------------------------------
   // Third part of evaluation process
   // --> output
   // ----------------------------------------------------------------------- 

   if (doRegression) {

      Log() << kINFO << Endl;
      TString hLine = "-------------------------------------------------------------------------";
      Log() << kINFO << "Evaluation results ranked by smallest RMS on test sample:" << Endl;
      Log() << kINFO << "(\"Bias\" quotes the mean deviation of the regression from true target." << Endl;
      Log() << kINFO << " \"MutInf\" is the \"Mutual Information\" between regression and target." << Endl;
      Log() << kINFO << " Indicated by \"_T\" are the corresponding \"truncated\" quantities ob-" << Endl;
      Log() << kINFO << " tained when removing events deviating more than 2sigma from average.)" << Endl;
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << "MVA Method:        <Bias>   <Bias_T>    RMS    RMS_T  |  MutInf MutInf_T" << Endl;
      Log() << kINFO << hLine << Endl;

      for (Int_t i=0; i<nmeth_used[0]; i++) {
         Log() << kINFO << Form("%-15s:%#9.3g%#9.3g%#9.3g%#9.3g  |  %#5.3f  %#5.3f",
                                (const char*)mname[0][i], 
                                biastest[0][i], biastestT[0][i], 
                                rmstest[0][i], rmstestT[0][i], 
                                minftest[0][i], minftestT[0][i] )
               << Endl;
      }
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << Endl;
      Log() << kINFO << "Evaluation results ranked by smallest RMS on training sample:" << Endl;
      Log() << kINFO << "(overtraining check)" << Endl;
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << "MVA Method:        <Bias>   <Bias_T>    RMS    RMS_T  |  MutInf MutInf_T" << Endl;
      Log() << kINFO << hLine << Endl;

      for (Int_t i=0; i<nmeth_used[0]; i++) {
         Log() << kINFO << Form("%-15s:%#9.3g%#9.3g%#9.3g%#9.3g  |  %#5.3f  %#5.3f",
                                (const char*)mname[0][i], 
                                biastrain[0][i], biastrainT[0][i], 
                                rmstrain[0][i], rmstrainT[0][i], 
                                minftrain[0][i], minftrainT[0][i] )
               << Endl;
      }
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << Endl;
   }
   else if( doMulticlass ){
      Log() << Endl;
      TString hLine = "--------------------------------------------------------------------------------";
      Log() << kINFO << "Evaluation results ranked by best signal efficiency times signal purity " << Endl;
      Log() << kINFO << hLine << Endl;
      TString header= "MVA Method     "; 
      for(UInt_t icls = 0; icls<DefaultDataSetInfo().GetNClasses(); ++icls){
         header += Form("%-12s ",DefaultDataSetInfo().GetClassInfo(icls)->GetName().Data());
      }
      Log() << kINFO << header << Endl;
      Log() << kINFO << hLine << Endl;
      for (Int_t i=0; i<nmeth_used[0]; i++) {
         TString res =  Form("%-15s",(const char*)mname[0][i]);
         for(UInt_t icls = 0; icls<DefaultDataSetInfo().GetNClasses(); ++icls){
            res += Form("%#1.3f        ",(multiclass_testEff[i][icls])*(multiclass_testPur[i][icls]));
         }
         Log() << kINFO << res << Endl;
      }
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << Endl;

   } 
   else {
      Log() << Endl;
      TString hLine = "--------------------------------------------------------------------------------";
      Log() << kINFO << "Evaluation results ranked by best signal efficiency and purity (area)" << Endl;
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << "MVA              Signal efficiency at bkg eff.(error):       | Sepa-    Signifi- "   << Endl;
      Log() << kINFO << "Method:          @B=0.01    @B=0.10    @B=0.30    ROC-integ. | ration:  cance:   "   << Endl;
      Log() << kINFO << hLine << Endl;
      for (Int_t k=0; k<2; k++) {
         if (k == 1 && nmeth_used[k] > 0) {
            Log() << kINFO << hLine << Endl;
            Log() << kINFO << "Input Variables: " << Endl << hLine << Endl;
         }
         for (Int_t i=0; i<nmeth_used[k]; i++) {
            if (k == 1) mname[k][i].ReplaceAll( "Variable_", "" );
            if (sep[k][i] < 0 || sig[k][i] < 0) {
               // cannot compute separation/significance -> no MVA (usually for Cuts)
               Log() << kINFO << Form("%-15s: %#1.3f(%02i)  %#1.3f(%02i)  %#1.3f(%02i)    %#1.3f    | --       --",
                                      (const char*)mname[k][i], 
                                      eff01[k][i], Int_t(1000*eff01err[k][i]), 
                                      eff10[k][i], Int_t(1000*eff10err[k][i]), 
                                      eff30[k][i], Int_t(1000*eff30err[k][i]), 
                                      effArea[k][i]) << Endl;
            }
            else {
               Log() << kINFO << Form("%-15s: %#1.3f(%02i)  %#1.3f(%02i)  %#1.3f(%02i)    %#1.3f    | %#1.3f    %#1.3f",
                                      (const char*)mname[k][i], 
                                      eff01[k][i], Int_t(1000*eff01err[k][i]), 
                                      eff10[k][i], Int_t(1000*eff10err[k][i]), 
                                      eff30[k][i], Int_t(1000*eff30err[k][i]), 
                                      effArea[k][i], 
                                      sep[k][i], sig[k][i]) << Endl;
            }
         }
      }
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << Endl;
      Log() << kINFO << "Testing efficiency compared to training efficiency (overtraining check)" << Endl;
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << "MVA              Signal efficiency: from test sample (from training sample) "   << Endl;
      Log() << kINFO << "Method:          @B=0.01             @B=0.10            @B=0.30   "   << Endl;
      Log() << kINFO << hLine << Endl;
      for (Int_t k=0; k<2; k++) {
         if (k == 1 && nmeth_used[k] > 0) {
            Log() << kINFO << hLine << Endl;
            Log() << kINFO << "Input Variables: " << Endl << hLine << Endl;
         }
         for (Int_t i=0; i<nmeth_used[k]; i++) {
            if (k == 1) mname[k][i].ReplaceAll( "Variable_", "" );
            Log() << kINFO << Form("%-15s: %#1.3f (%#1.3f)       %#1.3f (%#1.3f)      %#1.3f (%#1.3f)",
                                   (const char*)mname[k][i], 
                                   eff01[k][i],trainEff01[k][i], 
                                   eff10[k][i],trainEff10[k][i],
                                   eff30[k][i],trainEff30[k][i]) << Endl;
         }
      }
      Log() << kINFO << hLine << Endl;
      Log() << kINFO << Endl; 
   }

   // write test tree
   RootBaseDir()->cd();
   DefaultDataSetInfo().GetDataSet()->GetTree(Types::kTesting) ->Write( "", TObject::kOverwrite );
   DefaultDataSetInfo().GetDataSet()->GetTree(Types::kTraining)->Write( "", TObject::kOverwrite );

   // references for citation
   gTools().TMVACitation( Log(), Tools::kHtmlLink );
}

