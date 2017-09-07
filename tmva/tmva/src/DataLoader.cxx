// @(#)root/tmva $Id$
// Author: Omar Zapata
// Mentors: Lorenzo Moneta, Sergei Gleyzer
//NOTE: Based on TMVA::Factory

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataLoader                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This is a class to load datasets into every booked method                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Lorenzo Moneta <Lorenzo.Moneta@cern.ch> - CERN, Switzerland               *
 *      Omar Zapata <Omar.Zapata@cern.ch>  - ITM/UdeA, Colombia                   *
 *      Sergei Gleyzer<sergei.gleyzer@cern.ch> - CERN, Switzerland                *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      ITM/UdeA, Colombia                                                        *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/


/*! \class TMVA::DataLoader
\ingroup TMVA

*/

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
#include "TRandom3.h"

#include <string.h>

#include "TMVA/Configurable.h"
#include "TMVA/DataLoader.h"
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

#include "TMVA/VariableInfo.h"
#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"
#include "TMVA/VarTransformHandler.h"


#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsRegression.h"
#include "TMVA/ResultsMulticlass.h"
#include "TMVA/Types.h"

ClassImp(TMVA::DataLoader);


////////////////////////////////////////////////////////////////////////////////

TMVA::DataLoader::DataLoader( TString thedlName)
: Configurable( ),
   fDataSetManager       ( NULL ), //DSMTEST
   fDataInputHandler     ( new DataInputHandler ),
   fTransformations      ( "I" ),
   fVerbose              ( kFALSE ),
   fDataAssignType       ( kAssignEvents ),
   fATreeEvent           (0),
   fMakeFoldDataSet      ( kFALSE )
{
   fDataSetManager = new DataSetManager( *fDataInputHandler ); // DSMTEST
   SetName(thedlName.Data());
   fLogger->SetSource("DataLoader");
}

////////////////////////////////////////////////////////////////////////////////

TMVA::DataLoader::~DataLoader( void )
{
   // destructor

   std::vector<TMVA::VariableTransformBase*>::iterator trfIt = fDefaultTrfs.begin();
   for (;trfIt != fDefaultTrfs.end(); trfIt++) delete (*trfIt);

   delete fDataInputHandler;

   // destroy singletons
   //   DataSetManager::DestroyInstance(); // DSMTEST replaced by following line
   delete fDataSetManager; // DSMTEST

   // problem with call of REGISTER_METHOD macro ...
   //   ClassifierDataLoader::DestroyInstance();
   //   Types::DestroyInstance();
   //Tools::DestroyInstance();
   //Config::DestroyInstance();
}


////////////////////////////////////////////////////////////////////////////////

TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( DataSetInfo &dsi )
{
   return fDataSetManager->AddDataSetInfo(dsi); // DSMTEST
}

////////////////////////////////////////////////////////////////////////////////

TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( const TString& dsiName )
{
   DataSetInfo* dsi = fDataSetManager->GetDataSetInfo(dsiName); // DSMTEST

   if (dsi!=0) return *dsi;

   return fDataSetManager->AddDataSetInfo(*(new DataSetInfo(dsiName))); // DSMTEST
}

////////////////////////////////////////////////////////////////////////////////

TMVA::DataSetInfo& TMVA::DataLoader::GetDataSetInfo()
{
   return DefaultDataSetInfo(); // DSMTEST
}

////////////////////////////////////////////////////////////////////////////////
/// Transforms the variables and return a new DataLoader with the transformed
/// variables

TMVA::DataLoader* TMVA::DataLoader::VarTransform(TString trafoDefinition)
{
   TString trOptions = "0";
   TString trName = "None";
   if (trafoDefinition.Contains("(")) {

      // contains transformation parameters
      Ssiz_t parStart = trafoDefinition.Index( "(" );
      Ssiz_t parLen   = trafoDefinition.Index( ")", parStart )-parStart+1;

      trName = trafoDefinition(0,parStart);
      trOptions = trafoDefinition(parStart,parLen);
      trOptions.Remove(parLen-1,1);
      trOptions.Remove(0,1);
   }
   else
      trName = trafoDefinition;

   VarTransformHandler* handler = new VarTransformHandler(this);
   // variance threshold variable transformation
   if (trName == "VT") {

      // find threshold value from given input
      Double_t threshold = 0.0;
      if (!trOptions.IsFloat()){
         Log() << kFATAL << " VT transformation must be passed a floating threshold value" << Endl;
         delete handler;
         return this;
      }
      else
         threshold =  trOptions.Atof();
      TMVA::DataLoader *transformedLoader = handler->VarianceThreshold(threshold);
      delete handler;
      return transformedLoader;
   }
   else {
      Log() << kFATAL << "Incorrect transformation string provided, please check" << Endl;
   }
   Log() << kINFO << "No transformation applied, returning original loader" << Endl;
   return this;
}

////////////////////////////////////////////////////////////////////////////////
// the next functions are to assign events directly

////////////////////////////////////////////////////////////////////////////////
/// create the data assignment tree (for event-wise data assignment by user)

TTree* TMVA::DataLoader::CreateEventAssignTrees( const TString& name )
{
   TTree * assignTree = new TTree( name, name );
   assignTree->SetDirectory(0);
   assignTree->Branch( "type",   &fATreeType,   "ATreeType/I" );
   assignTree->Branch( "weight", &fATreeWeight, "ATreeWeight/F" );

   std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();
   std::vector<VariableInfo>& tgts = DefaultDataSetInfo().GetTargetInfos();
   std::vector<VariableInfo>& spec = DefaultDataSetInfo().GetSpectatorInfos();

   if (fATreeEvent.size()==0) fATreeEvent.resize(vars.size()+tgts.size()+spec.size());
   // add variables
   for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
      TString vname = vars[ivar].GetExpression();
      assignTree->Branch( vname, &fATreeEvent[ivar], vname + "/F" );
   }
   // add targets
   for (UInt_t itgt=0; itgt<tgts.size(); itgt++) {
      TString vname = tgts[itgt].GetExpression();
      assignTree->Branch( vname, &fATreeEvent[vars.size()+itgt], vname + "/F" );
   }
   // add spectators
   for (UInt_t ispc=0; ispc<spec.size(); ispc++) {
      TString vname = spec[ispc].GetExpression();
      assignTree->Branch( vname, &fATreeEvent[vars.size()+tgts.size()+ispc], vname + "/F" );
   }
   return assignTree;
}

////////////////////////////////////////////////////////////////////////////////
/// add signal training event

void TMVA::DataLoader::AddSignalTrainingEvent( const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( "Signal", Types::kTraining, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal testing event

void TMVA::DataLoader::AddSignalTestEvent( const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( "Signal", Types::kTesting, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal training event

void TMVA::DataLoader::AddBackgroundTrainingEvent( const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( "Background", Types::kTraining, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal training event

void TMVA::DataLoader::AddBackgroundTestEvent( const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( "Background", Types::kTesting, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal training event

void TMVA::DataLoader::AddTrainingEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( className, Types::kTraining, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal test event

void TMVA::DataLoader::AddTestEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight )
{
   AddEvent( className, Types::kTesting, event, weight );
}

////////////////////////////////////////////////////////////////////////////////
/// add event
/// vector event : the order of values is: variables + targets + spectators

void TMVA::DataLoader::AddEvent( const TString& className, Types::ETreeType tt,
                                 const std::vector<Double_t>& event, Double_t weight )
{
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

////////////////////////////////////////////////////////////////////////////////
///

Bool_t TMVA::DataLoader::UserAssignEvents(UInt_t clIndex)
{
   return fTrainAssignTree[clIndex]!=0;
}

////////////////////////////////////////////////////////////////////////////////
/// assign event-wise local trees to data set

void TMVA::DataLoader::SetInputTreesFromEventAssignTrees()
{
   UInt_t size = fTrainAssignTree.size();
   for(UInt_t i=0; i<size; i++) {
      if(!UserAssignEvents(i)) continue;
      const TString& className = DefaultDataSetInfo().GetClassInfo(i)->GetName();
      SetWeightExpression( "weight", className );
      AddTree(fTrainAssignTree[i], className, 1.0, TCut(""), Types::kTraining );
      AddTree(fTestAssignTree[i], className, 1.0, TCut(""), Types::kTesting );
   }
}

////////////////////////////////////////////////////////////////////////////////
/// number of signal events (used to compute significance)

void TMVA::DataLoader::AddTree( TTree* tree, const TString& className, Double_t weight,
                                const TCut& cut, const TString& treetype )
{
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

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::AddTree( TTree* tree, const TString& className, Double_t weight,
                                const TCut& cut, Types::ETreeType tt )
{
   if(!tree)
      Log() << kFATAL << "Tree does not exist (empty pointer)." << Endl;

   DefaultDataSetInfo().AddClass( className );

   // set analysistype to "kMulticlass" if more than two classes and analysistype == kNoAnalysisType
   if( fAnalysisType == Types::kNoAnalysisType && DefaultDataSetInfo().GetNClasses() > 2 )
      fAnalysisType = Types::kMulticlass;

   Log() << kINFO<< "Add Tree " << tree->GetName() << " of type " << className
         << " with " << tree->GetEntries() << " events" << Endl;
   DataInput().AddTree( tree, className, weight, cut, tt );
}

////////////////////////////////////////////////////////////////////////////////
/// number of signal events (used to compute significance)

void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////
/// add signal tree from text file

void TMVA::DataLoader::AddSignalTree( TString datFileS, Double_t weight, Types::ETreeType treetype )
{
   // create trees from these ascii files
   TTree* signalTree = new TTree( "TreeS", "Tree (S)" );
   signalTree->ReadFile( datFileS );

   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Signal file    : \""
         << datFileS << Endl;

   // number of signal events (used to compute significance)
   AddTree( signalTree, "Signal", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////
/// number of signal events (used to compute significance)

void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   AddTree( signal, "Background", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////
/// add background tree from text file

void TMVA::DataLoader::AddBackgroundTree( TString datFileB, Double_t weight, Types::ETreeType treetype )
{
   // create trees from these ascii files
   TTree* bkgTree = new TTree( "TreeB", "Tree (B)" );
   bkgTree->ReadFile( datFileB );

   Log() << kINFO << "Create TTree objects from ASCII input files ... \n- Background file    : \""
         << datFileB << Endl;

   // number of signal events (used to compute significance)
   AddTree( bkgTree, "Background", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Background", weight, TCut(""), treetype );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetSignalTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Signal", weight );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetBackgroundTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Background", weight );
}

////////////////////////////////////////////////////////////////////////////////
/// set background tree

void TMVA::DataLoader::SetTree( TTree* tree, const TString& className, Double_t weight )
{
   AddTree( tree, className, weight, TCut(""), Types::kMaxTreeType );
}

////////////////////////////////////////////////////////////////////////////////
/// define the input trees for signal and background; no cuts are applied

void  TMVA::DataLoader::SetInputTrees( TTree* signal, TTree* background,
                                       Double_t signalWeight, Double_t backgroundWeight )
{
   AddTree( signal,     "Signal",     signalWeight,     TCut(""), Types::kMaxTreeType );
   AddTree( background, "Background", backgroundWeight, TCut(""), Types::kMaxTreeType );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetInputTrees( const TString& datFileS, const TString& datFileB,
                                      Double_t signalWeight, Double_t backgroundWeight )
{
   DataInput().AddTree( datFileS, "Signal", signalWeight );
   DataInput().AddTree( datFileB, "Background", backgroundWeight );
}

////////////////////////////////////////////////////////////////////////////////
/// define the input trees for signal and background from single input tree,
/// containing both signal and background events distinguished by the type
/// identifiers: SigCut and BgCut

void TMVA::DataLoader::SetInputTrees( TTree* inputTree, const TCut& SigCut, const TCut& BgCut )
{
   AddTree( inputTree, "Signal",     1.0, SigCut, Types::kMaxTreeType );
   AddTree( inputTree, "Background", 1.0, BgCut , Types::kMaxTreeType );
}

////////////////////////////////////////////////////////////////////////////////
/// user inserts discriminating variable in data set info

void TMVA::DataLoader::AddVariable( const TString& expression, const TString& title, const TString& unit,
                                    char type, Double_t min, Double_t max )
{
   DefaultDataSetInfo().AddVariable( expression, title, unit, min, max, type );
}

////////////////////////////////////////////////////////////////////////////////
/// user inserts discriminating variable in data set info

void TMVA::DataLoader::AddVariable( const TString& expression, char type,
                                    Double_t min, Double_t max )
{
   DefaultDataSetInfo().AddVariable( expression, "", "", min, max, type );
}

////////////////////////////////////////////////////////////////////////////////
/// user inserts target in data set info

void TMVA::DataLoader::AddTarget( const TString& expression, const TString& title, const TString& unit,
                                  Double_t min, Double_t max )
{
   if( fAnalysisType == Types::kNoAnalysisType )
      fAnalysisType = Types::kRegression;

   DefaultDataSetInfo().AddTarget( expression, title, unit, min, max );
}

////////////////////////////////////////////////////////////////////////////////
/// user inserts target in data set info

void TMVA::DataLoader::AddSpectator( const TString& expression, const TString& title, const TString& unit,
                                     Double_t min, Double_t max )
{
   DefaultDataSetInfo().AddSpectator( expression, title, unit, min, max );
}

////////////////////////////////////////////////////////////////////////////////
/// default creation

TMVA::DataSetInfo& TMVA::DataLoader::DefaultDataSetInfo()
{
   return AddDataSet( fName );
}

////////////////////////////////////////////////////////////////////////////////
/// fill input variables in data set

void TMVA::DataLoader::SetInputVariables( std::vector<TString>* theVariables )
{
   for (std::vector<TString>::iterator it=theVariables->begin();
        it!=theVariables->end(); it++) AddVariable(*it);
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetSignalWeightExpression( const TString& variable)
{
   DefaultDataSetInfo().SetWeightExpression(variable, "Signal");
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetBackgroundWeightExpression( const TString& variable)
{
   DefaultDataSetInfo().SetWeightExpression(variable, "Background");
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetWeightExpression( const TString& variable, const TString& className )
{
   //Log() << kWarning << DefaultDataSetInfo().GetNClasses() /*fClasses.size()*/ << Endl;
   if (className=="") {
      SetSignalWeightExpression(variable);
      SetBackgroundWeightExpression(variable);
   }
   else  DefaultDataSetInfo().SetWeightExpression( variable, className );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetCut( const TString& cut, const TString& className ) {
   SetCut( TCut(cut), className );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::SetCut( const TCut& cut, const TString& className )
{
   DefaultDataSetInfo().SetCut( cut, className );
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::DataLoader::AddCut( const TString& cut, const TString& className )
{
   AddCut( TCut(cut), className );
}

////////////////////////////////////////////////////////////////////////////////
void TMVA::DataLoader::AddCut( const TCut& cut, const TString& className )
{
   DefaultDataSetInfo().AddCut( cut, className );
}

////////////////////////////////////////////////////////////////////////////////
/// prepare the training and test trees

void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut,
                                                   Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest,
                                                   const TString& otherOpt )
{
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:%s",
                                              NsigTrain, NbkgTrain, NsigTest, NbkgTest, otherOpt.Data()) );
}

////////////////////////////////////////////////////////////////////////////////
/// prepare the training and test trees
/// kept for backward compatibility

void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, Int_t Ntrain, Int_t Ntest )
{
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:SplitMode=Random:EqualTrainSample:!V",
                                              Ntrain, Ntrain, Ntest, Ntest) );
}

////////////////////////////////////////////////////////////////////////////////
/// prepare the training and test trees
/// -> same cuts for signal and background

void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, const TString& opt )
{
   SetInputTreesFromEventAssignTrees();

   DefaultDataSetInfo().PrintClasses();
   AddCut( cut );
   DefaultDataSetInfo().SetSplitOptions( opt );
}

////////////////////////////////////////////////////////////////////////////////
/// prepare the training and test trees

void TMVA::DataLoader::PrepareTrainingAndTestTree( TCut sigcut, TCut bkgcut, const TString& splitOpt )
{
   // if event-wise data assignment, add local trees to dataset first
   SetInputTreesFromEventAssignTrees();

   //Log() << kINFO <<"Preparing trees for training and testing..."<<  Endl;
   AddCut( sigcut, "Signal"  );
   AddCut( bkgcut, "Background" );

   DefaultDataSetInfo().SetSplitOptions( splitOpt );
}

////////////////////////////////////////////////////////////////////////////////
/// Function required to split the training and testing datasets into a
/// number of folds. Required by the CrossValidation and HyperParameterOptimisation
/// classes. The option to split the training dataset into a training set and
/// a validation set is implemented but not currently used.

void TMVA::DataLoader::MakeKFoldDataSet(UInt_t numberFolds, bool validationSet){
   // No need to do it again if the sets have already been split.
   if(fMakeFoldDataSet){
      Log() << kInfo << "Splitting in k-folds has been already done" << Endl;
      return;
   }

   fMakeFoldDataSet = kTRUE;

   // Get the original event vectors for testing and training from the dataset.
   const std::vector<Event*> TrainingData = DefaultDataSetInfo().GetDataSet()->GetEventCollection(Types::kTraining);
   const std::vector<Event*> TestingData = DefaultDataSetInfo().GetDataSet()->GetEventCollection(Types::kTesting);

   std::vector<Event*> TrainSigData;
   std::vector<Event*> TrainBkgData;
   std::vector<Event*> TestSigData;
   std::vector<Event*> TestBkgData;

   // Split the testing and training sets into signal and background classes.
   for(UInt_t i=0; i<TrainingData.size(); ++i){
      if( strncmp( DefaultDataSetInfo().GetClassInfo( TrainingData.at(i)->GetClass() )->GetName(), "Signal", 6) == 0){ TrainSigData.push_back(TrainingData.at(i)); }
      else if( strncmp( DefaultDataSetInfo().GetClassInfo( TrainingData.at(i)->GetClass() )->GetName(), "Background", 10) == 0){ TrainBkgData.push_back(TrainingData.at(i)); }
      else{
         Log() << kFATAL << "DataSets should only contain Signal and Background classes for classification, " << DefaultDataSetInfo().GetClassInfo( TrainingData.at(i)->GetClass() )->GetName() << " is not a recognised class" << Endl;
      }
   }

   for(UInt_t i=0; i<TestingData.size(); ++i){
      if( strncmp( DefaultDataSetInfo().GetClassInfo( TestingData.at(i)->GetClass() )->GetName(), "Signal", 6) == 0){ TestSigData.push_back(TestingData.at(i)); }
      else if( strncmp( DefaultDataSetInfo().GetClassInfo( TestingData.at(i)->GetClass() )->GetName(), "Background", 10) == 0){ TestBkgData.push_back(TestingData.at(i)); }
      else{
         Log() << kFATAL << "DataSets should only contain Signal and Background classes for classification, " << DefaultDataSetInfo().GetClassInfo( TestingData.at(i)->GetClass() )->GetName() << " is not a recognised class" << Endl;
      }
   }


   // Split the sets into the number of folds.
   if(validationSet){
      std::vector<std::vector<Event*>> tempSigEvents = SplitSets(TrainSigData,0,2);
      std::vector<std::vector<Event*>> tempBkgEvents = SplitSets(TrainBkgData,0,2);
      fTrainSigEvents = SplitSets(tempSigEvents.at(0),0,numberFolds);
      fTrainBkgEvents = SplitSets(tempBkgEvents.at(0),0,numberFolds);
      fValidSigEvents = SplitSets(tempSigEvents.at(1),0,numberFolds);
      fValidBkgEvents = SplitSets(tempBkgEvents.at(1),0,numberFolds);
   }
   else{
      fTrainSigEvents = SplitSets(TrainSigData,0,numberFolds);
      fTrainBkgEvents = SplitSets(TrainBkgData,0,numberFolds);
   }

   fTestSigEvents = SplitSets(TestSigData,0,numberFolds);
   fTestBkgEvents = SplitSets(TestBkgData,0,numberFolds);
}

////////////////////////////////////////////////////////////////////////////////
/// Function for assigning the correct folds to the testing or training set.

void TMVA::DataLoader::PrepareFoldDataSet(UInt_t foldNumber, Types::ETreeType tt){

   UInt_t numFolds = fTrainSigEvents.size();

   std::vector<Event*>* tempTrain = new std::vector<Event*>;
   std::vector<Event*>* tempTest = new std::vector<Event*>;

   UInt_t nTrain = 0;
   UInt_t nTest = 0;

   // Get the number of events so the memory can be reserved.
   for(UInt_t i=0; i<numFolds; ++i){
      if(tt == Types::kTraining){
         if(i!=foldNumber){
            nTrain += fTrainSigEvents.at(i).size();
            nTrain += fTrainBkgEvents.at(i).size();
         }
         else{
            nTest += fTrainSigEvents.at(i).size();
            nTest += fTrainSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kValidation){
         if(i!=foldNumber){
            nTrain += fValidSigEvents.at(i).size();
            nTrain += fValidBkgEvents.at(i).size();
         }
         else{
            nTest += fValidSigEvents.at(i).size();
            nTest += fValidSigEvents.at(i).size();
         }
      }
      else if(tt == Types::kTesting){
         if(i!=foldNumber){
            nTrain += fTestSigEvents.at(i).size();
            nTrain += fTestBkgEvents.at(i).size();
         }
         else{
            nTest += fTestSigEvents.at(i).size();
            nTest += fTestSigEvents.at(i).size();
         }
      }
   }

   // Reserve memory before filling vectors
   tempTrain->reserve(nTrain);
   tempTest->reserve(nTest);

   // Fill vectors with correct folds for testing and training.
   for(UInt_t j=0; j<numFolds; ++j){
      if(tt == Types::kTraining){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTrainSigEvents.at(j).begin(), fTrainSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTrainBkgEvents.at(j).begin(), fTrainBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kValidation){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fValidSigEvents.at(j).begin(), fValidSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fValidBkgEvents.at(j).begin(), fValidBkgEvents.at(j).end());
         }
      }
      else if(tt == Types::kTesting){
         if(j!=foldNumber){
            tempTrain->insert(tempTrain->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTrain->insert(tempTrain->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
         else{
            tempTest->insert(tempTest->end(), fTestSigEvents.at(j).begin(), fTestSigEvents.at(j).end());
            tempTest->insert(tempTest->end(), fTestBkgEvents.at(j).begin(), fTestBkgEvents.at(j).end());
         }
      }
   }

   // Assign the vectors of the events to rebuild the dataset
   DefaultDataSetInfo().GetDataSet()->SetEventCollection(tempTrain,Types::kTraining,false);
   DefaultDataSetInfo().GetDataSet()->SetEventCollection(tempTest,Types::kTesting,false);
   delete tempTest;
   delete tempTrain;
}

////////////////////////////////////////////////////////////////////////////////
/// Splits the input vector in to equally sized randomly sampled folds.

std::vector<std::vector<TMVA::Event*>> TMVA::DataLoader::SplitSets(std::vector<TMVA::Event*>& oldSet, int seedNum, int numFolds){

   ULong64_t nEntries = oldSet.size();
   ULong64_t foldSize = nEntries/numFolds;

   std::vector<std::vector<Event*>> tempSets;
   tempSets.resize(numFolds);

   TRandom3 r(seedNum);

   ULong64_t inSet = 0;

   for(ULong64_t i=0; i<nEntries; i++){
      bool inTree = false;
      if(inSet == foldSize*numFolds){
         break;
      }
      else{
         while(!inTree){
            int s = r.Integer(numFolds);
            if(tempSets.at(s).size()<foldSize){
               tempSets.at(s).push_back(oldSet.at(i));
               inSet++;
               inTree=true;
            }
         }
      }
   }

   return tempSets;

}

////////////////////////////////////////////////////////////////////////////////
/// Copy method use in VI and CV

TMVA::DataLoader* TMVA::DataLoader::MakeCopy(TString name)
{
   TMVA::DataLoader* des=new TMVA::DataLoader(name);
   DataLoaderCopy(des,this);
   return des;
}

////////////////////////////////////////////////////////////////////////////////
///Loading Dataset from DataInputHandler for subseed

void TMVA::DataLoaderCopy(TMVA::DataLoader* des, TMVA::DataLoader* src)
{
   for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Sbegin();treeinfo!=src->DataInput().Send();treeinfo++)
   {
      des->AddSignalTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
   }

   for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Bbegin();treeinfo!=src->DataInput().Bend();treeinfo++)
   {
      des->AddBackgroundTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// returns the correlation matrix of datasets

TH2* TMVA::DataLoader::GetCorrelationMatrix(const TString& className)
{
   const TMatrixD * m = DefaultDataSetInfo().CorrelationMatrix(className);
   return DefaultDataSetInfo().CreateCorrelationMatrixHist(m,
                                                           "CorrelationMatrix"+className, "Correlation Matrix ("+className+")");
}
