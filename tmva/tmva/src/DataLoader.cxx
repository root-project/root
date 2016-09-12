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

#include "TMVA/VariableIdentityTransform.h"
#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/VariablePCATransform.h"
#include "TMVA/VariableGaussTransform.h"
#include "TMVA/VariableNormalizeTransform.h"

#include "TMVA/ResultsClassification.h"
#include "TMVA/ResultsRegression.h"
#include "TMVA/ResultsMulticlass.h"


ClassImp(TMVA::DataLoader)


//_______________________________________________________________________
TMVA::DataLoader::DataLoader( TString thedlName)
: Configurable( ),
   fDataSetManager       ( NULL ), //DSMTEST
   fDataInputHandler     ( new DataInputHandler ),
   fTransformations      ( "I" ),
   fVerbose              ( kFALSE ),
   fDataAssignType       ( kAssignEvents ),
   fATreeEvent           (0)
{
   fDataSetManager = new DataSetManager( *fDataInputHandler ); // DSMTEST
   SetName(thedlName.Data());
}


//_______________________________________________________________________
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
   Tools::DestroyInstance();
   Config::DestroyInstance();
}


//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( DataSetInfo &dsi )
{
   return fDataSetManager->AddDataSetInfo(dsi); // DSMTEST
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::AddDataSet( const TString& dsiName )
{
   DataSetInfo* dsi = fDataSetManager->GetDataSetInfo(dsiName); // DSMTEST

   if (dsi!=0) return *dsi;
   
   return fDataSetManager->AddDataSetInfo(*(new DataSetInfo(dsiName))); // DSMTEST
}

// ________________________________________________
// the next functions are to assign events directly 

//_______________________________________________________________________
TTree* TMVA::DataLoader::CreateEventAssignTrees( const TString& name )
{
   // create the data assignment tree (for event-wise data assignment by user)
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

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Signal", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal testing event
   AddEvent( "Signal", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTrainingEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTestEvent( const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( "Background", Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTrainingEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal training event
   AddEvent( className, Types::kTraining, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTestEvent( const TString& className, const std::vector<Double_t>& event, Double_t weight ) 
{
   // add signal test event
   AddEvent( className, Types::kTesting, event, weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddEvent( const TString& className, Types::ETreeType tt,
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
Bool_t TMVA::DataLoader::UserAssignEvents(UInt_t clIndex) 
{
   // 
   return fTrainAssignTree[clIndex]!=0;
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTreesFromEventAssignTrees()
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
void TMVA::DataLoader::AddTree( TTree* tree, const TString& className, Double_t weight, 
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

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSignalTree( TString datFileS, Double_t weight, Types::ETreeType treetype )
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
void TMVA::DataLoader::AddSignalTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Signal", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // number of signal events (used to compute significance)
   AddTree( signal, "Background", weight, TCut(""), treetype );
}
//_______________________________________________________________________
void TMVA::DataLoader::AddBackgroundTree( TString datFileB, Double_t weight, Types::ETreeType treetype )
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
void TMVA::DataLoader::AddBackgroundTree( TTree* signal, Double_t weight, const TString& treetype )
{
   AddTree( signal, "Background", weight, TCut(""), treetype );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetSignalTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Signal", weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetBackgroundTree( TTree* tree, Double_t weight )
{
   AddTree( tree, "Background", weight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetTree( TTree* tree, const TString& className, Double_t weight )
{
   // set background tree
   AddTree( tree, className, weight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void  TMVA::DataLoader::SetInputTrees( TTree* signal, TTree* background, 
                                       Double_t signalWeight, Double_t backgroundWeight )
{
   // define the input trees for signal and background; no cuts are applied
   AddTree( signal,     "Signal",     signalWeight,     TCut(""), Types::kMaxTreeType );
   AddTree( background, "Background", backgroundWeight, TCut(""), Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTrees( const TString& datFileS, const TString& datFileB, 
                                      Double_t signalWeight, Double_t backgroundWeight )
{
   DataInput().AddTree( datFileS, "Signal", signalWeight );
   DataInput().AddTree( datFileB, "Background", backgroundWeight );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputTrees( TTree* inputTree, const TCut& SigCut, const TCut& BgCut )
{
   // define the input trees for signal and background from single input tree,
   // containing both signal and background events distinguished by the type 
   // identifiers: SigCut and BgCut
   AddTree( inputTree, "Signal",     1.0, SigCut, Types::kMaxTreeType );
   AddTree( inputTree, "Background", 1.0, BgCut , Types::kMaxTreeType );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddVariable( const TString& expression, const TString& title, const TString& unit, 
                                    char type, Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, title, unit, min, max, type ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddVariable( const TString& expression, char type,
                                    Double_t min, Double_t max )
{
   // user inserts discriminating variable in data set info
   DefaultDataSetInfo().AddVariable( expression, "", "", min, max, type ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddTarget( const TString& expression, const TString& title, const TString& unit, 
                                  Double_t min, Double_t max )
{
   // user inserts target in data set info

   if( fAnalysisType == Types::kNoAnalysisType )
      fAnalysisType = Types::kRegression;

   DefaultDataSetInfo().AddTarget( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
void TMVA::DataLoader::AddSpectator( const TString& expression, const TString& title, const TString& unit, 
                                     Double_t min, Double_t max )
{
   // user inserts target in data set info
   DefaultDataSetInfo().AddSpectator( expression, title, unit, min, max ); 
}

//_______________________________________________________________________
TMVA::DataSetInfo& TMVA::DataLoader::DefaultDataSetInfo() 
{ 
   // default creation
   return AddDataSet( fName );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetInputVariables( std::vector<TString>* theVariables ) 
{ 
   // fill input variables in data set
   for (std::vector<TString>::iterator it=theVariables->begin();
        it!=theVariables->end(); it++) AddVariable(*it);
}

//_______________________________________________________________________
void TMVA::DataLoader::SetSignalWeightExpression( const TString& variable)  
{ 
   DefaultDataSetInfo().SetWeightExpression(variable, "Signal"); 
}

//_______________________________________________________________________
void TMVA::DataLoader::SetBackgroundWeightExpression( const TString& variable) 
{
   DefaultDataSetInfo().SetWeightExpression(variable, "Background");
}

//_______________________________________________________________________
void TMVA::DataLoader::SetWeightExpression( const TString& variable, const TString& className )  
{
   //Log() << kWarning << DefaultDataSetInfo().GetNClasses() /*fClasses.size()*/ << Endl;
   if (className=="") {
      SetSignalWeightExpression(variable);
      SetBackgroundWeightExpression(variable);
   } 
   else  DefaultDataSetInfo().SetWeightExpression( variable, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetCut( const TString& cut, const TString& className ) {
   SetCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::DataLoader::SetCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().SetCut( cut, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddCut( const TString& cut, const TString& className ) 
{
   AddCut( TCut(cut), className );
}

//_______________________________________________________________________
void TMVA::DataLoader::AddCut( const TCut& cut, const TString& className ) 
{
   DefaultDataSetInfo().AddCut( cut, className );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, 
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
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, Int_t Ntrain, Int_t Ntest )
{
   // prepare the training and test trees 
   // kept for backward compatibility
   SetInputTreesFromEventAssignTrees();

   AddCut( cut  );

   DefaultDataSetInfo().SetSplitOptions( Form("nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:SplitMode=Random:EqualTrainSample:!V", 
                                              Ntrain, Ntrain, Ntest, Ntest) );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( const TCut& cut, const TString& opt )
{
   // prepare the training and test trees
   // -> same cuts for signal and background
   SetInputTreesFromEventAssignTrees();

   DefaultDataSetInfo().PrintClasses();
   AddCut( cut );
   DefaultDataSetInfo().SetSplitOptions( opt );
}

//_______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree( TCut sigcut, TCut bkgcut, const TString& splitOpt )
{
   // prepare the training and test trees

   // if event-wise data assignment, add local trees to dataset first
   SetInputTreesFromEventAssignTrees();

   //Log() << kINFO <<"Preparing trees for training and testing..."<<  Endl;
   AddCut( sigcut, "Signal"  );
   AddCut( bkgcut, "Background" );

   DefaultDataSetInfo().SetSplitOptions( splitOpt );
}

//______________________________________________________________________
void TMVA::DataLoader::PrepareTrainingAndTestTree(int foldNumber, Types::ETreeType tt)
{
  DataInput().ClearSignalTreeList();
  DataInput().ClearBackgroundTreeList();

  TString CrossValidate = "ParameterOpt";

  int numFolds = fTrainSigTree.size();

  for(int i=0; i<numFolds; ++i){
    if(CrossValidate == "PerformanceEst"){
      if(i!=foldNumber){
	AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	AddTree( fTestSigTree.at(i),      "Signal",     1.0,     TCut(""), Types::kTraining );
	AddTree( fTestBkgTree.at(i),      "Background", 1.0,     TCut(""), Types::kTraining );
      }
      else{
	AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	AddTree( fTestSigTree.at(i),      "Signal",     1.0,     TCut(""), Types::kTesting );
	AddTree( fTestBkgTree.at(i),      "Background", 1.0,     TCut(""), Types::kTesting );
      }
    }
    else if(CrossValidate == "ParameterOpt"){
      if(tt == Types::kTraining){
	if(i!=foldNumber){
	  AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	  AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	}
	else{
	  AddTree( fTrainSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	  AddTree( fTrainBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	}
      }
      else if(tt == Types::kTesting){
	if(i!=foldNumber){
	  AddTree( fTestSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTraining );
	  AddTree( fTestBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTraining );
	}
	else{
	  AddTree( fTestSigTree.at(i),     "Signal",     1.0,     TCut(""), Types::kTesting );
	  AddTree( fTestBkgTree.at(i),     "Background", 1.0,     TCut(""), Types::kTesting );
	}
      }
    }
  }

}

void TMVA::DataLoader::MakeKFoldDataSet(int numberFolds)
{
  
  UInt_t nSigTrees = DataInput().GetNSignalTrees();
  UInt_t nBkgTrees = DataInput().GetNBackgroundTrees();

  if(nSigTrees == 1){
    std::vector<TTree*> tempSigTrees = SplitSets(DataInput().SignalTreeInfo(0).GetTree(), 1, 2);
    fTrainSigTree = SplitSets(tempSigTrees.at(0), 0, numberFolds);
    fTestSigTree = SplitSets(tempSigTrees.at(1), 1, numberFolds);
  }
  if(nBkgTrees == 1){
    std::vector<TTree*> tempBkgTrees = SplitSets(DataInput().BackgroundTreeInfo(0).GetTree(), 1, 2);
    fTrainBkgTree = SplitSets(tempBkgTrees.at(0), 0, numberFolds);
    fTestBkgTree = SplitSets(tempBkgTrees.at(1), 1, numberFolds);
  }

  for(UInt_t i=0; i<nSigTrees; ++i){
    if(DataInput().SignalTreeInfo(i).GetTreeType() == Types::kTraining){
      fTrainSigTree = SplitSets(DataInput().SignalTreeInfo(i).GetTree(), i, numberFolds);
    }
    else if(DataInput().SignalTreeInfo(i).GetTreeType() == Types::kTesting){
      fTestSigTree = SplitSets(DataInput().SignalTreeInfo(i).GetTree(), i, numberFolds);
    }
  }
  for(UInt_t j=0; j<nBkgTrees; ++j){
    if(DataInput().BackgroundTreeInfo(j).GetTreeType() == Types::kTraining){
      fTrainBkgTree = SplitSets(DataInput().BackgroundTreeInfo(j).GetTree(), j, numberFolds);
    }
    else if(DataInput().BackgroundTreeInfo(j).GetTreeType() == Types::kTesting){
      fTestBkgTree = SplitSets(DataInput().BackgroundTreeInfo(j).GetTree(), j, numberFolds);
    }
  }

  DataInput().ClearSignalTreeList();
  DataInput().ClearBackgroundTreeList();

  nSigTrees = DataInput().GetNSignalTrees();
  nBkgTrees = DataInput().GetNBackgroundTrees();

}

void TMVA::DataLoader::ValidationKFoldSet(){
  DefaultDataSetInfo().GetDataSet()->DivideTrainingSet(2);
  DefaultDataSetInfo().GetDataSet()->MoveTrainingBlock(1, Types::kValidation, kTRUE);
}

std::vector<TTree*> TMVA::DataLoader::SplitSets(TTree * oldTree, int seedNum, int numFolds)
{
  std::vector<TTree*> tempTrees;

  for(int l=0; l<numFolds; ++l){
    tempTrees.push_back(oldTree->CloneTree(0));
    tempTrees.at(l)->SetDirectory(0);
  }

  TRandom3 r(seedNum);

  Long64_t nEntries = oldTree->GetEntries();

  std::vector<TBranch*> branches;

  //TBranch * typeBranch = oldTree->GetBranch("type");
  //branches.push_back(typeBranch);
  //oldTree->SetBranchAddress( "type",   &fATreeType);
  //TBranch * weightBranch = oldTree->GetBranch("weight");
  //branches.push_back(weightBranch);
  //oldTree->SetBranchAddress( "weight", &fATreeWeight);

  std::vector<VariableInfo>& vars = DefaultDataSetInfo().GetVariableInfos();
  std::vector<VariableInfo>& tgts = DefaultDataSetInfo().GetTargetInfos();
  std::vector<VariableInfo>& spec = DefaultDataSetInfo().GetSpectatorInfos();

  UInt_t varsSize = vars.size();

  if (fATreeEvent.size()==0) fATreeEvent.resize(vars.size()+tgts.size()+spec.size());
  // add variables
  for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
    TString vname = vars[ivar].GetExpression();
    if(vars[ivar].GetExpression() != vars[ivar].GetLabel()){
      varsSize--;
      continue; 
    }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress(vname, &fATreeEvent[ivar]);
  }
  // add targets
  for (UInt_t itgt=0; itgt<tgts.size(); itgt++) {
    TString vname = tgts[itgt].GetExpression();
    if(tgts[itgt].GetExpression() != tgts[itgt].GetLabel()){ continue; }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress( vname, &fATreeEvent[vars.size()+itgt]);
  }
  // add spectators
  for (UInt_t ispc=0; ispc<spec.size(); ispc++) {
    TString vname = spec[ispc].GetExpression();
    if(spec[ispc].GetExpression() != spec[ispc].GetLabel()){ continue; }
    TBranch * branch = oldTree->GetBranch(vname);
    branches.push_back(branch);
    oldTree->SetBranchAddress( vname, &fATreeEvent[vars.size()+tgts.size()+ispc]);
  }

  Long64_t foldSize = nEntries/numFolds;
  Long64_t inSet = 0;

  for(Long64_t i=0; i<nEntries; i++){
    for(UInt_t j=0; j<vars.size(); j++){ fATreeEvent[j]=0; }
    oldTree->GetEvent(i);
    bool inTree = false;
    if(inSet == foldSize*numFolds){
      break;
    }
    else{
      while(!inTree){
	int s = r.Integer(numFolds);
	if(tempTrees.at(s)->GetEntries()<foldSize){
	  tempTrees.at(s)->Fill();
	  inSet++;
	  inTree=true;
	}
      }
    }
  }

  return tempTrees;
}

//_______________________________________________________________________
//Copy method use in VI and CV
TMVA::DataLoader* TMVA::DataLoader::MakeCopy(TString name)
{
    TMVA::DataLoader* des=new TMVA::DataLoader(name);
    DataLoaderCopy(des,this);
    return des;
}

//_______________________________________________________________________
void TMVA::DataLoaderCopy(TMVA::DataLoader* des, TMVA::DataLoader* src)
{
    //Loading Dataset from DataInputHandler for subseed
    for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Sbegin();treeinfo!=src->DataInput().Send();treeinfo++)
    {
      des->AddSignalTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
    }

    for( std::vector<TreeInfo>::const_iterator treeinfo=src->DataInput().Bbegin();treeinfo!=src->DataInput().Bend();treeinfo++)
    {
      des->AddBackgroundTree( (*treeinfo).GetTree(), (*treeinfo).GetWeight(),(*treeinfo).GetTreeType());
    }
}



