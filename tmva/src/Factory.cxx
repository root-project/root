// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

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
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//________________________________________________________________________
/*
  Factory
  
  This is the main MVA steering class: it creates all MVA methods, and
  guides them through the training, testing and evaluation phases
*/
//________________________________________________________________________

#ifndef ROOT_TMVA_Factory
#include "TMVA/Factory.h"
#endif

#include "Riostream.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TEventList.h"
#include "TH1.h"
#include "TH2.h"
#include "TText.h"
#include "TTreeFormula.h"
#include "TStyle.h"
#include "TMatrixF.h"
#include "TMatrixDSym.h"
#include "TPaletteAxis.h"
#include "TPrincipal.h"
#include "TPluginManager.h"

#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/DataSet.h"
#include "TMVA/Methods.h"
#include "TMVA/Methods.h"

const Bool_t DEBUG_TMVA_Factory = kFALSE;

const int MinNoTrainingEvents = 10;
const int MinNoTestEvents     = 1;
const long int basketsize     = 1280000;

ClassImp(TMVA::Factory)

//_______________________________________________________________________
TMVA::Factory::Factory( TString jobName, TFile* theTargetFile, TString theOption )
   : Configurable           ( theOption ),
     fDataSet               ( new DataSet ),
     fTargetFile            ( theTargetFile ),
     fVerbose               ( kFALSE ),
     fJobName               ( jobName ),
     fDataAssignType        ( kUndefined ),
     fSuspendDATVerification( kFALSE ),
     fTrainSigAssignTree    ( 0 ),
     fTrainBkgAssignTree    ( 0 ),
     fTestSigAssignTree     ( 0 ),
     fTestBkgAssignTree     ( 0 ),
     fATreeEvent            ( 0 )
{  
   // standard constructor
   //   jobname       : this name will appear in all weight file names produced by the MVAs
   //   theTargetFile : output ROOT file; the test tree and all evaluation plots 
   //                   will be stored here
   //   theOption     : option string; currently: "V" for verbose

   // init configurable
   SetConfigDescription( "Configuration options for factory running" );

   // histograms are not automatically associated with the current
   // directory and hence don't go out of scope when closing the file
   // TH1::AddDirectory(kFALSE);
   Bool_t silent = kFALSE;
   Bool_t color  = !gROOT->IsBatch();
   SetConfigName( GetName() );
   DeclareOptionRef( fVerbose, "V", "Verbose mode" );
   DeclareOptionRef( color, "Color", "Color flag (default on)" );
   DeclareOptionRef( silent, "Silent", "Batch mode: boolean silent flag inhibiting any output from TMVA after the creation of the factory class object (default: False)" );

   ParseOptions( kFALSE );

   if (Verbose()) fLogger.SetMinType( kVERBOSE );

   gConfig().SetUseColor( color );
   gConfig().SetSilent( silent );

   Greetings();

   Data().SetBaseRootDir ( fTargetFile );
   Data().SetLocalRootDir( fTargetFile );
   Data().SetVerbose     ( Verbose() );
}

//_______________________________________________________________________
void TMVA::Factory::Greetings() 
{
   // print welcome message
   // options are: kLogoWelcomeMsg, kIsometricWelcomeMsg, kLeanWelcomeMsg

   gTools().ROOTVersionMessage( fLogger ); 
   gTools().TMVAWelcomeMessage( fLogger, gTools().kLogoWelcomeMsg );
   gTools().TMVAVersionMessage( fLogger ); fLogger << Endl;
}

//_______________________________________________________________________
TMVA::Factory::~Factory( void )
{
   // default destructor
   this->DeleteAllMethods();

   if (fTrainSigAssignTree) { delete fTrainSigAssignTree; fTrainSigAssignTree = 0; }
   if (fTrainBkgAssignTree) { delete fTrainBkgAssignTree; fTrainBkgAssignTree = 0; }
   if (fTestSigAssignTree)  { delete fTestSigAssignTree; fTestSigAssignTree = 0; }
   if (fTestBkgAssignTree)  { delete fTestBkgAssignTree; fTestBkgAssignTree = 0; }

   if ( fDataSet ) { delete fDataSet; fDataSet = 0; }
}

//_______________________________________________________________________
void TMVA::Factory::DeleteAllMethods( void )
{
   // delete methods
   vector<IMethod*>::iterator itrMethod = fMethods.begin();
   for (; itrMethod != fMethods.end(); itrMethod++) {
      fLogger << kVERBOSE << "Delete method: " << (*itrMethod)->GetName() << Endl;    
      delete (*itrMethod);
   }
   fMethods.clear();
}

//_______________________________________________________________________
void TMVA::Factory::SetInputVariables( vector<TString>* theVariables ) 
{ 
   // fill input variables in data set

   // sanity check
   if (theVariables->size() == 0) {
      fLogger << kFATAL << "<SetInputVariables> Vector of input variables is empty" << Endl;
   }

   for (UInt_t i=0; i<theVariables->size(); i++) Data().AddVariable((*theVariables)[i]);
}

//_______________________________________________________________________
Bool_t TMVA::Factory::VerifyDataAssignType( DataAssignType thisType )
{
   // sanity check to enforce exclusive usage of data assignment (either event-wise OR tree-wise)

   // is verification suspended ?
   if (fSuspendDATVerification) { fSuspendDATVerification = kFALSE; return kTRUE; }

   if      (thisType        == kUndefined) fLogger << kFATAL << "Big troubles in \"VerifyDataAssignType\"" << Endl;
   else if (fDataAssignType == kUndefined) fDataAssignType = thisType;
   else if (thisType        != fDataAssignType) {
      fLogger << kINFO 
              << "Given flags: thisType = " << thisType << ", DataAssignType = " << fDataAssignType << Endl;
      fLogger << kFATAL  
              << "You have added individual training or test events to the factory AND "
              << "also added complete trees. This is incompatible in TMVA -> please use " 
              << "the one OR the other (or contact the authors if such mixed access is required)." << Endl;
      return kFALSE;
   }
   
   return kTRUE;
}

//_______________________________________________________________________
void TMVA::Factory::CreateEventAssignTrees( TTree*& fAssignTree, const TString& name )
{
   // create the data assignment tree (for event-wise data assignment by user)
   fAssignTree = new TTree( name, name );
   fAssignTree->Branch( "type",   &fATreeType,   "ATreeType/I" );
   fAssignTree->Branch( "weight", &fATreeWeight, "ATreeWeight/I" );
   std::vector<VariableInfo>& vars = Data().GetVariableInfos();
   if (!fATreeEvent) fATreeEvent = new Float_t[vars.size()];
   for (UInt_t ivar=0; ivar<vars.size(); ivar++) {
      TString vname = vars[ivar].GetExpression();
      fAssignTree->Branch( vname, &(fATreeEvent[ivar]), vname + "/F" );
   }
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTrainingEvent( std::vector<Double_t>& event, Double_t weight )
{
   // add signal training event
   VerifyDataAssignType( kAssignEvents );

   // first call -> create tree !
   if (!fTrainSigAssignTree) CreateEventAssignTrees( fTrainSigAssignTree, "LocalSigTrainTree" );
   
   // sanity check
   if (event.size() != (UInt_t)fTrainSigAssignTree->GetNbranches()-2 || 
       event.size() != Data().GetVariableInfos().size()) {
      fLogger << kFATAL 
              << "Dimension mismatch in \"AddSignalTrainingEvent\", does the "
              << "input vector contain as many dimenions as variables defined ? => please check" << Endl;
   }
   
   // fill tree
   fATreeType   = 1;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];
   fTrainSigAssignTree->Fill();
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTrainingEvent( std::vector<Double_t>& event, Double_t weight )
{
   // add background training event
   VerifyDataAssignType( kAssignEvents );

   // first call -> create tree !
   if (!fTrainBkgAssignTree) CreateEventAssignTrees( fTrainBkgAssignTree, "LocalBkgTraiTree" );

   // sanity check
   if (event.size() != (UInt_t)fTrainBkgAssignTree->GetNbranches()-2 || 
       event.size() != Data().GetVariableInfos().size()) {
      fLogger << kFATAL 
              << "Dimension mismatch in \"AddBackgroundTrainingEvent\", does the "
              << "input vector contain as many dimenions as variables defined ? => please check" << Endl;
   }
   
   // fill tree
   fATreeType   = 0;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];
   fTrainBkgAssignTree->Fill();
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTestEvent( std::vector<Double_t>& event, Double_t weight )
{
   // add signal test event
   VerifyDataAssignType( kAssignEvents );

   // first call -> create tree !
   if (!fTestSigAssignTree) CreateEventAssignTrees( fTestSigAssignTree, "LocalSigTestTree" );

   // sanity check
   if (event.size() != (UInt_t)fTestSigAssignTree->GetNbranches()-2 || 
       event.size() != Data().GetVariableInfos().size()) {
      fLogger << kFATAL 
              << "Dimension mismatch in \"AddSignalTestEvent\", does the "
              << "input vector contain as many dimenions as variables defined ? => please check" << Endl;
   }
   
   // fill tree
   fATreeType   = 1;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];
   fTestSigAssignTree->Fill();
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTestEvent( std::vector<Double_t>& event, Double_t weight )
{
   // add background test event
   VerifyDataAssignType( kAssignEvents );

   // first call -> create tree !
   if (!fTestBkgAssignTree) CreateEventAssignTrees( fTestBkgAssignTree, "LocalBkgTestTree" );

   // sanity check
   if (event.size() != (UInt_t)fTestBkgAssignTree->GetNbranches()-2 || 
       event.size() != Data().GetVariableInfos().size()) {
      fLogger << kFATAL 
              << "Dimension mismatch in \"AddBackgroundTestEvent\", does the "
              << "input vector contain as many dimenions as variables defined ? => please check" << Endl;
   }
   
   // fill tree
   fATreeType   = 0;
   fATreeWeight = weight;
   for (UInt_t ivar=0; ivar<event.size(); ivar++) fATreeEvent[ivar] = event[ivar];
   fTestBkgAssignTree->Fill();
}

//_______________________________________________________________________
Bool_t TMVA::Factory::SetInputTrees( TTree* signal, TTree* background, 
                                     Double_t signalWeight, Double_t backgroundWeight )
{
   // define the input trees for signal and background; no cuts are applied
   if (!signal || !background) {
      fLogger << kFATAL << "Zero pointer for signal and/or background tree: " 
              << signal << " " << background << Endl;
      return kFALSE;
   }

   SetSignalTree    ( signal,     signalWeight );
   SetBackgroundTree( background, backgroundWeight );
   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::Factory::SetInputTrees( TTree* inputTree, TCut SigCut, TCut BgCut )
{
   // define the input trees for signal and background from single input tree,
   // containing both signal and background events distinguished by the type 
   // identifiers: SigCut and BgCut
   if (!inputTree) {
      fLogger << kFATAL << "Zero pointer for input tree: " << inputTree << Endl;
      return kFALSE;
   }

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   TTree* signalTree = inputTree->CloneTree(0);
   TTree* backgTree  = inputTree->CloneTree(0);
   
   TIter next_branch1( signalTree->GetListOfBranches() );
   while (TBranch *branch = (TBranch*)next_branch1())
      branch->SetBasketSize(basketsize);

   TIter next_branch2( backgTree->GetListOfBranches() );
   while (TBranch *branch = (TBranch*)next_branch2())
      branch->SetBasketSize(basketsize);

   inputTree->Draw(">>signalList",SigCut,"goff");
   TEventList *signalList = (TEventList*)gDirectory->Get("signalList");

   inputTree->Draw(">>backgList",BgCut,"goff");    
   TEventList *backgList = (TEventList*)gDirectory->Get("backgList");

   if (backgList->GetN() == inputTree->GetEntries()) {
      TCut bgcut= !SigCut;
      inputTree->Draw(">>backgList",bgcut,"goff");    
      backgList = (TEventList*)gDirectory->Get("backgList");
   }
   signalList->Print();
   backgList ->Print();
  
   for (Int_t i=0;i<inputTree->GetEntries(); i++) {
      inputTree->GetEntry(i);
      if ((backgList->Contains(i)) && (signalList->Contains(i))) {
         fLogger << kWARNING << "Event " << i
                 << " is selected for signal and background sample! Skip it!" << Endl;
         continue;
      }
      if (signalList->Contains(i)) signalTree->Fill();
      if (backgList->Contains(i) ) backgTree->Fill();
   }

   signalTree->ResetBranchAddresses();
   backgTree ->ResetBranchAddresses();


   Data().AddSignalTree    ( signalTree, 1.0 );
   Data().AddBackgroundTree( backgTree,  1.0 );

   delete signalList;
   delete backgList;
   return kTRUE;    
}

//_______________________________________________________________________
Bool_t TMVA::Factory::SetInputTrees( TString datFileS, TString datFileB, 
                                     Double_t signalWeight, Double_t backgroundWeight )
{
   // create trees from these ascii files
   TTree* signalTree = new TTree( "TreeS", "Tree (S)" );
   TTree* backgTree  = new TTree( "TreeB", "Tree (B)" );

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );
  
   signalTree->ReadFile( datFileS );
   backgTree->ReadFile( datFileB );

   fLogger << kINFO << "Create TTree objects from ASCII input files ... \n- Signal file    : \""
           << datFileS << "\" \n- Background file: \""
           << datFileB << "\"" << Endl;
  
   ifstream in; 
   in.open(datFileS);
   if (!in.good()) {
      fLogger << kFATAL << "Could not open file: " << datFileS << Endl;
      return kFALSE;
   }
   in.close();
   in.open(datFileB);
   if (!in.good()) {
      fLogger << kFATAL << "Could not open file: " << datFileB << Endl;
      return kFALSE;
   }
   in.close();
    
   //signalTree->Write();
   //backgTree ->Write();

   SetSignalTree    ( signalTree, signalWeight );
   SetBackgroundTree( backgTree,  backgroundWeight );

   return kTRUE;
}

//_______________________________________________________________________
void TMVA::Factory::SetInputTreesFromEventAssignTrees()
{
   // assign event-wise local trees to data set
   SetWeightExpression( "weight" );

   fSuspendDATVerification = kTRUE; AddSignalTree    ( fTrainSigAssignTree, 1.0, Types::kTraining );
   fSuspendDATVerification = kTRUE; AddSignalTree    ( fTestSigAssignTree,  1.0, Types::kTesting  );
   fSuspendDATVerification = kTRUE; AddBackgroundTree( fTrainBkgAssignTree, 1.0, Types::kTraining );
   fSuspendDATVerification = kTRUE; AddBackgroundTree( fTestBkgAssignTree,  1.0, Types::kTesting  );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, 
                                                Int_t NsigTrain, Int_t NbkgTrain, Int_t NsigTest, Int_t NbkgTest,
                                                const TString& otherOpt )
{
   // prepare the training and test trees
   TString opt(Form("NsigTrain=%i:NbkgTrain=%i:NsigTest=%i:NbkgTest=%i:%s", 
                    NsigTrain, NbkgTrain, NsigTest, NbkgTest, otherOpt.Data()));
   PrepareTrainingAndTestTree( cut, cut, opt );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, Int_t Ntrain, Int_t Ntest )
{
   // prepare the training and test trees 
   // kept for backward compatibility
   TString opt(Form("NsigTrain=%i:NbkgTrain=%i:NsigTest=%i:NbkgTest=%i:SplitMode=Random:EqualTrainSample:!V", 
                    Ntrain, Ntrain, Ntest, Ntest));
   PrepareTrainingAndTestTree( cut, cut, opt );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& cut, const TString& splitOpt )
{ 
   // prepare the training and test trees
   PrepareTrainingAndTestTree( cut, cut, splitOpt );
}

//_______________________________________________________________________
void TMVA::Factory::PrepareTrainingAndTestTree( const TCut& sigcut, const TCut& bkgcut, const TString& splitOpt )
{ 
   // prepare the training and test trees

   // if event-wise data assignment, add local trees to dataset first
   if (fDataAssignType == kAssignEvents) SetInputTreesFromEventAssignTrees();

   fLogger << kINFO << "Preparing trees for training and testing..." << Endl;
   Data().SetCuts(sigcut, bkgcut); // different cuts for signal and background

   Data().PrepareForTrainingAndTesting( splitOpt );
}

//_______________________________________________________________________
void TMVA::Factory::SetSignalTree( TTree* signal, Double_t weight )
{
   // set signal tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   // number of signal events (used to compute significance)
   Data().ClearSignalTreeList();
   AddSignalTree( signal, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTree( TTree* signal, Double_t weight, Types::ETreeType treetype )
{
   // add signal tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   // number of signal events (used to compute significance)
   Data().AddSignalTree( signal, weight, treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddSignalTree( TTree* signal, Double_t weight, const TString& treetype )
{
   // add signal tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   // number of signal events (used to compute significance)
   Types::ETreeType tt = Types::kMaxTreeType;
   TString tmpTreeType = treetype; tmpTreeType.ToLower();
   if      (tmpTreeType.Contains( "train" ) && tmpTreeType.Contains( "test" )) tt = Types::kMaxTreeType;
   else if (tmpTreeType.Contains( "train" ))                                   tt = Types::kTraining;
   else if (tmpTreeType.Contains( "test" ))                                    tt = Types::kTesting;
   else {
      fLogger << kFATAL << "<AddSignalTree> cannot interpret tree type: \"" << treetype 
              << "\" should be \"Training\" or \"Test\" or \"Training and Testing\"" << Endl;
   }

   Data().AddSignalTree( signal, weight, tt );
}

//_______________________________________________________________________
void TMVA::Factory::SetBackgroundTree( TTree* background, Double_t weight )
{
   // set background tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   // number of background events (used to compute significance)
   Data().ClearBackgroundTreeList();
   AddBackgroundTree( background, weight );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTree( TTree* background, Double_t weight, Types::ETreeType treetype )
{
   // add background tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );

   // number of background events (used to compute significance)
   Data().AddBackgroundTree( background, weight, treetype );
}

//_______________________________________________________________________
void TMVA::Factory::AddBackgroundTree( TTree* background, Double_t weight, const TString & treetype )
{
   // add background tree

   // sanity check that we are now using tree assignment as opposed to event assignment
   VerifyDataAssignType( kAssignTrees );
   
   // number of background events (used to compute significance)
   Types::ETreeType tt = Types::kMaxTreeType;
   TString tmpTreeType = treetype; tmpTreeType.ToLower();
   if      (tmpTreeType.Contains( "train" ) && tmpTreeType.Contains( "test" )) tt = Types::kMaxTreeType;
   else if (tmpTreeType.Contains( "train" ))                                   tt = Types::kTraining;
   else if (tmpTreeType.Contains( "test" ))                                    tt = Types::kTesting;
   else {
      fLogger << kFATAL << "<AddBackgroundTree> cannot interpret tree type: \"" << treetype 
              << "\" should be \"Training\" or \"Test\" or \"Training and Testing\"" << Endl;
   }

   Data().AddBackgroundTree( background, weight, tt );
}

//_______________________________________________________________________
Bool_t TMVA::Factory::BookMethod( TString theMethodName, TString methodTitle, TString theOption ) 
{
   // booking via name; the names are translated into enums and the 
   // corresponding overloaded BookMethod is called

   if (theMethodName != "Variable") fLogger << kINFO << "Create method: " << theMethodName << Endl;

   return BookMethod( Types::Instance().GetMethodType( theMethodName ), methodTitle, theOption );

   return kFALSE;
}

//_______________________________________________________________________
Bool_t TMVA::Factory::BookMethod( Types::EMVA theMethod, TString methodTitle, TString theOption ) 
{
   // books MVA method; the option configuration string is custom for each MVA
   // the TString field "theNameAppendix" serves to define (and distringuish) 
   // several instances of a given MVA, eg, when one wants to compare the 
   // performance of various configurations
   if (GetMethod( methodTitle ) != 0) {
      fLogger << kFATAL << "Booking failed since method with title <"
              << methodTitle <<"> already exists"
              << Endl;
   }

   TPluginManager *pluginManager(0);
   TPluginHandler *pluginHandler(0);

   // initialize methods
   MethodBase *method = 0;

   switch (theMethod) {
   case Types::kCuts:       
      method = new MethodCuts           ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kFisher:     
      method = new MethodFisher         ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kKNN:     
      method = new MethodKNN            ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kMLP:        
      method = new MethodMLP            ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kTMlpANN:    
      method = new MethodTMlpANN        ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kCFMlpANN:   
      method = new MethodCFMlpANN       ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kLikelihood: 
      method = new MethodLikelihood     ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kVariable:   
      method = new MethodVariable       ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kHMatrix:    
      method = new MethodHMatrix        ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kPDERS:      
      method = new MethodPDERS          ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kBDT:        
      method = new MethodBDT            ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kSVM:        
      method = new MethodSVM            ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kRuleFit:    
      method = new MethodRuleFit        ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kBayesClassifier:    
      method = new MethodBayesClassifier( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kFDA:    
      method = new MethodFDA            ( fJobName, methodTitle, Data(), theOption ); break;
   case Types::kSeedDistance:    
      method = new MethodSeedDistance   ( fJobName, methodTitle, Data(), theOption ); break;
   case TMVA::Types::kPlugins:
      pluginManager = gROOT->GetPluginManager();
      pluginHandler = pluginManager->FindHandler("TMVA@@MethodBase", methodTitle);
      if (pluginHandler) {
         if (pluginHandler->LoadPlugin() == 0) {
            method = (TMVA::MethodBase*) pluginHandler->ExecPlugin(4, &fJobName, &methodTitle, &Data(), &theOption);
            if(method==0) {
               fLogger << kFATAL << "Couldn't instantiate plugin for " << methodTitle << "." << Endl;
            } else {
               fLogger << kINFO << "Found plugin for " << methodTitle << "  " << method << Endl;
            }
         } else {
            fLogger << kFATAL << "Couldn't load any plugin for " << methodTitle << "." << Endl; 
         }
      } else {
         fLogger << kFATAL << "Couldn't find plugin handler for TMVA@@MethodBase and " << methodTitle << Endl; 
      }
      break;
   default:
      fLogger << kFATAL << "Method: \"" << theMethod << "\" does not exist" << Endl;
   }

   fLogger << kINFO << "Booking method: " << method->GetMethodTitle() << Endl;
   fMethods.push_back( method );

   return kTRUE;
}

//_______________________________________________________________________
Bool_t TMVA::Factory::BookMethod( Types::EMVA theMethod, TString methodTitle, TString methodOption,
                                  Types::EMVA theCommittee, TString committeeOption ) 
{
   // books MVA method; the option configuration string is custom for each MVA
   // the TString field "theNameAppendix" serves to define (and distringuish) 
   // several instances of a given MVA, eg, when one wants to compare the 
   // performance of various configurations

   IMethod *method = 0;

   // initialize methods
   switch(theMethod) {
   case Types::kCommittee:    
      method = new MethodCommittee( fJobName, methodTitle, Data(), methodOption, theCommittee, committeeOption ); break;
   default:
      fLogger << kFATAL << "Method: " << theMethod << " does not exist" << Endl;
   }

   fMethods.push_back( method );

   return kTRUE;
}

//_______________________________________________________________________
TMVA::IMethod* TMVA::Factory::GetMethod( const TString &methodTitle ) const
{
   // returns pointer to MVA that corresponds to given method title
   vector<IMethod*>::const_iterator itrMethod    = fMethods.begin();
   vector<IMethod*>::const_iterator itrMethodEnd = fMethods.end();
   //
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      MethodBase* mva = (MethodBase*)(*itrMethod);    
      if ( (mva->GetMethodTitle())==methodTitle ) return mva;
   }
   return 0;
}

//_______________________________________________________________________
void TMVA::Factory::TrainAllMethods( void ) 
{  
   // iterates over all MVAs that have been booked, and calls their training methods
   fLogger << kINFO << "Training all methods..." << Endl;

   // iterate over methods and train
   vector<IMethod*>::iterator itrMethod    = fMethods.begin();
   vector<IMethod*>::iterator itrMethodEnd = fMethods.end();
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      MethodBase* mva = (MethodBase*)*itrMethod;
      if (Data().GetTrainingTree()->GetEntries() >= MinNoTrainingEvents) {
         fLogger << kINFO << "Train method: " << mva->GetMethodTitle() << Endl;
         mva->TrainMethod();
      }
      else {
         fLogger << kWARNING << "Method " << mva->GetMethodName() 
                 << " not trained (training tree has less entries ["
                 << Data().GetTrainingTree()->GetEntries() 
                 << "] than required [" << MinNoTrainingEvents << "]" << Endl; 
      }
   }

   // variable ranking 
   fLogger << Endl;
   fLogger << kINFO << "Begin ranking of input variables..." << Endl;
   for (itrMethod = fMethods.begin(); itrMethod != itrMethodEnd; itrMethod++) {
      if (Data().GetTrainingTree()->GetEntries() >= MinNoTrainingEvents) {

         // create and print ranking
         const Ranking* ranking = (*itrMethod)->CreateRanking();
         if (ranking != 0) ranking->Print();
         else fLogger << kINFO << "No variable ranking supplied by classifier: " 
                      << ((MethodBase*)*itrMethod)->GetMethodTitle() << Endl;
      }
   }
   fLogger << Endl;
}

//_______________________________________________________________________
void TMVA::Factory::TestAllMethods( void )
{
   // iterates over all MVAs that have been booked, and calls their testing methods
   fLogger << kINFO << "Testing all classifiers..." << Endl;

   if (Data().GetTrainingTree() == NULL) {
      fLogger << kWARNING << "You perform testing without training before, hope you "
              << "provided a reasonable test tree and weight files " << Endl;
   } 

   // iterate over methods and test
   vector<IMethod*>::iterator itrMethod    = fMethods.begin();
   vector<IMethod*>::iterator itrMethodEnd = fMethods.end();
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      MethodBase* mva = (MethodBase*)*itrMethod;
      fLogger << kINFO << "Test method: " << mva->GetMethodTitle() << Endl;
      mva->AddClassifierToTestTree(0);
   }
   if (DEBUG_TMVA_Factory) Data().GetTestTree()->Print();
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
         fLogger << kWARNING << "<MakeClass> Could not find classifier \"" << methodTitle 
                 << "\" in list" << Endl;
      }
   }
   else {

      // no classifier specified, print all hepl messages
      vector<IMethod*>::const_iterator itrMethod    = fMethods.begin();
      vector<IMethod*>::const_iterator itrMethodEnd = fMethods.end();
      for (; itrMethod != itrMethodEnd; itrMethod++) {
         MethodBase* method = (MethodBase*)*itrMethod;
         fLogger << kINFO << "Make response class for classifier: " << method->GetMethodTitle() << Endl;
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
         fLogger << kWARNING << "<PrintHelpMessage> Could not find classifier \"" << methodTitle 
                 << "\" in list" << Endl;
      }
   }
   else {

      // no classifier specified, print all hepl messages
      vector<IMethod*>::const_iterator itrMethod    = fMethods.begin();
      vector<IMethod*>::const_iterator itrMethodEnd = fMethods.end();
      for (; itrMethod != itrMethodEnd; itrMethod++) {
         MethodBase* method = (MethodBase*)*itrMethod;
         fLogger << kINFO << "Print help message for classifier: " << method->GetMethodTitle() << Endl;
         method->PrintHelpMessage();
      }
   }
}

//_______________________________________________________________________
void TMVA::Factory::EvaluateAllVariables( TString options )
{
   // iterates over all MVA input varables and evaluates them
   fLogger << kINFO << "Evaluating all variables..." << Endl;

   if (Data().GetTrainingTree() == NULL) {
      fLogger << kWARNING << "You perform testing without training before, hope you "
              << "provided a reasonable test tree and weight files " << Endl;
   } 

   for (UInt_t i=0; i<Data().GetNVariables(); i++) {
      TString s = Data().GetInternalVarName(i);
      if (options.Contains("V")) s += ":V";
      this->BookMethod( "Variable", s );
   }
}

//_______________________________________________________________________
void TMVA::Factory::EvaluateAllMethods( void )
{
   // iterates over all MVAs that have been booked, and calls their evaluation methods

   fLogger << kINFO << "Evaluating all classifiers..." << Endl;

   if (Data().GetTrainingTree() == NULL) {
      fLogger << kWARNING << "You perform testing without training before, hope you "
              << "provided a reasonable test tree and weight files " << Endl;
   } 

   // -----------------------------------------------------------------------
   // First part of evaluation process
   // --> compute efficiencies, and other separation estimators
   // -----------------------------------------------------------------------

   // although equal, we now want to seperate the outpuf for the variables
   // and the real methods
   Int_t isel;                  // will be 0 for a Method; 1 for a Variable
   Int_t nmeth_used[2] = {0,0}; // 0 Method; 1 Variable

   vector<vector<TString> >  mname(2);
   vector<vector<Double_t> > sig(2), sep(2);
   vector<vector<Double_t> > eff01(2), eff10(2), eff30(2), effArea(2);
   vector<vector<Double_t> > eff01err(2), eff10err(2), eff30err(2);
   vector<vector<Double_t> > trainEff01(2), trainEff10(2), trainEff30(2);

   // following vector contains all methods - with the exception of Cuts, which are special
   vector<IMethod*> methodsNoCuts; 

   // iterate over methods and evaluate
   vector<IMethod*>::iterator itrMethod    = fMethods.begin();
   vector<IMethod*>::iterator itrMethodEnd = fMethods.end();
   for (; itrMethod != itrMethodEnd; itrMethod++) {
      MethodBase* theMethod = (MethodBase*)*itrMethod;
      if (theMethod->GetMethodType() != Types::kCuts) methodsNoCuts.push_back( *itrMethod );

      fLogger << kINFO << "Evaluate classifier: " << theMethod->GetMethodTitle() << Endl;
      isel = (theMethod->GetMethodName().Contains("Variable")) ? 1 : 0;
      
      // perform the evaluation
      theMethod->Test(0);

      // evaluate the classifier
      mname[isel].push_back( theMethod->GetMethodTitle() );
      sig[isel].push_back  ( theMethod->GetSignificance() );
      sep[isel].push_back  ( theMethod->GetSeparation() );
      //      TTree * testTree = theMethod->Data().GetTestTree();
      //      TTree* testTree = Data().GetTestTree();
      Double_t err;
      eff01[isel].push_back( theMethod->GetEfficiency("Efficiency:0.01", Data().GetTestTree(), err) );
      eff01err[isel].push_back( err );
      eff10[isel].push_back( theMethod->GetEfficiency("Efficiency:0.10", Data().GetTestTree(), err) );
      eff10err[isel].push_back( err );
      eff30[isel].push_back( theMethod->GetEfficiency("Efficiency:0.30", Data().GetTestTree(), err) );
      eff30err[isel].push_back( err );
      effArea[isel].push_back( theMethod->GetEfficiency("", Data().GetTestTree(), err)  ); // computes the area (average)

      trainEff01[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.01") ); // the first pass takes longer
      trainEff10[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.10") );
      trainEff30[isel].push_back( theMethod->GetTrainingEfficiency("Efficiency:0.30") );

      nmeth_used[isel]++;
      theMethod->WriteEvaluationHistosToFile();
   }

   // now sort the variables according to the best 'eff at Beff=0.10'
   for (Int_t k=0; k<2; k++) {
      vector< vector<Double_t> > vtemp;
      vtemp.push_back( effArea[k] );
      vtemp.push_back( eff10[k] ); // this is the vector that is ranked
      vtemp.push_back( eff01[k] );
      vtemp.push_back( eff30[k] );
      vtemp.push_back( eff10err[k] ); // this is the vector that is ranked
      vtemp.push_back( eff01err[k] );
      vtemp.push_back( eff30err[k] );
      vtemp.push_back( trainEff10[k] );
      vtemp.push_back( trainEff01[k] );
      vtemp.push_back( trainEff30[k] );
      vtemp.push_back( sig[k]   );
      vtemp.push_back( sep[k]   );
      vector<TString> vtemps = mname[k];
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
      mname[k]      = vtemps;
   }

   // -----------------------------------------------------------------------
   // Second part of evaluation process
   // --> compute correlations among MVAs
   // --> count overlaps
   // -----------------------------------------------------------------------
   
   const Int_t nvar = methodsNoCuts.size();
   if (nvar > 1) {

      // needed for correlations
      Float_t  *fvec = new Float_t[nvar];
      Double_t *dvec = new Double_t[nvar];
      vector<Double_t> rvec;
      Int_t    type;
      // for correlations
      TPrincipal* tpSig = new TPrincipal( nvar, "" );   
      TPrincipal* tpBkg = new TPrincipal( nvar, "" );   

      // set required tree branch references
      Int_t ivar = 0;
      vector<TString>* theVars = new vector<TString>;
      Data().GetTestTree()->ResetBranchAddresses();
      for (itrMethod = methodsNoCuts.begin(); itrMethod != methodsNoCuts.end(); itrMethod++, ivar++) {
         theVars->push_back( ((MethodBase*)*itrMethod)->GetTestvarName() );
         rvec.push_back( ((MethodBase*)*itrMethod)->GetSignalReferenceCut() );
         Data().GetTestTree()->SetBranchAddress( theVars->back(), &(fvec[ivar]) );
         theVars->back().ReplaceAll( "MVA_", "" );
      }
      Data().GetTestTree()->SetBranchAddress( "type", &type );

      // for overlap study
      TMatrixD* overlapS = new TMatrixD( nvar, nvar );
      TMatrixD* overlapB = new TMatrixD( nvar, nvar );
      (*overlapS) *= 0; // init...
      (*overlapB) *= 0; // init...

      // loop over test tree
      for (Int_t ievt=0; ievt<Data().GetTestTree()->GetEntries(); ievt++) {
         Data().GetTestTree()->GetEntry(ievt);

         // for correlations
         TMatrixD* theMat = 0;
         for (Int_t im=0; im<nvar; im++) dvec[im] = (Double_t)fvec[im];
         if (type == 1) { tpSig->AddRow( dvec ); theMat = overlapS; }
         else           { tpBkg->AddRow( dvec ); theMat = overlapB; }

         // count overlaps
         for (Int_t im=0; im<nvar; im++) {
            for (Int_t jm=im; jm<nvar; jm++) {
               if ((dvec[im] - rvec[im])*(dvec[jm] - rvec[jm]) > 0) { 
                  (*theMat)(im,jm)++; 
                  if (im != jm) (*theMat)(jm,im)++;
               }
            }
         }      
      }

      // renormalise overlap matrix
      (*overlapS) *= (1.0/Data().GetNEvtSigTest());  // init...
      (*overlapB) *= (1.0/Data().GetNEvtBkgdTest()); // init...
   
      tpSig->MakePrincipals();
      tpBkg->MakePrincipals();

      const TMatrixD* covMatS = tpSig->GetCovarianceMatrix();
      const TMatrixD* covMatB = tpBkg->GetCovarianceMatrix();
   
      const TMatrixD* corrMatS = gTools().GetCorrelationMatrix( covMatS );
      const TMatrixD* corrMatB = gTools().GetCorrelationMatrix( covMatB );

      // print correlation matrices
      if (corrMatS != 0 && corrMatB != 0) {

         fLogger << kINFO << Endl;
         fLogger << kINFO << "Inter-MVA correlation matrix (signal):" << Endl;
         gTools().FormattedOutput( *corrMatS, *theVars, fLogger );
         fLogger << kINFO << Endl;

         fLogger << kINFO << "Inter-MVA correlation matrix (background):" << Endl;
         gTools().FormattedOutput( *corrMatB, *theVars, fLogger );
         fLogger << kINFO << Endl;   
      }
      else fLogger << kWARNING << "<TestAllMethods> cannot compute correlation matrices" << Endl;

      // print overlap matrices
      fLogger << kINFO << "The following \"overlap\" matrices contain the fraction of events for which " << Endl;
      fLogger << kINFO << "the MVAs 'i' and 'j' have returned conform answers about \"signal-likeness\"" << Endl;
      fLogger << kINFO << "An event is signal-like, if its MVA output exceeds the following value:" << Endl;
      gTools().FormattedOutput( rvec, *theVars, "Method" , "Cut value", fLogger );
      fLogger << kINFO << "which correspond to the working point: eff(signal) = 1 - eff(background)" << Endl;

      // give notice that cut method has been excluded from this test
      if (nvar != (Int_t)fMethods.size()) 
         fLogger << kINFO << "Note: no correlations and overlap with cut method are provided at present" << Endl;

      fLogger << kINFO << Endl;
      fLogger << kINFO << "Inter-MVA overlap matrix (signal):" << Endl;
      gTools().FormattedOutput( *overlapS, *theVars, fLogger );
      fLogger << kINFO << Endl;
      
      fLogger << kINFO << "Inter-MVA overlap matrix (background):" << Endl;
      gTools().FormattedOutput( *overlapB, *theVars, fLogger );

      // cleanup
      delete tpSig;
      delete tpBkg;
      delete corrMatS;
      delete corrMatB;
      delete theVars;
      delete overlapS;
      delete overlapB;
      delete [] fvec;
      delete [] dvec;
   }

   // -----------------------------------------------------------------------
   // Third part of evaluation process
   // --> output
   // ----------------------------------------------------------------------- 

   fLogger << Endl;
   TString hLine = "-----------------------------------------------------------------------------";
   fLogger << kINFO << "Evaluation results ranked by best signal efficiency and purity (area)" << Endl;
   fLogger << kINFO << hLine << Endl;
   fLogger << kINFO << "MVA              Signal efficiency at bkg eff. (error):  |  Sepa-    Signifi-"   << Endl;
   fLogger << kINFO << "Methods:         @B=0.01    @B=0.10    @B=0.30    Area   |  ration:  cance:  "   << Endl;
   fLogger << kINFO << hLine << Endl;
   for (Int_t k=0; k<2; k++) {
      if (k == 1 && nmeth_used[k] > 0) {
         fLogger << kINFO << hLine << Endl;
         fLogger << kINFO << "Input Variables: " << Endl << hLine << Endl;
      }
      for (Int_t i=0; i<nmeth_used[k]; i++) {
         if (k == 1) mname[k][i].ReplaceAll( "Variable_", "" );
         fLogger << kINFO << Form("%-15s: %1.3f(%02i)  %1.3f(%02i)  %1.3f(%02i)  %1.3f  |  %1.3f    %1.3f",
                                  (const char*)mname[k][i], 
                                  eff01[k][i], Int_t(1000*eff01err[k][i]), 
                                  eff10[k][i], Int_t(1000*eff10err[k][i]), 
                                  eff30[k][i], Int_t(1000*eff30err[k][i]), 
                                  effArea[k][i], 
                                  sep[k][i], sig[k][i]) << Endl;
      }
   }
   fLogger << kINFO << hLine << Endl;
   fLogger << kINFO << Endl;
   fLogger << kINFO << "Testing efficiency compared to training efficiency (overtraining check)" << Endl;
   fLogger << kINFO << hLine << Endl;
   fLogger << kINFO << "MVA           Signal efficiency: from test sample (from traing sample) "   << Endl;
   fLogger << kINFO << "Methods:         @B=0.01             @B=0.10            @B=0.30   "   << Endl;
   fLogger << kINFO << hLine << Endl;
   for (Int_t k=0; k<2; k++) {
      if (k == 1 && nmeth_used[k] > 0) {
         fLogger << kINFO << hLine << Endl;
         fLogger << kINFO << "Input Variables: " << Endl << hLine << Endl;
      }
      for (Int_t i=0; i<nmeth_used[k]; i++) {
         if (k == 1) mname[k][i].ReplaceAll( "Variable_", "" );
         fLogger << kINFO << Form("%-15s: %1.3f (%1.3f)       %1.3f (%1.3f)      %1.3f (%1.3f)",
                                  (const char*)mname[k][i], 
                                  eff01[k][i],trainEff01[k][i], 
                                  eff10[k][i],trainEff10[k][i],
                                  eff30[k][i],trainEff30[k][i]) << Endl;
      }
   }
   fLogger << kINFO << hLine << Endl;
   fLogger << kINFO << Endl;  
   fLogger << kINFO << "Write Test Tree '"<< Data().GetTestTree()->GetName()<<"' to file" << Endl;
   Data().BaseRootDir()->cd();
   Data().GetTestTree()->SetDirectory( Data().BaseRootDir() );
   Data().GetTestTree()->Write("",TObject::kOverwrite);
}

