// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBDT (BDT = Boosted Decision Trees)                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Doug Schouten   <dschoute@sfu.ca>        - Simon Fraser U., Canada        *
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
//
// Analysis of Boosted Decision Trees
//
// Boosted decision trees have been successfully used in High Energy
// Physics analysis for example by the MiniBooNE experiment
// (Yang-Roe-Zhu, physics/0508045). In Boosted Decision Trees, the
// selection is done on a majority vote on the result of several decision
// trees, which are all derived from the same training sample by
// supplying different event weights during the training.
//
// Decision trees:
//
// Successive decision nodes are used to categorize the
// events out of the sample as either signal or background. Each node
// uses only a single discriminating variable to decide if the event is
// signal-like ("goes right") or background-like ("goes left"). This
// forms a tree like structure with "baskets" at the end (leave nodes),
// and an event is classified as either signal or background according to
// whether the basket where it ends up has been classified signal or
// background during the training. Training of a decision tree is the
// process to define the "cut criteria" for each node. The training
// starts with the root node. Here one takes the full training event
// sample and selects the variable and corresponding cut value that gives
// the best separation between signal and background at this stage. Using
// this cut criterion, the sample is then divided into two subsamples, a
// signal-like (right) and a background-like (left) sample. Two new nodes
// are then created for each of the two sub-samples and they are
// constructed using the same mechanism as described for the root
// node. The devision is stopped once a certain node has reached either a
// minimum number of events, or a minimum or maximum signal purity. These
// leave nodes are then called "signal" or "background" if they contain
// more signal respective background events from the training sample.
//
// Boosting:
//
// The idea behind the boosting is, that signal events from the training
// sample, that end up in a background node (and vice versa) are given a
// larger weight than events that are in the correct leave node. This
// results in a re-weighed training event sample, with which then a new
// decision tree can be developed. The boosting can be applied several
// times (typically 100-500 times) and one ends up with a set of decision
// trees (a forest).
//
// Bagging:
//
// In this particular variant of the Boosted Decision Trees the boosting
// is not done on the basis of previous training results, but by a simple
// stochastic re-sampling of the initial training event sample.
//
// Random Trees:
// Similar to the "Random Forests" from Leo Breiman and Adele Cutler, it
// uses the bagging algorithm together and bases the determination of the
// best node-split during the training on a random subset of variables only
// which is individually chosen for each split.
//
// Analysis:
//
// Applying an individual decision tree to a test event results in a
// classification of the event as either signal or background. For the
// boosted decision tree selection, an event is successively subjected to
// the whole set of decision trees and depending on how often it is
// classified as signal, a "likelihood" estimator is constructed for the
// event being signal or background. The value of this estimator is the
// one which is then used to select the events from an event sample, and
// the cut value on this estimator defines the efficiency and purity of
// the selection.
//
//_______________________________________________________________________

#include <algorithm>

#include <math.h>
#include <fstream>

#include "Riostream.h"
#include "TRandom3.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TObjString.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodBDT.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Ranking.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/BinarySearchTree.h"
#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/GiniIndexWithLaplace.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/Results.h"

using std::vector;

REGISTER_METHOD(BDT)

ClassImp(TMVA::MethodBDT)

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption,
                            TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kBDT, methodTitle, theData, theOption, theTargetDir )
{
   // the standard constructor for the "boosted decision trees"
}

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( DataSetInfo& theData,
                            const TString& theWeightFile,
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( Types::kBDT, theData, theWeightFile, theTargetDir )
{
   // constructor for calculating BDT-MVA using previously generated decision trees
   // the result of the previous training (the decision trees) are read in via the
   // weight file. Make sure the the variables correspond to the ones used in
   // creating the "weight"-file
}

//_______________________________________________________________________
Bool_t TMVA::MethodBDT::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   // BDT can handle classification with 2 classes and regression with one regression-target
   if( type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   if( type == Types::kRegression && numberTargets == 1 ) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodBDT::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // know options:
   // nTrees        number of trees in the forest to be created
   // BoostType     the boosting type for the trees in the forest (AdaBoost e.t.c..)
   //                  known: AdaBoost
   //                         AdaBoostR2 (Adaboost for regression)
   //                         Bagging
   //                         GradBoost
   // AdaBoostBeta     the boosting parameter, beta, for AdaBoost
   // UseRandomisedTrees  choose at each node splitting a random set of variables
   // UseNvars         use UseNvars variables in randomised trees
   // UseNTrainEvents  number of training events used in randomised (and bagged) trees
   // SeparationType   the separation criterion applied in the node splitting
   //                  known: GiniIndex
   //                         MisClassificationError
   //                         CrossEntropy
   //                         SDivSqrtSPlusB
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // nCuts:           the number of steps in the optimisation of the cut for a node (if < 0, then
   //                  step size is determined by the events)
   // UseYesNoLeaf     decide if the classification is done simply by the node type, or the S/B
   //                  (from the training) in the leaf node
   // NodePurityLimit  the minimum purity to classify a node as a signal node (used in pruning and boosting to determine
   //                  misclassification error rate)
   // UseWeightedTrees use average classification from the trees, or have the individual trees
   //                  trees in the forest weighted (e.g. log(boostweight) from AdaBoost
   // PruneMethod      The Pruning method:
   //                  known: NoPruning  // switch off pruning completely
   //                         ExpectedError
   //                         CostComplexity
   // PruneStrength    a parameter to adjust the amount of pruning. Should be large enough such that overtraining is avoided.
   // PruneBeforeBoost flag to prune the tree before applying boosting algorithm
   // PruningValFraction   number of events to use for optimizing pruning (only if PruneStrength < 0, i.e. automatic pruning)
   // NoNegWeightsInTraining  Ignore negative weight events in the training.
   // NNodesMax        maximum number of nodes allwed in the tree splitting, then it stops.
   // MaxDepth         maximum depth of the decision tree allowed before further splitting is stopped

   DeclareOptionRef(fNTrees, "NTrees", "Number of trees in the forest");
   DeclareOptionRef(fBoostType, "BoostType", "Boosting type for the trees in the forest");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Bagging"));
   AddPreDefVal(TString("RegBoost"));
   AddPreDefVal(TString("AdaBoostR2"));
   AddPreDefVal(TString("Grad"));
   DeclareOptionRef(fAdaBoostR2Loss, "AdaBoostR2Loss", "Type of Loss function in AdaBoostR2t (Linear,Quadratic or Exponential)");
   AddPreDefVal(TString("Linear"));
   AddPreDefVal(TString("Quadratic"));
   AddPreDefVal(TString("Exponential"));

   DeclareOptionRef(fBaggedGradBoost=kFALSE, "UseBaggedGrad","Use only a random subsample of all events for growing the trees in each iteration. (Only valid for GradBoost)");
   DeclareOptionRef(fSampleFraction=0.6, "GradBaggingFraction","Defines the fraction of events to be used in each iteration when UseBaggedGrad=kTRUE. (Only valid for GradBoost)");
   DeclareOptionRef(fShrinkage=1.0, "Shrinkage", "Learning rate for GradBoost algorithm");
   DeclareOptionRef(fAdaBoostBeta=1.0, "AdaBoostBeta", "Parameter for AdaBoost algorithm");
   DeclareOptionRef(fRandomisedTrees,"UseRandomisedTrees","Choose at each node splitting a random set of variables");
   DeclareOptionRef(fUseNvars,"UseNvars","Number of variables used if randomised tree option is chosen");
   DeclareOptionRef(fUseNTrainEvents,"UseNTrainEvents","number of randomly picked training events used in randomised (and bagged) trees");

   DeclareOptionRef(fUseWeightedTrees=kTRUE, "UseWeightedTrees",
                    "Use weighted trees or simple average in classification from the forest");
   DeclareOptionRef(fUseYesNoLeaf=kTRUE, "UseYesNoLeaf",
                    "Use Sig or Bkg categories, or the purity=S/(S+B) as classification of the leaf node");
   DeclareOptionRef(fNodePurityLimit=0.5, "NodePurityLimit", "In boosting/pruning, nodes with purity > NodePurityLimit are signal; background otherwise.");
   DeclareOptionRef(fSepTypeS="GiniIndex", "SeparationType", "Separation criterion for node splitting");
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("GiniIndexWithLaplace"));
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   AddPreDefVal(TString("RegressionVariance"));
   DeclareOptionRef(fNodeMinEvents, "nEventsMin", "Minimum number of events required in a leaf node (default: max(20, N_train/(Nvar^2)/10) ) ");
   DeclareOptionRef(fNCuts, "nCuts", "Number of steps during node cut optimisation");
   DeclareOptionRef(fPruneStrength, "PruneStrength", "Pruning strength");
   DeclareOptionRef(fPruneMethodS, "PruneMethod", "Method used for pruning (removal) of statistically insignificant branches");
   AddPreDefVal(TString("NoPruning"));
   AddPreDefVal(TString("ExpectedError"));
   AddPreDefVal(TString("CostComplexity"));
   DeclareOptionRef(fPruneBeforeBoost=kFALSE, "PruneBeforeBoost", "Flag to prune the tree before applying boosting algorithm");
   DeclareOptionRef(fFValidationEvents=0.5, "PruningValFraction", "Fraction of events to use for optimizing automatic pruning.");
   DeclareOptionRef(fNNodesMax=100000,"NNodesMax","Max number of nodes in tree");
   DeclareOptionRef(fMaxDepth=3,"MaxDepth","Max depth of the decision tree allowed");
}
//_______________________________________________________________________
void TMVA::MethodBDT::ProcessOptions()
{
   // the option string is decoded, for available options see "DeclareOptions"

   fSepTypeS.ToLower();
   if      (fSepTypeS == "misclassificationerror") fSepType = new MisClassificationError();
   else if (fSepTypeS == "giniindex")              fSepType = new GiniIndex();
   else if (fSepTypeS == "giniindexwithlaplace")   fSepType = new GiniIndexWithLaplace();
   else if (fSepTypeS == "crossentropy")           fSepType = new CrossEntropy();
   else if (fSepTypeS == "sdivsqrtsplusb")         fSepType = new SdivSqrtSplusB();
   else if (fSepTypeS == "regressionvariance")     fSepType = NULL;
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown Separation Index option called" << Endl;
   }

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror")  fPruneMethod = DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity") fPruneMethod = DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "nopruning")      fPruneMethod = DecisionTree::kNoPruning;
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown PruneMethod option called" << Endl;
   }
   if (fPruneStrength < 0 && (fPruneMethod != DecisionTree::kNoPruning) && fBoostType!="Grad") fAutomatic = kTRUE;
   else fAutomatic = kFALSE;
   if (fAutomatic && fPruneMethod==DecisionTree::kExpectedErrorPruning){
      Log() << kFATAL 
            <<  "Sorry autmoatic pruning strength determination is not implemented yet for ExpectedErrorPruning" << Endl;
   }
   fAdaBoostR2Loss.ToLower();
   
   if (fBoostType!="Grad") fBaggedGradBoost=kFALSE;
   else fPruneMethod = DecisionTree::kNoPruning;
   if (fFValidationEvents < 0.0) fFValidationEvents = 0.0;
   if (fAutomatic && fFValidationEvents > 0.5) {
      Log() << kWARNING << "You have chosen to use more than half of your training sample "
            << "to optimize the automatic pruning algorithm. This is probably wasteful "
            << "and your overall results will be degraded. Are you sure you want this?"
            << Endl;
   }


   if (this->Data()->HasNegativeEventWeights()){
      Log() << kINFO << " You are using a Monte Carlo that has also negative weights. "
            << "That should in principle be fine as long as on average you end up with "
            << "something positive. For this you have to make sure that the minimal number "
            << "of (un-weighted) events demanded for a tree node (currently you use: nEventsMin="
            <<fNodeMinEvents<<", you can set this via the BDT option string when booking the "
            << "classifier) is large enough to allow for reasonable averaging!!! "
            << " If this does not help.. maybe you want to try the option: NoNegWeightsInTraining  "
            << "which ignores events with negative weight in the training. " << Endl
            << Endl << "Note: You'll get a WARNING message during the training if that should ever happen" << Endl;
   }

   if (DoRegression()) {
      if (fUseYesNoLeaf && !IsConstructedFromWeightFile()){
         Log() << kWARNING << "Regression Trees do not work with fUseYesNoLeaf=TRUE --> I will set it to FALSE" << Endl;
         fUseYesNoLeaf = kFALSE;
      }

      if (fSepType != NULL){
         Log() << kWARNING << "Regression Trees do not work with Separation type other than <RegressionVariance> --> I will use it instead" << Endl;
         fSepType = NULL;
      }
   }
   if (fRandomisedTrees){
      Log() << kINFO << " Randomised trees use *bagging* as *boost* method and no pruning" << Endl;
      fPruneMethod = DecisionTree::kNoPruning;
      fBoostType   = "Bagging";
   }

   //    if (2*fNodeMinEvents >  Data()->GetNTrainingEvents()) {
   //       Log() << kFATAL << "you've demanded a minimun number of events in a leaf node " 
   //             << " that is larger than 1/2 the total number of events in the training sample."
   //             << " Hence I cannot make any split at all... this will not work!" << Endl;
   //    }


}
//_______________________________________________________________________
void TMVA::MethodBDT::Init( void )
{
   // common initialisation with defaults for the BDT-Method
   fNTrees         = 400;
   fBoostType      = "AdaBoost";
   fAdaBoostR2Loss = "Quadratic";
   fNodeMinEvents  = TMath::Max( Int_t(40), Int_t( Data()->GetNTrainingEvents() / (10*GetNvar()*GetNvar())) );
   fNCuts          = 20;
   fPruneMethodS   = "CostComplexity";
   fPruneMethod    = DecisionTree::kCostComplexityPruning;
   fPruneStrength  = -1.0;
   fFValidationEvents = 0.5;
   fRandomisedTrees = kFALSE;
   fUseNvars        =  (GetNvar()>12) ? UInt_t(GetNvar()/8) : TMath::Max(UInt_t(2),UInt_t(GetNvar()/3));
   fUseNTrainEvents = Data()->GetNTrainingEvents();
   fNNodesMax       = 1000000;
   fMaxDepth        = 3;
   fShrinkage       = 1.0;

   // reference cut value to distinguish signal-like from background-like events
   SetSignalReferenceCut( 0 );

}

//_______________________________________________________________________
TMVA::MethodBDT::~MethodBDT( void )
{
   //destructor
   for (UInt_t i=0; i<fEventSample.size();      i++) delete fEventSample[i];
   for (UInt_t i=0; i<fValidationSample.size(); i++) delete fValidationSample[i];
   for (UInt_t i=0; i<fForest.size();           i++) delete fForest[i];
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitEventSample( void )
{
   // Write all Events from the Tree into a vector of Events, that are
   // more easily manipulated. This method should never be called without
   // existing trainingTree, as it the vector of events from the ROOT training tree
   if (!HasTrainingTree()) Log() << kFATAL << "<Init> Data().TrainingTree() is zero pointer" << Endl;

   UInt_t nevents = Data()->GetNTrainingEvents();
   Bool_t first=kTRUE;

   for (UInt_t ievt=0; ievt<nevents; ievt++) {

      Event* event = new Event( *GetTrainingEvent(ievt) );

      if (!IgnoreEventsWithNegWeightsInTraining() || event->GetWeight() > 0) {
         if (first && event->GetWeight() < 0) {
            first = kFALSE;
            Log() << kINFO << "Events with negative event weights are ignored during "
                  << "the BDT training (option NoNegWeightsInTraining=" 
                  << IgnoreEventsWithNegWeightsInTraining() <<  ")" << Endl;
         }
         // if fAutomatic == true you need a validation sample to optimize pruning
         if (fAutomatic) {
            Double_t modulo = 1.0/(fFValidationEvents);
            Int_t   imodulo = static_cast<Int_t>( fmod(modulo,1.0) > 0.5 ? ceil(modulo) : floor(modulo) );
            if (ievt % imodulo == 0) fValidationSample.push_back( event );
            else                     fEventSample.push_back( event );
         }
         else {
            fEventSample.push_back(event);
         }
      }
   }
   if (fAutomatic) {
      Log() << kINFO << "<InitEventSample> Internally I use " << fEventSample.size()
            << " for Training  and " << fValidationSample.size()
            << " for Pruning Validation (" << ((Float_t)fValidationSample.size())/((Float_t)fEventSample.size()+fValidationSample.size())*100.0
            << "% of training used for validation)" << Endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::Train()
{
   // BDT training

   // fill the STL Vector with the event sample
   InitEventSample();

   if (IsNormalised()) Log() << kFATAL << "\"Normalise\" option cannot be used with BDT; "
                             << "please remove the option from the configuration string, or "
                             << "use \"!Normalise\""
                             << Endl;

   Log() << kINFO << "Training "<< fNTrees << " Decision Trees ... patience please" << Endl;

   Results* results = Data()->GetResults(GetMethodName(), Types::kTraining, GetAnalysisType());

   // book monitoring histograms (currently for AdaBost, only)

   
   // weights applied in boosting
   Int_t nBins;
   Double_t xMin,xMax;
   TString hname = "AdaBooost weight distribution";

   nBins= 100;
   xMin = 0;
   xMax = 30;

   if (DoRegression()) {
      nBins= 100;
      xMin = 0;
      xMax = 1;
      hname="Boost event weights distribution";
   }
      
   TH1* h = new TH1F("BoostWeight",hname,nBins,xMin,xMax);
   h->SetXTitle("boost weight");
   results->Store(h, "BoostWeights");

   // weights applied in boosting vs tree number
   h = new TH1F("BoostWeightVsTree","Boost weights vs tree",fNTrees,0,fNTrees);
   h->SetXTitle("#tree");
   h->SetYTitle("boost weight");
   results->Store(h, "BoostWeightsVsTree");

   // error fraction vs tree number
   h = new TH1F("ErrFractHist","error fraction vs tree number",fNTrees,0,fNTrees);
   h->SetXTitle("#tree");
   h->SetYTitle("error fraction");
   results->Store(h, "ErrorFrac");

   // nNodesBeforePruning vs tree number
   TH1* nodesBeforePruningVsTree = new TH1I("NodesBeforePruning","nodes before pruning",fNTrees,0,fNTrees);
   nodesBeforePruningVsTree->SetXTitle("#tree");
   nodesBeforePruningVsTree->SetYTitle("#tree nodes");
   results->Store(nodesBeforePruningVsTree);

   // nNodesAfterPruning vs tree number
   TH1* nodesAfterPruningVsTree = new TH1I("NodesAfterPruning","nodes after pruning",fNTrees,0,fNTrees);
   nodesAfterPruningVsTree->SetXTitle("#tree");
   nodesAfterPruningVsTree->SetYTitle("#tree nodes");
   results->Store(nodesAfterPruningVsTree);

   fMonitorNtuple= new TTree("MonitorNtuple","BDT variables");
   fMonitorNtuple->Branch("iTree",&fITree,"iTree/I");
   fMonitorNtuple->Branch("boostWeight",&fBoostWeight,"boostWeight/D");
   fMonitorNtuple->Branch("errorFraction",&fErrorFraction,"errorFraction/D");

   Timer timer( fNTrees, GetName() );
   Int_t nNodesBeforePruningCount = 0;
   Int_t nNodesAfterPruningCount = 0;

   Int_t nNodesBeforePruning = 0;
   Int_t nNodesAfterPruning = 0;

   TH1D *alpha = new TH1D("alpha","PruneStrengths",fNTrees,0,fNTrees);
   alpha->SetXTitle("#tree");
   alpha->SetYTitle("PruneStrength");

   if(fBoostType=="Grad"){
      InitGradBoost(fEventSample);
   }

   for (int itree=0; itree<fNTrees; itree++) {
      timer.DrawProgressBar( itree );

      fForest.push_back( new DecisionTree( fSepType, fNodeMinEvents, fNCuts,
                                           fRandomisedTrees, fUseNvars, fNNodesMax, fMaxDepth,
                                           itree, fNodePurityLimit, itree));
      if (fBaggedGradBoost) nNodesBeforePruning = fForest.back()->BuildTree(fSubSample);
      else                  nNodesBeforePruning = fForest.back()->BuildTree(fEventSample);

      if (fBoostType!="Grad")
         if (fUseYesNoLeaf && !DoRegression() ){ // remove leaf nodes where both daughter nodes are of same type
            nNodesBeforePruning = fForest.back()->CleanTree();
         }
      nNodesBeforePruningCount += nNodesBeforePruning;
      nodesBeforePruningVsTree->SetBinContent(itree+1,nNodesBeforePruning);

      fForest.back()->SetPruneMethod(fPruneMethod); // set the pruning method for the tree
      fForest.back()->SetPruneStrength(fPruneStrength); // set the strength parameter

      std::vector<Event*> * validationSample = NULL;
      if(fAutomatic) validationSample = &fValidationSample;

      if(fBoostType=="Grad"){
         this->Boost(fEventSample, fForest.back(), itree);
      }
      else {
         if(!fPruneBeforeBoost) { // only prune after boosting
            fBoostWeights.push_back( this->Boost(fEventSample, fForest.back(), itree) );
            // if fAutomatic == true, pruneStrength will be the optimal pruning strength
            // determined by the pruning algorithm; otherwise, it is simply the strength parameter
            // set by the user
            Double_t pruneStrength = fForest.back()->PruneTree(validationSample);
            alpha->SetBinContent(itree+1,pruneStrength);
         }
         else { // prune first, then apply a boosting cycle
            Double_t pruneStrength = fForest.back()->PruneTree(validationSample);
            alpha->SetBinContent(itree+1,pruneStrength);
            fBoostWeights.push_back( this->Boost(fEventSample, fForest.back(), itree) );
         }
         
         if (fUseYesNoLeaf && !DoRegression() ){ // remove leaf nodes where both daughter nodes are of same type
            fForest.back()->CleanTree();
         }
      }
      nNodesAfterPruning = fForest.back()->GetNNodes();
      nNodesAfterPruningCount += nNodesAfterPruning;
      nodesAfterPruningVsTree->SetBinContent(itree+1,nNodesAfterPruning);

      fITree = itree;
      fMonitorNtuple->Fill();
   }

   alpha->Write();

   // get elapsed time
   Log() << kINFO << "<Train> elapsed time: " << timer.GetElapsedTime()
         << "                              " << Endl;
   if (fPruneMethod == DecisionTree::kNoPruning) {
      Log() << kINFO << "<Train> average number of nodes (w/o pruning) : "
            << nNodesBeforePruningCount/fNTrees << Endl;
   }
   else {
      Log() << kINFO << "<Train> average number of nodes before/after pruning : "
            << nNodesBeforePruningCount/fNTrees << " / "
            << nNodesAfterPruningCount/fNTrees
            << Endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::GetRandomSubSample()
{
   // fills fEventSample with fSampleFraction*NEvents random training events
   UInt_t nevents = fEventSample.size();
   UInt_t nfraction = static_cast<UInt_t>(fSampleFraction*Data()->GetNTrainingEvents());

   //for (UInt_t i=0; i<fSubSample.size();i++)
   if (fSubSample.size()!=0) fSubSample.clear();
   TRandom3 *trandom   = new TRandom3(fForest.size());

   for (UInt_t ievt=0; ievt<nfraction; ievt++) { // recreate new random subsample
      fSubSample.push_back(fEventSample[(static_cast<UInt_t>(trandom->Uniform(nevents)-1))]);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetGradBoostMVA(TMVA::Event& e, UInt_t nTrees)
{
   //returns MVA value: -1 for background, 1 for signal
   Double_t sum=0;
   for (UInt_t itree=0; itree<nTrees; itree++) {
      //loop over all trees in forest
      sum += fForest[itree]->CheckEvent(e,kFALSE);
 
   }
   return 2.0/(1.0+exp(-2.0*sum))-1; //MVA output between -1 and 1
}


//_______________________________________________________________________
void TMVA::MethodBDT::UpdateTargets(vector<TMVA::Event*> eventSample)
{
   //Calculate residua for all events;
   UInt_t iValue=0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      fBoostWeights[iValue]+=fForest.back()->CheckEvent(*(*e),kFALSE);
      Double_t p_sig=1.0/(1.0+exp(-2.0*fBoostWeights[iValue]));
      Double_t res = ((*e)->IsSignal()?1:0)-p_sig;
      (*e)->SetTarget(0,res);
      iValue++;
   }   
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GradBoost( vector<TMVA::Event*> eventSample, DecisionTree *dt )
{
   //Calculate the desired response value for each region (line search)
   std::map<TMVA::DecisionTreeNode*,vector<Double_t> > leaves;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      TMVA::DecisionTreeNode* node = dt->GetEventNode(*(*e));
      if ((leaves[node]).size()==0){
         (leaves[node]).push_back((*e)->GetTarget(0) * (*e)->GetWeight());
         (leaves[node]).push_back(fabs((*e)->GetTarget(0))*(1.0-fabs((*e)->GetTarget(0))) * (*e)->GetWeight() * (*e)->GetWeight());
      }
      else {
         (leaves[node])[0]+=((*e)->GetTarget(0) * (*e)->GetWeight());
         (leaves[node])[1]+=fabs((*e)->GetTarget(0))*(1.0-fabs((*e)->GetTarget(0))) *
            ((*e)->GetWeight()) * ((*e)->GetWeight());
      }
   }
   for (std::map<TMVA::DecisionTreeNode*,vector<Double_t> >::iterator iLeave=leaves.begin();
        iLeave!=leaves.end();++iLeave){
      if ((iLeave->second)[1]<1e-30) (iLeave->second)[1]=1e-30;

      (iLeave->first)->SetResponse(fShrinkage*0.5*(iLeave->second)[0]/((iLeave->second)[1]));
   }
   //call UpdateTargets before next tree is grown
   UpdateTargets(eventSample);
   if (fBaggedGradBoost) GetRandomSubSample();
   return 1; //trees all have the same weight
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitGradBoost( vector<TMVA::Event*> eventSample)
{
   // initialize targets for first tree
   fSepType=NULL; //set fSepType to NULL (regression)
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t r = ((*e)->IsSignal()?1:0)-0.5; //Calculate initial residua
      (*e)->SetTarget(0,r);
      fBoostWeights.push_back(0);
   }
   if (fBaggedGradBoost) GetRandomSubSample(); 
}
//_______________________________________________________________________
Double_t TMVA::MethodBDT::TestTreeQuality( DecisionTree *dt )
{
   // test the tree quality.. in terms of Miscalssification

   Double_t ncorrect=0, nfalse=0;
   for (UInt_t ievt=0; ievt<fValidationSample.size(); ievt++) {
      Bool_t isSignalType= (dt->CheckEvent(*(fValidationSample[ievt])) > fNodePurityLimit ) ? 1 : 0;

      if (isSignalType == ((fValidationSample[ievt])->IsSignal()) ) {
         ncorrect += fValidationSample[ievt]->GetWeight();
      }
      else{
         nfalse += fValidationSample[ievt]->GetWeight();
      }
   }

   return  ncorrect / (ncorrect + nfalse);
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Boost( vector<TMVA::Event*> eventSample, DecisionTree *dt, Int_t iTree )
{
   // apply the boosting alogrithim (the algorithm is selecte via the the "option" given
   // in the constructor. The return value is the boosting weight

   if      (fBoostType=="AdaBoost")    return this->AdaBoost  (eventSample, dt);
   else if (fBoostType=="Bagging")     return this->Bagging   (eventSample, iTree);
   else if (fBoostType=="RegBoost")    return this->RegBoost  (eventSample, dt);
   else if (fBoostType=="AdaBoostR2")  return this->AdaBoostR2(eventSample, dt);
   else if (fBoostType=="Grad")        return this->GradBoost (eventSample, dt);
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<Boost> unknown boost option called" << Endl;
   }

   return -1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::AdaBoost( vector<TMVA::Event*> eventSample, DecisionTree *dt )
{
   // the AdaBoost implementation.
   // a new training sample is generated by weighting
   // events that are misclassified by the decision tree. The weight
   // applied is w = (1-err)/err or more general:
   //            w = ((1-err)/err)^beta
   // where err is the fraction of misclassified events in the tree ( <0.5 assuming
   // demanding the that previous selection was better than random guessing)
   // and "beta" beeing a free parameter (standard: beta = 1) that modifies the
   // boosting.

   Double_t err=0, sumw=0, sumwfalse=0, sumwfalse2=0;
   Double_t maxDev=0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t w = (*e)->GetWeight();
      sumw += w;
      if ( DoRegression() ) {
         Double_t tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ); 
         sumwfalse += w * tmpDev;
         sumwfalse2 += w * tmpDev*tmpDev;
         if (tmpDev > maxDev) maxDev = tmpDev;
      }else{
         Bool_t isSignalType = (dt->CheckEvent(*(*e),fUseYesNoLeaf) > fNodePurityLimit );
         //       if (!(isSignalType == DataInfo().IsSignal((*e)))) {
         if (!(isSignalType == (*e)->IsSignal())) {
            sumwfalse+= w;
         }
      }
   }
   err = sumwfalse/sumw ;
   if ( DoRegression() ) {
      //if quadratic loss:
      if (fAdaBoostR2Loss=="linear"){
         err = sumwfalse/maxDev/sumw ;
      }
      else if (fAdaBoostR2Loss=="quadratic"){
         err = sumwfalse2/maxDev/maxDev/sumw ;
      }
      else if (fAdaBoostR2Loss=="exponential"){
         err = 0;
         for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
            Double_t w = (*e)->GetWeight();
            Double_t  tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ); 
            err += w * (1 - exp (-tmpDev/maxDev)) / sumw;
         }
         
      }
      else {
         Log() << kFATAL << " you've chosen a Loss type for Adaboost other than linear, quadratic or exponential " 
               << " namely " << fAdaBoostR2Loss << "\n" 
               << "and this is not implemented... a typo in the options ??" <<Endl;
      }
   }
   Double_t newSumw=0;

   Double_t boostWeight=1.;
   if (err >= 0.5) { // sanity check ... should never happen as otherwise there is apparently
      // something odd with the assignement of the leaf nodes (rem: you use the training
      // events for this determination of the error rate)
      Log() << kWARNING << " The error rate in the BDT boosting is > 0.5. ("<< err
            << ") That should not happen, please check your code (i.e... the BDT code), I "
            << " set it to 0.5.. just to continue.." <<  Endl;
      err = 0.5;
   } else if (err < 0) {
      Log() << kWARNING << " The error rate in the BDT boosting is < 0. That can happen"
            << " due to improper treatment of negative weights in a Monte Carlo.. (if you have"
            << " an idea on how to do it in a better way, please let me know (Helge.Voss@cern.ch)"
            << " for the time being I set it to its absolute value.. just to continue.." <<  Endl;
      err = TMath::Abs(err);
   }
   if (fAdaBoostBeta == 1) {
      boostWeight = (1.-err)/err;
   }
   else {
      boostWeight =  TMath::Power((1.0 - err)/err, fAdaBoostBeta);
   }

   Results* results = Data()->GetResults(GetMethodName(),Types::kTraining, Types::kMaxAnalysisType);

   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      //       if ((!( (dt->CheckEvent(*(*e),fUseYesNoLeaf) > fNodePurityLimit ) == DataInfo().IsSignal((*e)))) || DoRegression()) {
      if ((!( (dt->CheckEvent(*(*e),fUseYesNoLeaf) > fNodePurityLimit ) == (*e)->IsSignal())) || DoRegression()) {
         Double_t boostfactor = boostWeight;
         if (DoRegression()) boostfactor = TMath::Power(1/boostWeight,(1.-TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) )/maxDev ) );
         if ( (*e)->GetWeight() > 0 ){
            (*e)->SetBoostWeight( (*e)->GetBoostWeight() * boostfactor);
            if (DoRegression()) results->GetHist("BoostWeights")->Fill(boostfactor);
            //            cout << "  " << boostfactor << endl;
         } else {
            (*e)->SetBoostWeight( (*e)->GetBoostWeight() / boostfactor);
         }
      }
      newSumw+=(*e)->GetWeight();
   }

   // re-normalise the weights
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetBoostWeight( (*e)->GetBoostWeight() * sumw / newSumw );
   }

   if (!(DoRegression()))results->GetHist("BoostWeights")->Fill(boostWeight);
   results->GetHist("BoostWeightsVsTree")->SetBinContent(fForest.size(),boostWeight);
   results->GetHist("ErrorFrac")->SetBinContent(fForest.size(),err);

   fBoostWeight = boostWeight;
   fErrorFraction = err;

   return TMath::Log(boostWeight);
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Bagging( vector<TMVA::Event*> eventSample, Int_t iTree )
{
   // call it boot-strapping, re-sampling or whatever you like, in the end it is nothing
   // else but applying "random" weights to each event.
   Double_t newSumw=0;
   Double_t newWeight;
   TRandom3 *trandom   = new TRandom3(iTree);
   Double_t eventFraction = fUseNTrainEvents/Data()->GetNTrainingEvents();
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      newWeight = trandom->PoissonD(eventFraction);
      (*e)->SetBoostWeight(newWeight);
      newSumw+=(*e)->GetBoostWeight();
   }
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetBoostWeight( (*e)->GetBoostWeight() * eventSample.size() / newSumw );
   }
   return 1.;  //here as there are random weights for each event, just return a constant==1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::RegBoost( vector<TMVA::Event*> /* eventSample */, DecisionTree* /* dt */ )
{
   // a special boosting only for Regression ...
   // maybe I'll implement it later...

   return 1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::AdaBoostR2( vector<TMVA::Event*> eventSample, DecisionTree *dt )
{
   // adaption of the AdaBoost to regression problems (see H.Drucker 1997)

   if ( !DoRegression() ) Log() << kFATAL << "Somehow you chose a regression boost method for a classification job" << Endl;

   Double_t err=0, sumw=0, sumwfalse=0, sumwfalse2=0;
   Double_t maxDev=0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t w = (*e)->GetWeight();
      sumw += w;

      Double_t  tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) );
      sumwfalse  += w * tmpDev;
      sumwfalse2 += w * tmpDev*tmpDev;
      if (tmpDev > maxDev) maxDev = tmpDev;
   }

   //if quadratic loss:
   if (fAdaBoostR2Loss=="linear"){
      err = sumwfalse/maxDev/sumw ;
   }
   else if (fAdaBoostR2Loss=="quadratic"){
      err = sumwfalse2/maxDev/maxDev/sumw ;
   }
   else if (fAdaBoostR2Loss=="exponential"){
      err = 0;
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         Double_t w = (*e)->GetWeight();
         Double_t  tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ); 
         err += w * (1 - exp (-tmpDev/maxDev)) / sumw;
      }
      
   }
   else {
      Log() << kFATAL << " you've chosen a Loss type for Adaboost other than linear, quadratic or exponential " 
            << " namely " << fAdaBoostR2Loss << "\n" 
            << "and this is not implemented... a typo in the options ??" <<Endl;
   }



   if (err >= 0.5) {
      Log() << kFATAL << " The error rate in the BDT boosting is > 0.5. "
            << " i.e. " << err 
            << " That should induce a stop condition of the boosting " << Endl;
   } else if (err < 0) {
      Log() << kWARNING << " The error rate in the BDT boosting is < 0. That can happen"
            << " due to improper treatment of negative weights in a Monte Carlo.. (if you have"
            << " an idea on how to do it in a better way, please let me know (Helge.Voss@cern.ch)"
            << " for the time being I set it to its absolute value.. just to continue.." <<  Endl;
      err = TMath::Abs(err);
   }

   Double_t boostWeight = err / (1.-err);
   Double_t newSumw=0;

   Results* results = Data()->GetResults(GetMethodName(), Types::kTraining, Types::kMaxAnalysisType);

   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t boostfactor =  TMath::Power(boostWeight,(1.-TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) )/maxDev ) );
      results->GetHist("BoostWeights")->Fill(boostfactor);
      //      cout << "R2  " << boostfactor << "   " << boostWeight << "   " << (1.-TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) )/maxDev)  << endl;
      if ( (*e)->GetWeight() > 0 ){
         Float_t newBoostWeight = (*e)->GetBoostWeight() * boostfactor;
         Float_t newWeight = (*e)->GetWeight() * (*e)->GetBoostWeight() * boostfactor;
         if (newWeight == 0) {
            std::cout << "Weight=    "   <<   (*e)->GetWeight() << std::endl;
            std::cout << "BoostWeight= " <<   (*e)->GetBoostWeight() << std::endl;
            std::cout << "boostweight="<<boostWeight << "  err= " <<err << std::endl; 
            std::cout << "NewBoostWeight= " <<   newBoostWeight << std::endl;
            std::cout << "boostfactor= " <<  boostfactor << std::endl;
            std::cout << "maxDev     = " <<  maxDev << std::endl;
            std::cout << "tmpDev     = " <<  TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ) << std::endl;
            std::cout << "target     = " <<  (*e)->GetTarget(0)  << std::endl; 
            std::cout << "estimate   = " <<  dt->CheckEvent(*(*e),kFALSE)  << std::endl;
         }
         (*e)->SetBoostWeight( newBoostWeight );
         //         (*e)->SetBoostWeight( (*e)->GetBoostWeight() * boostfactor);
      } else {
         (*e)->SetBoostWeight( (*e)->GetBoostWeight() / boostfactor);
      }
      newSumw+=(*e)->GetWeight();
   }

   // re-normalise the weights
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetBoostWeight( (*e)->GetBoostWeight() * sumw / newSumw );
   }


   results->GetHist("BoostWeightsVsTree")->SetBinContent(fForest.size(),1./boostWeight);
   results->GetHist("ErrorFrac")->SetBinContent(fForest.size(),err);

   fBoostWeight = boostWeight;
   fErrorFraction = err;

   return TMath::Log(1./boostWeight);
}

//_______________________________________________________________________
void TMVA::MethodBDT::WriteWeightsToStream( ostream& out) const
{
   // and save the weights
   out << "NTrees= " << fForest.size() 
       << " of" << " type " << fForest.back()->GetAnalysisType() << endl;
   for (UInt_t i=0; i< fForest.size(); i++) {
      out << "Tree " << i << "  boostWeight " << fBoostWeights[i] << endl;
      (fForest[i])->Print(out);
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::AddWeightsXMLTo( void* parent ) const
{
   // write weights to XML 
   void* wght = gTools().xmlengine().NewChild(parent, 0, "Weights");
   gTools().AddAttr( wght, "NTrees", fForest.size() );
   gTools().AddAttr( wght, "TreeType", fForest.back()->GetAnalysisType() );

   for (UInt_t i=0; i< fForest.size(); i++) {
      void* trxml = fForest[i]->AddXMLTo(wght);
      gTools().AddAttr( trxml, "boostWeight", fBoostWeights[i] );
      gTools().AddAttr( trxml, "itree", i );
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::ReadWeightsFromXML(void* parent) {
   // reads the BDT from the xml file

   for (UInt_t i=0; i<fForest.size(); i++) delete fForest[i];
   fForest.clear();
   fBoostWeights.clear();

   UInt_t ntrees;
   UInt_t analysisType;
   Float_t boostWeight;

   gTools().ReadAttr( parent, "NTrees", ntrees );
   gTools().ReadAttr( parent, "TreeType", analysisType );

   void* ch = gTools().xmlengine().GetChild(parent);
   UInt_t i=0;
   while(ch) {
      fForest.push_back( dynamic_cast<DecisionTree*>( BinaryTree::CreateFromXML(ch) ) );
      fForest.back()->SetAnalysisType(Types::EAnalysisType(analysisType));
      fForest.back()->SetTreeID(i++);
      gTools().ReadAttr(ch,"boostWeight",boostWeight);
      fBoostWeights.push_back(boostWeight);
      ch = gTools().xmlengine().GetNext(ch);
   }
}

//_______________________________________________________________________
void  TMVA::MethodBDT::ReadWeightsFromStream( istream& istr )
{
   // read the weights (BDT coefficients)
   TString var, dummy;
   //   Types::EAnalysisType analysisType;
   Int_t analysisType;

   istr >> dummy >> fNTrees >> dummy >> dummy >> analysisType;
   Log() << kINFO << "Read " << fNTrees << " Decision trees" << Endl;

   for (UInt_t i=0;i<fForest.size();i++) delete fForest[i];
   fForest.clear();
   fBoostWeights.clear();
   Int_t iTree;
   Double_t boostWeight;
   for (int i=0;i<fNTrees;i++) {
      istr >> dummy >> iTree >> dummy >> boostWeight;
      if (iTree != i) {
         fForest.back()->Print( cout );
         Log() << kFATAL << "Error while reading weight file; mismatch iTree="
               << iTree << " i=" << i
               << " dummy " << dummy
               << " boostweight " << boostWeight
               << Endl;
      }
      fForest.push_back( new DecisionTree() );
      fForest.back()->SetAnalysisType(Types::EAnalysisType(analysisType));
      fForest.back()->SetTreeID(i);
      fForest.back()->Read(istr);
      fBoostWeights.push_back(boostWeight);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue( Double_t* err ){
   return this->GetMvaValue( err, 0 );
}
//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue( Double_t* err, UInt_t useNTrees )
{
   // Return the MVA value (range [-1;1]) that classifies the
   // event according to the majority vote from the total number of
   // decision trees.

   // cannot determine error
   if (err != 0) *err = -1;
   
   // allow for the possibility to use less trees in the actual MVA calculation
   // than have been originally trained.
   UInt_t nTrees = fForest.size();
   if (useNTrees > 0 ) nTrees = useNTrees;

   if (fBoostType=="Grad") return GetGradBoostMVA(const_cast<TMVA::Event&>(*GetEvent()),nTrees);
   
   Double_t myMVA = 0;
   Double_t norm  = 0;
   for (UInt_t itree=0; itree<nTrees; itree++) {
      //
      if (fUseWeightedTrees) {
         myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(*GetEvent(),fUseYesNoLeaf);
         norm  += fBoostWeights[itree];
      }
      else {
         myMVA += fForest[itree]->CheckEvent(*GetEvent(),fUseYesNoLeaf);
         norm  += 1;
      }
   }
   return myMVA /= norm;
}
//_______________________________________________________________________
const std::vector<Float_t> & TMVA::MethodBDT::GetRegressionValues()
{
   // get the regression value generated by the BDTs


   if (fRegressionReturnVal == NULL) fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();

   Double_t myMVA = 0;
   Double_t norm  = 0;
   if (fBoostType=="AdaBoostR2") {
      // rather than using the weighted average of the tree respones in the forest
      // H.Decker(1997) proposed to use the "weighted median"
     
      // sort all individual tree responses according to the prediction value 
      //   (keep the association to their tree weight)
      // the sum up all the associated weights (starting from the one whose tree
      //   yielded the smalles response) up to the tree "t" at which you've
      //   added enough tree weights to have more than half of the sum of all tree weights.
      // choose as response of the forest that one which belongs to this "t"

      vector< Double_t > response(fForest.size());
      vector< Double_t > weight(fForest.size());
      Double_t           totalSumOfWeights = 0;

      for (UInt_t itree=0; itree<fForest.size(); itree++) {
         response[itree]    = fForest[itree]->CheckEvent(*GetEvent(),kFALSE);
         weight[itree]      = fBoostWeights[itree];
         totalSumOfWeights += fBoostWeights[itree];
      }

      vector< vector<Double_t> > vtemp;
      vtemp.push_back( response ); // this is the vector that will get sorted
      vtemp.push_back( weight ); 
      gTools().UsefulSortAscending( vtemp );

      Int_t t=0;
      Double_t sumOfWeights = 0;
      while (sumOfWeights <= totalSumOfWeights/2.) {
         sumOfWeights += vtemp[1][t];
         t++;
      }

      Double_t rVal=0;
      Int_t    count=0;
      for (UInt_t i= TMath::Max(UInt_t(0),UInt_t(t-(fForest.size()/6)-0.5)); 
           i< TMath::Min(UInt_t(fForest.size()),UInt_t(t+(fForest.size()/6)+0.5)); i++) {
         count++;
         rVal+=vtemp[0][i];
      }
      fRegressionReturnVal->push_back( rVal/Double_t(count));
   }
   else {
      for (UInt_t itree=0; itree<fForest.size(); itree++) {
         //
         if (fUseWeightedTrees) {
            myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(*GetEvent(),kFALSE);
            norm  += fBoostWeights[itree];
         }
         else {
            myMVA += fForest[itree]->CheckEvent(*GetEvent(),kFALSE);
            norm  += 1;
         }
      }
      fRegressionReturnVal->push_back( myMVA/norm );
   }
   return *fRegressionReturnVal;
}

//_______________________________________________________________________
void  TMVA::MethodBDT::WriteMonitoringHistosToFile( void ) const
{
   // Here we could write some histograms created during the processing
   // to the output file.
   Log() << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;

   Results* results = Data()->GetResults(GetMethodName(), Types::kTraining, Types::kMaxAnalysisType);
   results->GetStorage()->Write();
   fMonitorNtuple->Write();
}

//_______________________________________________________________________
vector< Double_t > TMVA::MethodBDT::GetVariableImportance()
{
   // Return the relative variable importance, normalized to all
   // variables together having the importance 1. The importance in
   // evaluated as the total separation-gain that this variable had in
   // the decision trees (weighted by the number of events)

   fVariableImportance.resize(GetNvar());
   Double_t  sum=0;
   for (int itree = 0; itree < fNTrees; itree++) {
      vector<Double_t> relativeImportance(fForest[itree]->GetVariableImportance());
      for (UInt_t i=0; i< relativeImportance.size(); i++) {
         fVariableImportance[i] += relativeImportance[i];
      }
   }
   for (UInt_t i=0; i< fVariableImportance.size(); i++) sum += fVariableImportance[i];
   for (UInt_t i=0; i< fVariableImportance.size(); i++) fVariableImportance[i] /= sum;

   return fVariableImportance;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetVariableImportance( UInt_t ivar )
{
   // Returns the measure for the variable importance of variable "ivar"
   // which is later used in GetVariableImportance() to calculate the
   // relative variable importances.

   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar < (UInt_t)relativeImportance.size()) return relativeImportance[ivar];
   else Log() << kFATAL << "<GetVariableImportance> ivar = " << ivar << " is out of range " << Endl;

   return -1;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodBDT::CreateRanking()
{
   // Compute ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Variable Importance" );
   vector< Double_t> importance(this->GetVariableImportance());

   for (UInt_t ivar=0; ivar<GetNvar(); ivar++) {

      fRanking->AddRank( Rank( GetInputLabel(ivar), importance[ivar] ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void TMVA::MethodBDT::GetHelpMessage() const
{
   // Get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "Boosted Decision Trees are a collection of individual decision" << Endl;
   Log() << "trees which form a multivariate classifier by (weighted) majority " << Endl;
   Log() << "vote of the individual trees. Consecutive decision trees are  " << Endl;
   Log() << "trained using the original training data set with re-weighted " << Endl;
   Log() << "events. By default, the AdaBoost method is employed, which gives " << Endl;
   Log() << "events that were misclassified in the previous tree a larger " << Endl;
   Log() << "weight in the training of the following tree." << Endl;
   Log() << Endl;
   Log() << "Decision trees are a sequence of binary splits of the data sample" << Endl;
   Log() << "using a single descriminant variable at a time. A test event " << Endl;
   Log() << "ending up after the sequence of left-right splits in a final " << Endl;
   Log() << "(\"leaf\") node is classified as either signal or background" << Endl;
   Log() << "depending on the majority type of training events in that node." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "By the nature of the binary splits performed on the individual" << Endl;
   Log() << "variables, decision trees do not deal well with linear correlations" << Endl;
   Log() << "between variables (they need to approximate the linear split in" << Endl;
   Log() << "the two dimensional space by a sequence of splits on the two " << Endl;
   Log() << "variables individually). Hence decorrelation could be useful " << Endl;
   Log() << "to optimise the BDT performance." << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "The two most important parameters in the configuration are the  " << Endl;
   Log() << "minimal number of events requested by a leaf node (option " << Endl;
   Log() << "\"nEventsMin\"). If this number is too large, detailed features " << Endl;
   Log() << "in the parameter space cannot be modelled. If it is too small, " << Endl;
   Log() << "the risk to overtrain rises." << Endl;
   Log() << "   (Imagine the decision tree is split until the leaf node contains" << Endl;
   Log() << "    only a single event. In such a case, no training event is  " << Endl;
   Log() << "    misclassified, while the situation will look very different" << Endl;
   Log() << "    for the test sample.)" << Endl;
   Log() << Endl;
   Log() << "The default minimal number is currently set to " << Endl;
   Log() << "   max(20, (N_training_events / N_variables^2 / 10)) " << Endl;
   Log() << "and can be changed by the user." << Endl;
   Log() << Endl;
   Log() << "The other crucial parameter, the pruning strength (\"PruneStrength\")," << Endl;
   Log() << "is also related to overtraining. It is a regularisation parameter " << Endl;
   Log() << "that is used when determining after the training which splits " << Endl;
   Log() << "are considered statistically insignificant and are removed. The" << Endl;
   Log() << "user is advised to carefully watch the BDT screen output for" << Endl;
   Log() << "the comparison between efficiencies obtained on the training and" << Endl;
   Log() << "the independent test sample. They should be equal within statistical" << Endl;
   Log() << "errors, in order to minimize statistical fluctuations in different samples." << Endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // make ROOT-independent C++ class for classifier response (classifier-specific implementation)

   // write BDT-specific classifier response
   fout << "   std::vector<BDT_DecisionTreeNode*> fForest;       // i.e. root nodes of decision trees" << endl;
   fout << "   std::vector<double>                fBoostWeights; // the weights applied in the individual boosts" << endl;
   fout << "};" << endl << endl;
   fout << "double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double myMVA = 0;" << endl;
   fout << "   double norm  = 0;" << endl;
   fout << "   for (unsigned int itree=0; itree<fForest.size(); itree++){" << endl;
   fout << "      BDT_DecisionTreeNode *current = fForest[itree];" << endl;
   fout << "      while (current->GetNodeType() == 0) { //intermediate node" << endl;
   fout << "         if (current->GoesRight(inputValues)) current=(BDT_DecisionTreeNode*)current->GetRight();" << endl;
   fout << "         else current=(BDT_DecisionTreeNode*)current->GetLeft();" << endl;
   fout << "      }" << endl;
   if (fBoostType=="Grad"){
      fout << "      myMVA += current->GetResponse();" << endl;
   }
   else if (fUseWeightedTrees) {
      if (fUseYesNoLeaf) fout << "      myMVA += fBoostWeights[itree] *  current->GetNodeType();" << endl;
      else               fout << "      myMVA += fBoostWeights[itree] *  current->GetPurity();" << endl;
      fout << "      norm  += fBoostWeights[itree];" << endl;
   }
   else {
      if (fUseYesNoLeaf) fout << "      myMVA += current->GetNodeType();" << endl;
      else               fout << "      myMVA += current->GetPurity();" << endl;
      fout << "      norm  += 1.;" << endl;
   }
   fout << "   }" << endl;
   if (fBoostType=="Grad"){
      fout << "   return 2.0/(1.0+exp(-2.0*myMVA))-1.0;" << endl;
   }
   else fout << "   return myMVA /= norm;" << endl;
   fout << "};" << endl << endl;
   fout << "void " << className << "::Initialize()" << endl;
   fout << "{" << endl;
   //Now for each decision tree, write directly the constructors of the nodes in the tree structure
   for (int itree=0; itree<fNTrees; itree++) {
      fout << "  // itree = " << itree << endl;
      fout << "  fBoostWeights.push_back(" << fBoostWeights[itree] << ");" << endl;
      fout << "  fForest.push_back( " << endl;
      this->MakeClassInstantiateNode((DecisionTreeNode*)fForest[itree]->GetRoot(), fout, className);
      fout <<"   );" << endl;
   }
   fout << "   return;" << endl;
   fout << "};" << endl;
   fout << " " << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   for (unsigned int itree=0; itree<fForest.size(); itree++) { " << endl;
   fout << "      delete fForest[itree]; " << endl;
   fout << "   }" << endl;
   fout << "}" << endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassSpecificHeader(  std::ostream& fout, const TString& ) const
{
   // specific class header
   fout << "#ifndef NN" << endl;
   fout << "#define NN new BDT_DecisionTreeNode" << endl;
   fout << "#endif" << endl;
   fout << "   " << endl;
   fout << "#ifndef BDT_DecisionTreeNode__def" << endl;
   fout << "#define BDT_DecisionTreeNode__def" << endl;
   fout << "   " << endl;
   fout << "class BDT_DecisionTreeNode {" << endl;
   fout << "   " << endl;
   fout << "public:" << endl;
   fout << "   " << endl;
   fout << "   // constructor of an essentially \"empty\" node floating in space" << endl;
   fout << "   BDT_DecisionTreeNode ( BDT_DecisionTreeNode* left," << endl;
   fout << "                          BDT_DecisionTreeNode* right," << endl;
   fout << "                          double cutValue, Bool_t cutType, int selector," << endl;
   fout << "                          int nodeType, double purity, double response ) :" << endl;
   fout << "   fLeft    ( left     )," << endl;
   fout << "   fRight   ( right    )," << endl;
   fout << "   fCutValue( cutValue )," << endl;
   fout << "   fCutType ( cutType  )," << endl;
   fout << "   fSelector( selector )," << endl;
   fout << "   fNodeType( nodeType )," << endl;
   fout << "   fPurity  ( purity   )," << endl;
   fout << "   fResponse( response ){}" << endl << endl;
   fout << "   virtual ~BDT_DecisionTreeNode();" << endl << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;
   fout << "   virtual Bool_t GoesRight( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   BDT_DecisionTreeNode* GetRight( void )  {return fRight; };" << endl << endl;
   fout << "   // test event if it decends the tree at this node to the left " << endl;
   fout << "   virtual Bool_t GoesLeft ( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   BDT_DecisionTreeNode* GetLeft( void ) { return fLeft; };   " << endl << endl;
   fout << "   // return  S/(S+B) (purity) at this node (from  training)" << endl << endl;
   fout << "   double GetPurity( void ) const { return fPurity; } " << endl;
   fout << "   // return the node type" << endl;
   fout << "   int    GetNodeType( void ) const { return fNodeType; }" << endl;
   fout << "   double GetResponse(void) const {return fResponse;}" << endl << endl;
   fout << "private:" << endl << endl;
   fout << "   BDT_DecisionTreeNode*   fLeft;     // pointer to the left daughter node" << endl;
   fout << "   BDT_DecisionTreeNode*   fRight;    // pointer to the right daughter node" << endl;
   fout << "   double                  fCutValue; // cut value appplied on this node to discriminate bkg against sig" << endl;
   fout << "   Bool_t                  fCutType;  // true: if event variable > cutValue ==> signal , false otherwise" << endl;
   fout << "   int                     fSelector; // index of variable used in node selection (decision tree)   " << endl;
   fout << "   int                     fNodeType; // Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal " << endl;
   fout << "   double                  fPurity;   // Purity of node from training"<< endl;
   fout << "   double                  fResponse; // Regression response value of node" << endl;
   fout << "}; " << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "BDT_DecisionTreeNode::~BDT_DecisionTreeNode()" << endl;
   fout << "{" << endl;
   fout << "   if (fLeft  != NULL) delete fLeft;" << endl;
   fout << "   if (fRight != NULL) delete fRight;" << endl;
   fout << "}; " << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "Bool_t BDT_DecisionTreeNode::GoesRight( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;
   fout << "   Bool_t result = (inputValues[fSelector] > fCutValue );" << endl;
   fout << "   if (fCutType == true) return result; //the cuts are selecting Signal ;" << endl;
   fout << "   else return !result;" << endl;
   fout << "}" << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "Bool_t BDT_DecisionTreeNode::GoesLeft( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   // test event if it decends the tree at this node to the left" << endl;
   fout << "   if (!this->GoesRight(inputValues)) return true;" << endl;
   fout << "   else return false;" << endl;
   fout << "}" << endl;
   fout << "   " << endl;
   fout << "#endif" << endl;
   fout << "   " << endl;
}

//_______________________________________________________________________
void TMVA::MethodBDT::MakeClassInstantiateNode( DecisionTreeNode *n, std::ostream& fout, const TString& className ) const
{
   // recursively descends a tree and writes the node instance to the output streem
   if (n == NULL) {
      Log() << kFATAL << "MakeClassInstantiateNode: started with undefined node" <<Endl;
      return ;
   }
   fout << "NN("<<endl;
   if (n->GetLeft() != NULL){
      this->MakeClassInstantiateNode( (DecisionTreeNode*)n->GetLeft() , fout, className);
   }
   else {
      fout << "0";
   }
   fout << ", " <<endl;
   if (n->GetRight() != NULL){
      this->MakeClassInstantiateNode( (DecisionTreeNode*)n->GetRight(), fout, className );
   }
   else {
      fout << "0";
   }
   fout << ", " <<  endl
        << setprecision(6)
        << n->GetCutValue() << ", "
        << n->GetCutType() << ", "
        << n->GetSelector() << ", "
        << n->GetNodeType() << ", "
        << n->GetPurity() << ","
        << n->GetResponse() << ") ";

}
