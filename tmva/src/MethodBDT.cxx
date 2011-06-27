// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard v. Toerne, Jan Therhaag

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
 *      Jan Therhaag    <jan.therhaag@cern.ch>   - U. of Bonn, Germany            *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
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
// The idea behind adaptive boosting (AdaBoost) is, that signal events
// from the training sample, that end up in a background node
// (and vice versa) are given a larger weight than events that are in
// the correct leave node. This results in a re-weighed training event
// sample, with which then a new decision tree can be developed.
// The boosting can be applied several times (typically 100-500 times)
// and one ends up with a set of decision trees (a forest).
// Gradient boosting works more like a function expansion approach, where
// each tree corresponds to a summand. The parameters for each summand (tree)
// are determined by the minimization of a error function (binomial log-
// likelihood for classification and Huber loss for regression).
// A greedy algorithm is used, which means, that only one tree is modified
// at a time, while the other trees stay fixed.
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
#include "TGraph.h"

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
#include "TMVA/ResultsMulticlass.h"
#include "TMVA/Interval.h"
#include "TMVA/PDF.h"

using std::vector;

REGISTER_METHOD(BDT)

ClassImp(TMVA::MethodBDT)

const Int_t TMVA::MethodBDT::fgDebugLevel = 0;

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( const TString& jobName,
                            const TString& methodTitle,
                            DataSetInfo& theData,
                            const TString& theOption,
                            TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kBDT, methodTitle, theData, theOption, theTargetDir )
   , fNTrees(0)
   , fRenormByClass(0)        // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fAdaBoostBeta(0)
   , fTransitionPoint(0)
   , fShrinkage(0)
   , fBaggedGradBoost(kFALSE)
   , fSampleFraction(0)
   , fSumOfWeights(0)
   , fNodeMinEvents(0)
   , fNCuts(0)
   , fUseFisherCuts(0)        // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fMinLinCorrForFisher(.8) // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fUseExclusiveVars(0)     // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fUseYesNoLeaf(kFALSE)
   , fNodePurityLimit(0)
   , fUseWeightedTrees(kFALSE)
   , fNNodesMax(0)
   , fMaxDepth(0)
   , fPruneMethod(DecisionTree::kNoPruning)
   , fPruneStrength(0)
   , fPruneBeforeBoost(kFALSE)
   , fFValidationEvents(0)
   , fAutomatic(kFALSE)
   , fRandomisedTrees(kFALSE)
   , fUseNvars(0)
   , fUsePoissonNvars(0)  // don't use this initialisation, only here to make  Coverity happy. Is set in Init()
   , fUseNTrainEvents(0)
   , fSampleSizeFraction(0)
   , fNoNegWeightsInTraining(kFALSE)
   , fInverseBoostNegWeights(kFALSE)
   , fPairNegWeightsGlobal(kFALSE)
   , fPairNegWeightsInNode(kFALSE)
   , fTrainWithNegWeights(kFALSE)
   , fDoBoostMonitor(kFALSE)
   , fITree(0)
   , fBoostWeight(0)
   , fErrorFraction(0)
{
   // the standard constructor for the "boosted decision trees"
   fMonitorNtuple = NULL;
   fSepType = NULL;
}

//_______________________________________________________________________
TMVA::MethodBDT::MethodBDT( DataSetInfo& theData,
                            const TString& theWeightFile,
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( Types::kBDT, theData, theWeightFile, theTargetDir )
   , fNTrees(0)
   , fRenormByClass(0)        // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fAdaBoostBeta(0)
   , fTransitionPoint(0)
   , fShrinkage(0)
   , fBaggedGradBoost(kFALSE)
   , fSampleFraction(0)
   , fSumOfWeights(0)
   , fNodeMinEvents(0)
   , fNCuts(0)
   , fUseFisherCuts(0)        // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fMinLinCorrForFisher(.8) // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fUseExclusiveVars(0)     // don't use this initialisation, only here to make  Coverity happy. Is set in DeclarOptions()
   , fUseYesNoLeaf(kFALSE)
   , fNodePurityLimit(0)
   , fUseWeightedTrees(kFALSE)
   , fNNodesMax(0)
   , fMaxDepth(0)
   , fPruneMethod(DecisionTree::kNoPruning)
   , fPruneStrength(0)
   , fPruneBeforeBoost(kFALSE)
   , fFValidationEvents(0)
   , fAutomatic(kFALSE)
   , fRandomisedTrees(kFALSE)
   , fUseNvars(0)
   , fUsePoissonNvars(0)  // don't use this initialisation, only here to make  Coverity happy. Is set in Init()
   , fUseNTrainEvents(0)
   , fSampleSizeFraction(0)
   , fNoNegWeightsInTraining(kFALSE)
   , fInverseBoostNegWeights(kFALSE)
   , fPairNegWeightsGlobal(kFALSE)
   , fPairNegWeightsInNode(kFALSE)
   , fTrainWithNegWeights(kFALSE)
   , fDoBoostMonitor(kFALSE)
   , fITree(0)
   , fBoostWeight(0)
   , fErrorFraction(0)
{
   fMonitorNtuple = NULL;
   fSepType = NULL;
   // constructor for calculating BDT-MVA using previously generated decision trees
   // the result of the previous training (the decision trees) are read in via the
   // weight file. Make sure the the variables correspond to the ones used in
   // creating the "weight"-file
}

//_______________________________________________________________________
Bool_t TMVA::MethodBDT::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets )
{
   // BDT can handle classification with multiple classes and regression with one regression-target
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass ) return kTRUE;
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
   // UsePoission Nvars use UseNvars not as fixed number but as mean of a possion distribution 
   // UseNTrainEvents  number of training events used in randomised (and bagged) trees
   // SeparationType   the separation criterion applied in the node splitting
   //                  known: GiniIndex
   //                         MisClassificationError
   //                         CrossEntropy
   //                         SDivSqrtSPlusB
   // nEventsMin:      the minimum number of events in a node (leaf criteria, stop splitting)
   // nCuts:           the number of steps in the optimisation of the cut for a node (if < 0, then
   //                  step size is determined by the events)
   // UseFisherCuts:   use multivariate splits using the Fisher criterion
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
   // NegWeightTreatment      IgnoreNegWeightsInTraining  Ignore negative weight events in the training.
   //                         DecreaseBoostWeight     Boost ev. with neg. weight with 1/boostweight instead of boostweight
   //                         PairNegWeightsGlobal    Pair ev. with neg. and pos. weights in traning sample and "annihilate" them 
   //                         PairNegWeightsInNode    Randomly pair miscl. ev. with neg. and pos. weights in node and don't boost them
   // NNodesMax        maximum number of nodes allwed in the tree splitting, then it stops.
   // MaxDepth         maximum depth of the decision tree allowed before further splitting is stopped

   DeclareOptionRef(fNTrees, "NTrees", "Number of trees in the forest");
   DeclareOptionRef(fRenormByClass=kFALSE,"RenormByClass","Individually re-normalize each event class to the original size after boosting");
   DeclareOptionRef(fBoostType, "BoostType", "Boosting type for the trees in the forest");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Bagging"));
   AddPreDefVal(TString("RegBoost"));
   AddPreDefVal(TString("AdaBoostR2"));
   AddPreDefVal(TString("Grad"));
   if (DoRegression()) {
      fBoostType = "AdaBoostR2";
   }else{
      fBoostType = "AdaBoost";
   }
   DeclareOptionRef(fAdaBoostR2Loss="Quadratic", "AdaBoostR2Loss", "Type of Loss function in AdaBoostR2t (Linear,Quadratic or Exponential)");
   AddPreDefVal(TString("Linear"));
   AddPreDefVal(TString("Quadratic"));
   AddPreDefVal(TString("Exponential"));

   DeclareOptionRef(fBaggedGradBoost=kFALSE, "UseBaggedGrad","Use only a random subsample of all events for growing the trees in each iteration. (Only valid for GradBoost)");
   DeclareOptionRef(fSampleFraction=0.6, "GradBaggingFraction","Defines the fraction of events to be used in each iteration when UseBaggedGrad=kTRUE. (Only valid for GradBoost)");
   DeclareOptionRef(fShrinkage=1.0, "Shrinkage", "Learning rate for GradBoost algorithm");
   DeclareOptionRef(fAdaBoostBeta=1.0, "AdaBoostBeta", "Parameter for AdaBoost algorithm");
   DeclareOptionRef(fRandomisedTrees,"UseRandomisedTrees","Choose at each node splitting a random set of variables");
   DeclareOptionRef(fUseNvars,"UseNvars","Number of variables used if randomised tree option is chosen");
   DeclareOptionRef(fUsePoissonNvars,"UsePoissonNvars", "Interpret \"UseNvars\" not as fixed number but as mean of a Possion distribution in each split");
   DeclareOptionRef(fUseNTrainEvents,"UseNTrainEvents","Number of randomly picked training events used in randomised (and bagged) trees");

   DeclareOptionRef(fUseWeightedTrees=kTRUE, "UseWeightedTrees",
                    "Use weighted trees or simple average in classification from the forest");
   DeclareOptionRef(fUseYesNoLeaf=kTRUE, "UseYesNoLeaf",
                    "Use Sig or Bkg categories, or the purity=S/(S+B) as classification of the leaf node");
   if (DoRegression()) {
      fUseYesNoLeaf = kFALSE;
   }


   DeclareOptionRef(fNodePurityLimit=0.5, "NodePurityLimit", "In boosting/pruning, nodes with purity > NodePurityLimit are signal; background otherwise.");
   DeclareOptionRef(fSepTypeS, "SeparationType", "Separation criterion for node splitting");
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("GiniIndexWithLaplace"));
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   AddPreDefVal(TString("RegressionVariance"));
   if (DoRegression()) {
      fSepTypeS = "RegressionVariance";
   }else{
      fSepTypeS = "GiniIndex";
   }
   DeclareOptionRef(fNodeMinEvents, "nEventsMin", "Minimum number of events required in a leaf node (default: Classification: max(40, N_train/(Nvar^2)/10), Regression: 10)");
   DeclareOptionRef(fNCuts, "nCuts", "Number of steps during node cut optimisation");
   DeclareOptionRef(fUseFisherCuts=kFALSE, "UseFisherCuts", "Use multivariate splits using the Fisher criterion");
   DeclareOptionRef(fMinLinCorrForFisher=.8,"MinLinCorrForFisher", "The minimum linear correlation between two variables demanded for use in Fisher criterion in node splitting");
   DeclareOptionRef(fUseExclusiveVars=kFALSE,"UseExclusiveVars","Variables already used in fisher criterion are not anymore analysed individually for node splitting");

   DeclareOptionRef(fPruneStrength, "PruneStrength", "Pruning strength");
   DeclareOptionRef(fPruneMethodS, "PruneMethod", "Method used for pruning (removal) of statistically insignificant branches");
   AddPreDefVal(TString("NoPruning"));
   AddPreDefVal(TString("ExpectedError"));
   AddPreDefVal(TString("CostComplexity"));
   DeclareOptionRef(fPruneBeforeBoost=kFALSE, "PruneBeforeBoost", "Flag to prune the tree before applying boosting algorithm");
   DeclareOptionRef(fFValidationEvents=0.5, "PruningValFraction", "Fraction of events to use for optimizing automatic pruning.");
   DeclareOptionRef(fNNodesMax=100000,"NNodesMax","Max number of nodes in tree");
   if (DoRegression()) {
      DeclareOptionRef(fMaxDepth=50,"MaxDepth","Max depth of the decision tree allowed");
   }else{
      DeclareOptionRef(fMaxDepth=3,"MaxDepth","Max depth of the decision tree allowed");
   }
   DeclareOptionRef(fDoBoostMonitor=kFALSE,"DoBoostMonitor","Create control plot with ROC integral vs tree number");

   DeclareOptionRef(fNegWeightTreatment="InverseBoostNegWeights","NegWeightTreatment","How to treat events with negative weights in the BDT training (particular the boosting) : Ignore;  Boost With inverse boostweight; Pair events with negative and positive weights in traning sample and *annihilate* them (experimental!); Randomly pair events with negative and positive weights in leaf node and do not boost them (experimental!) ");
   AddPreDefVal(TString("IgnoreNegWeights"));
   AddPreDefVal(TString("NoNegWeightsInTraining"));
   AddPreDefVal(TString("InverseBoostNegWeights"));
   AddPreDefVal(TString("PairNegWeightsGlobal"));
   AddPreDefVal(TString("PairNegWeightsInNode"));

}

void TMVA::MethodBDT::DeclareCompatibilityOptions() {
   MethodBase::DeclareCompatibilityOptions();
   DeclareOptionRef(fSampleSizeFraction=1.0,"SampleSizeFraction","Relative size of bagged event sample to original size of the data sample" );

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
      Log() << kFATAL << "<ProcessOptions> unknown Separation Index option " << fSepTypeS << " called" << Endl;
   }

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror")  fPruneMethod = DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity") fPruneMethod = DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "nopruning")      fPruneMethod = DecisionTree::kNoPruning;
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown PruneMethod " << fPruneMethodS << " option called" << Endl;
   }
   if (fPruneStrength < 0 && (fPruneMethod != DecisionTree::kNoPruning) && fBoostType!="Grad") fAutomatic = kTRUE;
   else fAutomatic = kFALSE;
   if (fAutomatic && fPruneMethod==DecisionTree::kExpectedErrorPruning){
      Log() << kFATAL 
            <<  "Sorry autmoatic pruning strength determination is not implemented yet for ExpectedErrorPruning" << Endl;
   }
   fAdaBoostR2Loss.ToLower();
   
   if (fBoostType!="Grad") fBaggedGradBoost=kFALSE;
   else {
      fPruneMethod = DecisionTree::kNoPruning;
      if (fNegWeightTreatment=="InverseBoostNegWeights"){
         Log() << kWARNING << "the option *InverseBoostNegWeights* does not exist for BoostType=Grad --> change to *IgnoreNegWeights*" << Endl;
         fNegWeightTreatment="IgnoreNegWeights";
         fNoNegWeightsInTraining=kTRUE;
       }
   }
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
            << " If this does not help.. maybe you want to try the option: IgnoreNegWeightsInTraining  "
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
      Log() << kINFO << " Randomised trees use no pruning" << Endl;
      fPruneMethod = DecisionTree::kNoPruning;
      //      fBoostType   = "Bagging";
   }

   //    if (2*fNodeMinEvents >  Data()->GetNTrainingEvents()) {
   //       Log() << kFATAL << "you've demanded a minimun number of events in a leaf node " 
   //             << " that is larger than 1/2 the total number of events in the training sample."
   //             << " Hence I cannot make any split at all... this will not work!" << Endl;
   //    }
   
   if (fNTrees==0){
     Log() << kERROR << " Zero Decision Trees demanded... that does not work !! "
           << " I set it to 1 .. just so that the program does not crash"
           << Endl;
     fNTrees = 1;
   }

   fNegWeightTreatment.ToLower();
   if      (fNegWeightTreatment == "ignorenegweights")       fNoNegWeightsInTraining = kTRUE;
   else if (fNegWeightTreatment == "nonegweightsintraining") fNoNegWeightsInTraining = kTRUE;
   else if (fNegWeightTreatment == "inverseboostnegweights") fInverseBoostNegWeights = kTRUE;
   else if (fNegWeightTreatment == "pairnegweightsglobal")   fPairNegWeightsGlobal   = kTRUE;
   else if (fNegWeightTreatment == "pairnegweightsinnode")   fPairNegWeightsInNode   = kTRUE;
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown option for treating negative event weights during training " << fNegWeightTreatment << " requested" << Endl;
   }
   
   if (fNegWeightTreatment == "pairnegweightsglobal") 
      Log() << kWARNING << " you specified the option NegWeightTreatment=PairNegWeightsGlobal : This option is still considered EXPERIMENTAL !! " << Endl;
   if (fNegWeightTreatment == "pairnegweightsginnode") 
      Log() << kWARNING << " you specified the option NegWeightTreatment=PairNegWeightsInNode : This option is still considered EXPERIMENTAL !! " << Endl;
   if (fNegWeightTreatment == "pairnegweightsginnode" && fNCuts <= 0) 
      Log() << kFATAL << " sorry, the option NegWeightTreatment=PairNegWeightsInNode is not yet implemented for NCuts < 0" << Endl;


}
//_______________________________________________________________________
void TMVA::MethodBDT::Init( void )
{
   // common initialisation with defaults for the BDT-Method
      
   fNTrees         = 400;
   if (fAnalysisType == Types::kClassification || fAnalysisType == Types::kMulticlass ) {
      fMaxDepth        = 3;
      fBoostType      = "AdaBoost";
      if(DataInfo().GetNClasses()!=0) //workaround for multiclass application
         fNodeMinEvents  = TMath::Max( Int_t(40), Int_t( Data()->GetNTrainingEvents() / (10*GetNvar()*GetNvar())) );
   }else {
      fMaxDepth = 50;
      fBoostType      = "AdaBoostR2";
      fAdaBoostR2Loss = "Quadratic";
      if(DataInfo().GetNClasses()!=0) //workaround for multiclass application
         fNodeMinEvents  = 10;
   }

   fNCuts          = 20;
   fPruneMethodS   = "NoPruning";
   fPruneMethod    = DecisionTree::kNoPruning;
   fPruneStrength  = 0;
   fAutomatic      = kFALSE;
   fFValidationEvents = 0.5;
   fRandomisedTrees = kFALSE;
   //   fUseNvars        =  (GetNvar()>12) ? UInt_t(GetNvar()/8) : TMath::Max(UInt_t(2),UInt_t(GetNvar()/3));
   fUseNvars        =  UInt_t(TMath::Sqrt(GetNvar())+0.6);
   fUsePoissonNvars = kTRUE;
   if(DataInfo().GetNClasses()!=0) //workaround for multiclass application
      fUseNTrainEvents = Data()->GetNTrainingEvents();
   fNNodesMax       = 1000000;
   fShrinkage       = 1.0;
   fSumOfWeights    = 0.0;

   // reference cut value to distinguish signal-like from background-like events
   SetSignalReferenceCut( 0 );

}


//_______________________________________________________________________
void TMVA::MethodBDT::Reset( void )
{
   // reset the method, as if it had just been instantiated (forget all training etc.)
   
   // I keep the BDT EventSample and its Validation sample (eventuall they should all
   // disappear and just use the DataSet samples ..
   
   // remove all the trees 
   for (UInt_t i=0; i<fForest.size();           i++) delete fForest[i];
   fForest.clear();

   fBoostWeights.clear();
   if (fMonitorNtuple) fMonitorNtuple->Delete(); fMonitorNtuple=NULL;
   fVariableImportance.clear();
   fResiduals.clear();
   // now done in "InitEventSample" which is called in "Train"
   // reset all previously stored/accumulated BOOST weights in the event sample
   //for (UInt_t iev=0; iev<fEventSample.size(); iev++) fEventSample[iev]->SetBoostWeight(1.);
   if (Data()) Data()->DeleteResults(GetMethodName(), Types::kTraining, GetAnalysisType());
   Log() << kDEBUG << " successfully(?) resetted the method " << Endl;                                      
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

   if (fEventSample.size() > 0) { // do not re-initialise the event sample, just set all boostweights to 1. as if it were untouched
      // reset all previously stored/accumulated BOOST weights in the event sample
      for (UInt_t iev=0; iev<fEventSample.size(); iev++) fEventSample[iev]->SetBoostWeight(1.);
   } else {

      UInt_t nevents = Data()->GetNTrainingEvents();
      Bool_t firstNegWeight=kTRUE;

      for (UInt_t ievt=0; ievt<nevents; ievt++) {
         
         Event* event = new Event( *GetTrainingEvent(ievt) );
         
         if (event->GetWeight() < 0 && (IgnoreEventsWithNegWeightsInTraining() || fNoNegWeightsInTraining)){
            if (firstNegWeight) {
               Log() << kWARNING << " Note, you have events with negative event weight in the sample, but you've chosen to ignore them" << Endl;
               firstNegWeight=kFALSE;
            }
            delete event;
         }else{
            if (event->GetWeight() < 0) {
               fTrainWithNegWeights=kTRUE;
               if (firstNegWeight){
                  firstNegWeight = kFALSE;
                  Log() << kWARNING << "Events with negative event weights are USED during "
                        << "the BDT training. This might cause problems with small node sizes " 
                        << "or with the boosting. Please remove negative events from training "
                        << "using the option *IgnoreEventsWithNegWeightsInTraining* in case you "
                        << "observe problems with the boosting"
                        << Endl;
               }
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
      
      // some pre-processing for events with negative weights
      if (fPairNegWeightsGlobal) PreProcessNegativeEventWeights();
   }
}

void TMVA::MethodBDT::PreProcessNegativeEventWeights(){
   // o.k. you know there are events with negative event weights. This routine will remove
   // them by pairing them with the closest event(s) of the same event class with positive
   // weights
   // A first attempt is "brute force", I dont' try to be clever using search trees etc, 
   // just quick and dirty to see if the result is any good  
   Double_t totalNegWeights = 0;
   std::vector<Event*> negEvents;
   for (UInt_t iev = 0; iev < fEventSample.size(); iev++){
      if (fEventSample[iev]->GetWeight() < 0) {
         totalNegWeights += fEventSample[iev]->GetWeight();
         negEvents.push_back(fEventSample[iev]);
      }
   }
   if (totalNegWeights == 0 ) {
      Log() << kINFO << "no negative event weights found .. no preprocessing necessary" << Endl;
      return;
   }
   
   std::vector<TMatrixDSym*>* cov = gTools().CalcCovarianceMatrices( fEventSample, 2);
   
   TMatrixDSym *invCov;

   for (Int_t i=0; i<2; i++){
      invCov = ((*cov)[i]);
      if ( TMath::Abs(invCov->Determinant()) < 10E-24 ) {
         std::cout << "<MethodBDT::PreProcessNeg...> matrix is almost singular with deterninant="
                   << TMath::Abs(invCov->Determinant()) 
                   << " did you use the variables that are linear combinations or highly correlated?" 
                   << std::endl;
      }
      if ( TMath::Abs(invCov->Determinant()) < 10E-120 ) {
         std::cout << "<MethodBDT::PreProcessNeg...> matrix is singular with determinant="
                   << TMath::Abs(invCov->Determinant())  
                   << " did you use the variables that are linear combinations?" 
                   << std::endl;
      }
      
      invCov->Invert();
   }
   


   Log() << kINFO << "Found a total of " << totalNegWeights << " in negative weights out of " << fEventSample.size() << " training events "  << Endl;
   for (UInt_t nev = 0; nev < negEvents.size(); nev++){
      Double_t weight = negEvents[nev]->GetWeight();
      UInt_t  iClassID = negEvents[nev]->GetClass();
      invCov = ((*cov)[iClassID]);
      while (weight < 0){
         // find closest event with positive event weight and "pair" it with the negative event
         // (add their weight) until there is no negative weight anymore
         Int_t iMin=-1;
         Double_t dist, minDist=10E270;
         for (UInt_t iev = 0; iev < fEventSample.size(); iev++){
            if (iClassID==fEventSample[iev]->GetClass() && fEventSample[iev]->GetWeight() > 0){
               dist=0;
               for (UInt_t ivar=0; ivar < GetNvar(); ivar++){
                  for (UInt_t jvar=0; jvar<GetNvar(); jvar++){
                     dist += (negEvents[nev]->GetValue(ivar)-fEventSample[iev]->GetValue(ivar))*
                        (*invCov)[ivar][jvar]*
                        (negEvents[nev]->GetValue(jvar)-fEventSample[iev]->GetValue(jvar));
                  }
               }
               if (dist < minDist) { iMin=iev; minDist=dist;}
            }
         }
         
         if (iMin > -1) { 
            //std::cout << "Happily pairing .. weight before : " << negEvents[nev]->GetWeight() << " and " << fEventSample[iMin]->GetWeight();
            Double_t newWeight= (negEvents[nev]->GetWeight() + fEventSample[iMin]->GetWeight());
            negEvents[nev]->SetBoostWeight( newWeight/negEvents[nev]->GetWeight() );
            fEventSample[iMin]->SetBoostWeight( newWeight/fEventSample[iMin]->GetWeight() );
            //std::cout << " and afterwards " <<  negEvents[nev]->GetWeight() <<  " and the paired " << fEventSample[iMin]->GetWeight() << " dist="<<minDist<< std::endl;
         } else Log() << kFATAL << "preprocessing didn't find event to pair with the negative weight ... probably a bug" << Endl;
         weight = negEvents[nev]->GetWeight();
      }
   }

   // just check.. now there should be no negative event weight left anymore
   totalNegWeights = 0;
   Double_t sigWeight=0;
   Double_t bkgWeight=0;
   Int_t    nSig=0;
   Int_t    nBkg=0;

   std::vector<Event*> newEventSample;

   for (UInt_t iev = 0; iev < fEventSample.size(); iev++){
      if (fEventSample[iev]->GetWeight() < 0) {
         totalNegWeights += fEventSample[iev]->GetWeight();
      }
      if (fEventSample[iev]->GetWeight() > 0) {
         newEventSample.push_back(fEventSample[iev]);
         if (fEventSample[iev]->GetClass() == fSignalClass){
            sigWeight += fEventSample[iev]->GetWeight();
            nSig+=1;
         }else{
            bkgWeight += fEventSample[iev]->GetWeight();
            nBkg+=1;
         }
      }
   }
   if (totalNegWeights < 0) Log() << kFATAL << " compenstion of negative event weights with positive ones did not work " << totalNegWeights << Endl;

   fEventSample = newEventSample;

   Log() << kINFO  << " after PreProcessing, the Event sample is left with " << fEventSample.size() << " events, all positive weight" << Endl;
   Log() << kINFO  << " nSig="<<nSig << " sigWeight="<<sigWeight <<  " nBkg="<<nBkg << " bkgWeight="<<bkgWeight << Endl;
   

}

//

//_______________________________________________________________________
std::map<TString,Double_t>  TMVA::MethodBDT::OptimizeTuningParameters(TString fomType, TString fitType)
{
   // call the Optimzier with the set of paremeters and ranges that
   // are meant to be tuned.

   // fill all the tuning parameters that should be optimized into a map:
   std::map<TString,TMVA::Interval> tuneParameters;
   std::map<TString,Double_t> tunedParameters;

   // note: the 3rd paraemter in the inteval is the "number of bins", NOT the stepsize !!
   //       the actual VALUES at (at least for the scan, guess also in GA) are always
   //       read from the middle of the bins. Hence.. the choice of Intervals e.g. for the
   //       MaxDepth, in order to make nice interger values!!!

   // find some reasonable ranges for the optimisation of NodeMinEvents:
   
   Int_t N  = Int_t( Data()->GetNEvtSigTrain()) ;            
   Int_t min  = TMath::Max( 20,    ( ( N/10000 - (N/10000)%10)  ) );
   Int_t max  = TMath::Max( min*10, TMath::Min( 10000, ( ( N/10    - (N/10)   %100) ) ) );

   tuneParameters.insert(std::pair<TString,Interval>("NTrees",         Interval(50,1000,5))); //  stepsize 50
   tuneParameters.insert(std::pair<TString,Interval>("MaxDepth",       Interval(3,10,8)));    // stepsize 1
   tuneParameters.insert(std::pair<TString,Interval>("NodeMinEvents",  Interval(min,max,5))); // 
   //tuneParameters.insert(std::pair<TString,Interval>("NodePurityLimit",Interval(.4,.6,3)));   // stepsize .1

   // method-specific parameters
   if        (fBoostType=="AdaBoost"){
     tuneParameters.insert(std::pair<TString,Interval>("AdaBoostBeta",   Interval(.5,1.50,5)));   
  
   }else if (fBoostType=="Grad"){
     tuneParameters.insert(std::pair<TString,Interval>("Shrinkage",      Interval(0.05,0.50,5)));  
  
   }else if (fBoostType=="Bagging" && fRandomisedTrees){
     Int_t min_var  = TMath::FloorNint( GetNvar() * .25 );
     Int_t max_var  = TMath::CeilNint(  GetNvar() * .75 ); 
     tuneParameters.insert(std::pair<TString,Interval>("UseNvars",       Interval(min_var,max_var,4)));
     
   }
   
   
   OptimizeConfigParameters optimize(this, tuneParameters, fomType, fitType);
   tunedParameters=optimize.optimize();

   return tunedParameters;

}

//_______________________________________________________________________
void TMVA::MethodBDT::SetTuneParameters(std::map<TString,Double_t> tuneParameters)
{
   // set the tuning parameters accoding to the argument

   std::map<TString,Double_t>::iterator it;
   for(it=tuneParameters.begin(); it!= tuneParameters.end(); it++){
      if (it->first ==  "MaxDepth"       ) SetMaxDepth        ((Int_t)it->second);
      if (it->first ==  "NodeMinEvents"  ) SetNodeMinEvents   ((Int_t)it->second);
      if (it->first ==  "NTrees"         ) SetNTrees          ((Int_t)it->second);
      if (it->first ==  "NodePurityLimit") SetNodePurityLimit (it->second);
      if (it->first ==  "AdaBoostBeta"   ) SetAdaBoostBeta    (it->second);
   }
   
}

//_______________________________________________________________________
void TMVA::MethodBDT::Train()
{
   // BDT training
   TMVA::DecisionTreeNode::fgIsTraining=true;

   // fill the STL Vector with the event sample
   // (needs to be done here and cannot be done in "init" as the options need to be 
   // known). 
   InitEventSample();

   if (fNTrees==0){
     Log() << kERROR << " Zero Decision Trees demanded... that does not work !! "
           << " I set it to 1 .. just so that the program does not crash"
           << Endl;
     fNTrees = 1;
   }

   // HHV (it's been here since looong but I really don't know why we cannot handle
   // normalized variables in BDTs...  todo
   if (IsNormalised()) Log() << kFATAL << "\"Normalise\" option cannot be used with BDT; "
                             << "please remove the option from the configuration string, or "
                             << "use \"!Normalise\""
                             << Endl;

   Log() << kINFO << "Training "<< fNTrees << " Decision Trees ... patience please" << Endl;

   Log() << kDEBUG << "Training with maximal depth = " <<fMaxDepth 
         << ", NodeMinEvents=" << fNodeMinEvents
         << ", NTrees="<<fNTrees
         << ", NodePurityLimit="<<fNodePurityLimit
         << ", AdaBoostBeta="<<fAdaBoostBeta
         << Endl;

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

   // book monitoring histograms (for AdaBost only)   

   TH1* h = new TH1F("BoostWeight",hname,nBins,xMin,xMax);
   TH1* nodesBeforePruningVsTree = new TH1I("NodesBeforePruning","nodes before pruning",fNTrees,0,fNTrees);
   TH1* nodesAfterPruningVsTree = new TH1I("NodesAfterPruning","nodes after pruning",fNTrees,0,fNTrees);

      

   if(!DoMulticlass()){
      Results* results = Data()->GetResults(GetMethodName(), Types::kTraining, GetAnalysisType());

      h->SetXTitle("boost weight");
      results->Store(h, "BoostWeights");
  

      // Monitor the performance (on TEST sample) versus number of trees
      if (fDoBoostMonitor){
         TH2* boostMonitor = new TH2F("BoostMonitor","ROC Integral Vs iTree",2,0,fNTrees,2,0,1.05);
         boostMonitor->SetXTitle("#tree");
         boostMonitor->SetYTitle("ROC Integral");
         results->Store(boostMonitor, "BoostMonitor");
         TGraph *boostMonitorGraph = new TGraph();
         boostMonitorGraph->SetName("BoostMonitorGraph");
         boostMonitorGraph->SetTitle("ROCIntegralVsNTrees");
         results->Store(boostMonitorGraph, "BoostMonitorGraph");
      }

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
      nodesBeforePruningVsTree->SetXTitle("#tree");
      nodesBeforePruningVsTree->SetYTitle("#tree nodes");
      results->Store(nodesBeforePruningVsTree);
      
      // nNodesAfterPruning vs tree number
      nodesAfterPruningVsTree->SetXTitle("#tree");
      nodesAfterPruningVsTree->SetYTitle("#tree nodes");
      results->Store(nodesAfterPruningVsTree);

   }
   
   fMonitorNtuple= new TTree("MonitorNtuple","BDT variables");
   fMonitorNtuple->Branch("iTree",&fITree,"iTree/I");
   fMonitorNtuple->Branch("boostWeight",&fBoostWeight,"boostWeight/D");
   fMonitorNtuple->Branch("errorFraction",&fErrorFraction,"errorFraction/D");

   Timer timer( fNTrees, GetName() );
   Int_t nNodesBeforePruningCount = 0;
   Int_t nNodesAfterPruningCount = 0;

   Int_t nNodesBeforePruning = 0;
   Int_t nNodesAfterPruning = 0;

   if(fBoostType=="Grad"){
      InitGradBoost(fEventSample);
   }

   for (int itree=0; itree<fNTrees; itree++) {
      timer.DrawProgressBar( itree );
      if(DoMulticlass()){
         if (fBoostType!="Grad"){
            Log() << kFATAL << "Multiclass is currently only supported by gradient boost. "
                  << "Please change boost option accordingly (GradBoost)."
                  << Endl;
         }
         UInt_t nClasses = DataInfo().GetNClasses();
         for (UInt_t i=0;i<nClasses;i++){
            fForest.push_back( new DecisionTree( fSepType, fNodeMinEvents, fNCuts, i,
                                                 fRandomisedTrees, fUseNvars, fUsePoissonNvars, fNNodesMax, fMaxDepth,
                                                 itree*nClasses+i, fNodePurityLimit, itree*nClasses+i));
            if (fPairNegWeightsInNode) fForest.back()->SetPairNegWeightsInNode();
            if (fUseFisherCuts) {
               fForest.back()->SetUseFisherCuts();
               fForest.back()->SetMinLinCorrForFisher(fMinLinCorrForFisher); 
               fForest.back()->SetUseExclusiveVars(fUseExclusiveVars); 
            }
            // the minimum linear correlation between two variables demanded for use in fisher criterion in node splitting

            if (fBaggedGradBoost){
               nNodesBeforePruning = fForest.back()->BuildTree(fSubSample);
               fBoostWeights.push_back(this->Boost(fSubSample, fForest.back(), itree, i));
}
            else{
               nNodesBeforePruning = fForest.back()->BuildTree(fEventSample);  
               fBoostWeights.push_back(this->Boost(fEventSample, fForest.back(), itree, i));
            }
         }
      }
      else{
         fForest.push_back( new DecisionTree( fSepType, fNodeMinEvents, fNCuts, fSignalClass,
                                              fRandomisedTrees, fUseNvars, fUsePoissonNvars, fNNodesMax, fMaxDepth,
                                              itree, fNodePurityLimit, itree));
         if (fPairNegWeightsInNode) fForest.back()->SetPairNegWeightsInNode();
         if (fUseFisherCuts) {
            fForest.back()->SetUseFisherCuts();
            fForest.back()->SetMinLinCorrForFisher(fMinLinCorrForFisher); 
            fForest.back()->SetUseExclusiveVars(fUseExclusiveVars); 
         }
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
            if(fBaggedGradBoost)
               fBoostWeights.push_back(this->Boost(fSubSample, fForest.back(), itree));
            else
               fBoostWeights.push_back(this->Boost(fEventSample, fForest.back(), itree));
         }
         else {
            if(!fPruneBeforeBoost) { // only prune after boosting
               fBoostWeights.push_back( this->Boost(fEventSample, fForest.back(), itree) );
               // if fAutomatic == true, pruneStrength will be the optimal pruning strength
               // determined by the pruning algorithm; otherwise, it is simply the strength parameter
               // set by the user
               fForest.back()->PruneTree(validationSample);
            }
            else { // prune first, then apply a boosting cycle
               fForest.back()->PruneTree(validationSample);
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
         if (fDoBoostMonitor){
            if (! DoRegression() ){
               if (  itree==fNTrees-1 ||  (!(itree%500)) ||
                     (!(itree%250) && itree <1000)||
                     (!(itree%100) && itree < 500)||
                     (!(itree%50)  && itree < 250)||
                     (!(itree%25)  && itree < 150)||
                     (!(itree%10)  && itree <  50)||
                     (!(itree%5)   && itree <  20)
                     ) BoostMonitor(itree);
            }
         }
      }
   }

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
   TMVA::DecisionTreeNode::fgIsTraining=false;
}

//_______________________________________________________________________
void TMVA::MethodBDT::GetRandomSubSample()
{
   // fills fEventSample with fSampleFraction*NEvents random training events
   UInt_t nevents = fEventSample.size();
   
   if (fSubSample.size()!=0) fSubSample.clear();
   TRandom3 *trandom   = new TRandom3(fForest.size()+1);

   for (UInt_t ievt=0; ievt<nevents; ievt++) { // recreate new random subsample
      if(trandom->Rndm()<fSampleFraction)
         fSubSample.push_back(fEventSample[ievt]);
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
void TMVA::MethodBDT::UpdateTargets(vector<TMVA::Event*> eventSample, UInt_t cls)
{
   //Calculate residua for all events;

   if(DoMulticlass()){
      UInt_t nClasses = DataInfo().GetNClasses();
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         fResiduals[*e].at(cls)+=fForest.back()->CheckEvent(*(*e),kFALSE);
         if(cls == nClasses-1){
            for(UInt_t i=0;i<nClasses;i++){
               Double_t norm = 0.0;
               for(UInt_t j=0;j<nClasses;j++){
                  if(i!=j)
                     norm+=exp(fResiduals[*e].at(j)-fResiduals[*e].at(i));
               }
               Double_t p_cls = 1.0/(1.0+norm);
               Double_t res = ((*e)->GetClass()==i)?(1.0-p_cls):(-p_cls);
               (*e)->SetTarget(i,res);
            }
         }
      }
   }
   else{
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         fResiduals[*e].at(0)+=fForest.back()->CheckEvent(*(*e),kFALSE);
         Double_t p_sig=1.0/(1.0+exp(-2.0*fResiduals[*e].at(0)));
         Double_t res = (DataInfo().IsSignal(*e)?1:0)-p_sig;
         (*e)->SetTarget(0,res);
      }
   }   
}

//_______________________________________________________________________
void TMVA::MethodBDT::UpdateTargetsRegression(vector<TMVA::Event*> eventSample, Bool_t first)
{
   //Calculate current residuals for all events and update targets for next iteration
   for (vector<TMVA::Event*>::iterator e=fEventSample.begin(); e!=fEventSample.end();e++) {
      if(!first){
         fWeightedResiduals[*e].first -= fForest.back()->CheckEvent(*(*e),kFALSE);
      }
      
   }
   
   fSumOfWeights = 0;
   vector< pair<Double_t, Double_t> > temp;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++){
      temp.push_back(make_pair(fabs(fWeightedResiduals[*e].first),fWeightedResiduals[*e].second));
      fSumOfWeights += (*e)->GetWeight();
   }
   fTransitionPoint = GetWeightedQuantile(temp,0.7,fSumOfWeights);

   Int_t i=0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
 
      if(temp[i].first<=fTransitionPoint)
         (*e)->SetTarget(0,fWeightedResiduals[*e].first);
      else
         (*e)->SetTarget(0,fTransitionPoint*(fWeightedResiduals[*e].first<0?-1.0:1.0));
      i++;
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetWeightedQuantile(vector<  pair<Double_t, Double_t> > vec, const Double_t quantile, const Double_t norm){
   //calculates the quantile of the distribution of the first pair entries weighted with the values in the second pair entries
   Double_t temp = 0.0;
   std::sort(vec.begin(), vec.end());
   Int_t i = 0;
   while(temp <= norm*quantile){
      temp += vec[i].second;
      i++;
   }
      
   return vec[i].first;
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GradBoost( vector<TMVA::Event*> eventSample, DecisionTree *dt, UInt_t cls)
{
   //Calculate the desired response value for each region
   std::map<TMVA::DecisionTreeNode*,vector<Double_t> > leaves;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t weight = (*e)->GetWeight();
      TMVA::DecisionTreeNode* node = dt->GetEventNode(*(*e));
      if ((leaves[node]).size()==0){
         (leaves[node]).push_back((*e)->GetTarget(cls)* weight);
         (leaves[node]).push_back(fabs((*e)->GetTarget(cls))*(1.0-fabs((*e)->GetTarget(cls))) * weight* weight);
      }
      else {
         (leaves[node])[0]+=((*e)->GetTarget(cls)* weight);
         (leaves[node])[1]+=fabs((*e)->GetTarget(cls))*(1.0-fabs((*e)->GetTarget(cls))) * weight* weight;
      }
   }
   for (std::map<TMVA::DecisionTreeNode*,vector<Double_t> >::iterator iLeave=leaves.begin();
        iLeave!=leaves.end();++iLeave){
      if ((iLeave->second)[1]<1e-30) (iLeave->second)[1]=1e-30;

      (iLeave->first)->SetResponse(fShrinkage/DataInfo().GetNClasses()*(iLeave->second)[0]/((iLeave->second)[1]));
   }
   
   //call UpdateTargets before next tree is grown
   if (fBaggedGradBoost){
      GetRandomSubSample();
   }
   DoMulticlass() ? UpdateTargets(fEventSample, cls) : UpdateTargets(fEventSample);
   return 1; //trees all have the same weight
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GradBoostRegression( vector<TMVA::Event*> eventSample, DecisionTree *dt )
{
   // Implementation of M_TreeBoost using a Huber loss function as desribed by Friedman 1999
   std::map<TMVA::DecisionTreeNode*,Double_t > leaveWeights;
   std::map<TMVA::DecisionTreeNode*,vector< pair<Double_t, Double_t> > > leaves;
   UInt_t i =0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      TMVA::DecisionTreeNode* node = dt->GetEventNode(*(*e));      
      (leaves[node]).push_back(make_pair(fWeightedResiduals[*e].first,(*e)->GetWeight()));
      (leaveWeights[node]) += (*e)->GetWeight();
      i++;
   }

   for (std::map<TMVA::DecisionTreeNode*,vector< pair<Double_t, Double_t> > >::iterator iLeave=leaves.begin();
        iLeave!=leaves.end();++iLeave){
      Double_t shift=0,diff= 0;
      Double_t ResidualMedian = GetWeightedQuantile(iLeave->second,0.5,leaveWeights[iLeave->first]);
      for(UInt_t j=0;j<((iLeave->second).size());j++){
         diff = (iLeave->second)[j].first-ResidualMedian;
         shift+=1.0/((iLeave->second).size())*((diff<0)?-1.0:1.0)*TMath::Min(fTransitionPoint,fabs(diff));
      }
      (iLeave->first)->SetResponse(fShrinkage*(ResidualMedian+shift));          
   }
   
   if (fBaggedGradBoost){
      GetRandomSubSample();
      UpdateTargetsRegression(fSubSample);
   }
   else
      UpdateTargetsRegression(fEventSample);
   return 1;
}

//_______________________________________________________________________
void TMVA::MethodBDT::InitGradBoost( vector<TMVA::Event*> eventSample)
{
   // initialize targets for first tree
   fSumOfWeights = 0;
   fSepType=NULL; //set fSepType to NULL (regression trees are used for both classification an regression)
   std::vector<std::pair<Double_t, Double_t> > temp;
   if(DoRegression()){
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         fWeightedResiduals[*e]= make_pair((*e)->GetTarget(0), (*e)->GetWeight());
         fSumOfWeights+=(*e)->GetWeight();
         temp.push_back(make_pair(fWeightedResiduals[*e].first,fWeightedResiduals[*e].second));
      }
      Double_t weightedMedian = GetWeightedQuantile(temp,0.5, fSumOfWeights);
     
      //Store the weighted median as a first boosweight for later use
      fBoostWeights.push_back(weightedMedian);
      std::map<TMVA::Event*, std::pair<Double_t, Double_t> >::iterator res = fWeightedResiduals.begin();
      for (; res!=fWeightedResiduals.end(); ++res ) {
         //substract the gloabl median from all residuals
         (*res).second.first -= weightedMedian;  
      }
      if (fBaggedGradBoost){
         GetRandomSubSample();
         UpdateTargetsRegression(fSubSample,kTRUE);
      }
      else
         UpdateTargetsRegression(fEventSample,kTRUE);
      return;
   }
   else if(DoMulticlass()){
      UInt_t nClasses = DataInfo().GetNClasses();
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         for (UInt_t i=0;i<nClasses;i++){
            //Calculate initial residua, assuming equal probability for all classes
            Double_t r = (*e)->GetClass()==i?(1-1.0/nClasses):(-1.0/nClasses);
            (*e)->SetTarget(i,r);
            fResiduals[*e].push_back(0);   
         }
      }
   }
   else{
      for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
         Double_t r = (DataInfo().IsSignal(*e)?1:0)-0.5; //Calculate initial residua
         (*e)->SetTarget(0,r);
         fResiduals[*e].push_back(0);         
      }
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

      if (isSignalType == (DataInfo().IsSignal(fValidationSample[ievt])) ) {
         ncorrect += fValidationSample[ievt]->GetWeight();
      }
      else{
         nfalse += fValidationSample[ievt]->GetWeight();
      }
   }

   return  ncorrect / (ncorrect + nfalse);
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::Boost( vector<TMVA::Event*> eventSample, DecisionTree *dt, Int_t iTree, UInt_t cls )
{
   // apply the boosting alogrithim (the algorithm is selecte via the the "option" given
   // in the constructor. The return value is the boosting weight

   Double_t returnVal=-1;

   if      (fBoostType=="AdaBoost")    returnVal = this->AdaBoost  (eventSample, dt);
   else if (fBoostType=="Bagging")     returnVal = this->Bagging   (eventSample, iTree);
   else if (fBoostType=="RegBoost")    returnVal = this->RegBoost  (eventSample, dt);
   else if (fBoostType=="AdaBoostR2")  returnVal = this->AdaBoostR2(eventSample, dt);
   else if (fBoostType=="Grad"){
      if(DoRegression())
         returnVal = this->GradBoostRegression(eventSample, dt);
      else if(DoMulticlass())
         returnVal = this->GradBoost (eventSample, dt, cls);
      else
         returnVal = this->GradBoost (eventSample, dt);
   }
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<Boost> unknown boost option " << fBoostType<< " called" << Endl;
   }

   return returnVal;
}

//_______________________________________________________________________
void TMVA::MethodBDT::BoostMonitor(Int_t iTree)
{
   // fills the ROCIntegral vs Itree from the testSample for the monitoring plots
   // during the training .. but using the testing events 

   TH1F *tmpS = new TH1F( "tmpS", "",     100 , -1., 1.00001 );
   TH1F *tmpB = new TH1F( "tmpB", "",     100 , -1., 1.00001 );
   TH1F *tmp;

   const std::vector<Event*> events=Data()->GetEventCollection(Types::kTesting);
   UInt_t signalClassNr = DataInfo().GetClassInfo("Signal")->GetNumber();
 
   //   fMethod->GetTransformationHandler().CalcTransformations(fMethod->Data()->GetEventCollection(Types::kTesting));
   for (UInt_t iev=0; iev < events.size() ; iev++){
      if (events[iev]->GetClass() == signalClassNr) tmp=tmpS;
      else                                          tmp=tmpB;
      tmp->Fill(PrivateGetMvaValue(*(events[iev])),events[iev]->GetWeight());
   }
   
   TMVA::PDF *sig = new TMVA::PDF( " PDF Sig", tmpS, TMVA::PDF::kSpline3 );
   TMVA::PDF *bkg = new TMVA::PDF( " PDF Bkg", tmpB, TMVA::PDF::kSpline3 );
   
   Results* results = Data()->GetResults(GetMethodName(),Types::kTraining, Types::kMaxAnalysisType);
   TGraph*  gr=results->GetGraph("BoostMonitorGraph");
   Int_t nPoints = gr->GetN();
   gr->Set(nPoints+1);
   gr->SetPoint(nPoints,(Double_t)iTree+1,GetROCIntegral(sig,bkg));

   tmpS->Delete();
   tmpB->Delete();
   
   delete sig;
   delete bkg;

   return;
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
   // and "beta" being a free parameter (standard: beta = 1) that modifies the
   // boosting.

   Double_t err=0, sumGlobalw=0, sumGlobalwfalse=0, sumGlobalwfalse2=0;

   vector<Double_t> sumw; //for individually re-scaling  each class
   map<Node*,Int_t> sigEventsInNode; // how many signal events of the training tree

   UInt_t maxCls = sumw.size();
   Double_t maxDev=0;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      Double_t w = (*e)->GetWeight();
      sumGlobalw += w;
      UInt_t iclass=(*e)->GetClass();
      if (iclass+1 > maxCls) {
	 sumw.resize(iclass+1,0);
	 maxCls = sumw.size();
      }
      sumw[iclass] += w;

      if ( DoRegression() ) {
         Double_t tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ); 
         sumGlobalwfalse += w * tmpDev;
         sumGlobalwfalse2 += w * tmpDev*tmpDev;
         if (tmpDev > maxDev) maxDev = tmpDev;
      }else{
         Bool_t isSignalType = (dt->CheckEvent(*(*e),fUseYesNoLeaf) > fNodePurityLimit );

         if (!(isSignalType == DataInfo().IsSignal(*e))) {
            sumGlobalwfalse+= w;
         }
      }
   }
   err = sumGlobalwfalse/sumGlobalw ;
   if ( DoRegression() ) {
      //if quadratic loss:
      if (fAdaBoostR2Loss=="linear"){
         err = sumGlobalwfalse/maxDev/sumGlobalw ;
      }
      else if (fAdaBoostR2Loss=="quadratic"){
         err = sumGlobalwfalse2/maxDev/maxDev/sumGlobalw ;
      }
      else if (fAdaBoostR2Loss=="exponential"){
         err = 0;
         for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
            Double_t w = (*e)->GetWeight();
            Double_t  tmpDev = TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ); 
            err += w * (1 - exp (-tmpDev/maxDev)) / sumGlobalw;
         }
         
      }
      else {
         Log() << kFATAL << " you've chosen a Loss type for Adaboost other than linear, quadratic or exponential " 
               << " namely " << fAdaBoostR2Loss << "\n" 
               << "and this is not implemented... a typo in the options ??" <<Endl;
      }
   }

   Log() << kDEBUG << "BDT AdaBoos  wrong/all: " << sumGlobalwfalse << "/" << sumGlobalw << Endl;


   Double_t newSumGlobalw=0;
   vector<Double_t> newSumw(sumw.size(),0);

   Double_t boostWeight=1.;
   if (err >= 0.5) { // sanity check ... should never happen as otherwise there is apparently
      // something odd with the assignement of the leaf nodes (rem: you use the training
      // events for this determination of the error rate)
      if (dt->GetNNodes() == 1){
         Log() << kERROR << " YOUR tree has only 1 Node... kind of a funny *tree*. I cannot " 
               << "boost such a thing... if after 1 step the error rate is == 0.5"
               << Endl
               << "please check why this happens, maybe too many events per node requested ?"
               << Endl;
         
      }else{
         Log() << kERROR << " The error rate in the BDT boosting is > 0.5. ("<< err
               << ") That should not happen, please check your code (i.e... the BDT code), I "
               << " set it to 0.5.. just to continue.." <<  Endl;
      }
      err = 0.5;
   } else if (err < 0) {
      Log() << kERROR << " The error rate in the BDT boosting is < 0. That can happen"
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
   Log() << kDEBUG << "BDT AdaBoos  wrong/all: " << sumGlobalwfalse << "/" << sumGlobalw << " 1-err/err="<<boostWeight<< " log.."<<TMath::Log(boostWeight)<<Endl;

   Results* results = Data()->GetResults(GetMethodName(),Types::kTraining, Types::kMaxAnalysisType);


   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
 
      if ((!( (dt->CheckEvent(*(*e),fUseYesNoLeaf) > fNodePurityLimit ) == DataInfo().IsSignal(*e))) || DoRegression()) {
         Double_t boostfactor = boostWeight;
         if (DoRegression()) boostfactor = TMath::Power(1/boostWeight,(1.-TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) )/maxDev ) );
         if ( (*e)->GetWeight() > 0 ){
            (*e)->SetBoostWeight( (*e)->GetBoostWeight() * boostfactor);
            // Helge change back            (*e)->ScaleBoostWeight(boostfactor);
            if (DoRegression()) results->GetHist("BoostWeights")->Fill(boostfactor);
         } else {
            if ( fInverseBoostNegWeights )(*e)->ScaleBoostWeight( 1. / boostfactor); // if the original event weight is negative, and you want to "increase" the events "positive" influence, you'd reather make the event weight "smaller" in terms of it's absolute value while still keeping it something "negative"
         }
      }
      newSumGlobalw+=(*e)->GetWeight();
      newSumw[(*e)->GetClass()] += (*e)->GetWeight();
   }


   // re-normalise the weights (independent for Signal and Background)
   Double_t globalNormWeight=sumGlobalw/newSumGlobalw;
   vector<Double_t>  normWeightByClass;
   for (UInt_t i=0; i<sumw.size(); i++) normWeightByClass.push_back(sumw[i]/newSumw[i]);

   Log() << kDEBUG << "new Nsig="<<newSumw[0]*globalNormWeight << " new Nbkg="<<newSumw[1]*globalNormWeight << Endl;


   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      if (fRenormByClass) (*e)->ScaleBoostWeight( normWeightByClass[(*e)->GetClass()] );
      else                (*e)->ScaleBoostWeight( globalNormWeight );
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
   Double_t normWeight =  eventSample.size() / newSumw ;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      (*e)->SetBoostWeight( (*e)->GetBoostWeight() * normWeight );
      // change this backwards      (*e)->ScaleBoostWeight( normWeight );
   }
   delete trandom;
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


   if (err >= 0.5) { // sanity check ... should never happen as otherwise there is apparently
      // something odd with the assignement of the leaf nodes (rem: you use the training
      // events for this determination of the error rate)
      if (dt->GetNNodes() == 1){
         Log() << kERROR << " YOUR tree has only 1 Node... kind of a funny *tree*. I cannot " 
               << "boost such a thing... if after 1 step the error rate is == 0.5"
               << Endl
               << "please check why this happens, maybe too many events per node requested ?"
               << Endl;
         
      }else{
         Log() << kERROR << " The error rate in the BDT boosting is > 0.5. ("<< err
               << ") That should not happen, but is possible for regression trees, and"
	       << " should trigger a stop for the boosting. please check your code (i.e... the BDT code), I "
               << " set it to 0.5.. just to continue.." <<  Endl;
      }
      err = 0.5;
   } else if (err < 0) {
      Log() << kERROR << " The error rate in the BDT boosting is < 0. That can happen"
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
            Log() << kINFO << "Weight=    "   <<   (*e)->GetWeight() << Endl;
            Log() << kINFO  << "BoostWeight= " <<   (*e)->GetBoostWeight() << Endl;
            Log() << kINFO  << "boostweight="<<boostWeight << "  err= " <<err << Endl; 
            Log() << kINFO  << "NewBoostWeight= " <<   newBoostWeight << Endl;
            Log() << kINFO  << "boostfactor= " <<  boostfactor << Endl;
            Log() << kINFO  << "maxDev     = " <<  maxDev << Endl;
            Log() << kINFO  << "tmpDev     = " <<  TMath::Abs(dt->CheckEvent(*(*e),kFALSE) - (*e)->GetTarget(0) ) << Endl;
            Log() << kINFO  << "target     = " <<  (*e)->GetTarget(0)  << Endl; 
            Log() << kINFO  << "estimate   = " <<  dt->CheckEvent(*(*e),kFALSE)  << Endl;
         }
         (*e)->SetBoostWeight( newBoostWeight );
         //         (*e)->SetBoostWeight( (*e)->GetBoostWeight() * boostfactor);
      } else {
         (*e)->SetBoostWeight( (*e)->GetBoostWeight() / boostfactor);
      }
      newSumw+=(*e)->GetWeight();
   }

   // re-normalise the weights
   Double_t normWeight =  sumw / newSumw;
   for (vector<TMVA::Event*>::iterator e=eventSample.begin(); e!=eventSample.end();e++) {
      //Helge    (*e)->ScaleBoostWeight( sumw/newSumw);
      // (*e)->ScaleBoostWeight( normWeight);
      (*e)->SetBoostWeight( (*e)->GetBoostWeight() * normWeight );
   }


   results->GetHist("BoostWeightsVsTree")->SetBinContent(fForest.size(),1./boostWeight);
   results->GetHist("ErrorFrac")->SetBinContent(fForest.size(),err);

   fBoostWeight = boostWeight;
   fErrorFraction = err;

   return TMath::Log(1./boostWeight);
}

//_______________________________________________________________________
void TMVA::MethodBDT::AddWeightsXMLTo( void* parent ) const
{
   // write weights to XML
   void* wght = gTools().AddChild(parent, "Weights");
   gTools().AddAttr( wght, "NTrees", fForest.size() );
   gTools().AddAttr( wght, "AnalysisType", fForest.back()->GetAnalysisType() );

   for (UInt_t i=0; i< fForest.size(); i++) {
      void* trxml = fForest[i]->AddXMLTo(wght);
      gTools().AddAttr( trxml, "boostWeight", fBoostWeights[i] );
      gTools().AddAttr( trxml, "itree", i );
   }
}

//_______________________________________________________________________
void TMVA::MethodBDT::ReadWeightsFromXML(void* parent) {
   // reads the BDT from the xml file

   UInt_t i;
   for (i=0; i<fForest.size(); i++) delete fForest[i];
   fForest.clear();
   fBoostWeights.clear();

   UInt_t ntrees;
   UInt_t analysisType;
   Float_t boostWeight;

   gTools().ReadAttr( parent, "NTrees", ntrees );
   
   if(gTools().HasAttr(parent, "TreeType")) { // pre 4.1.0 version
      gTools().ReadAttr( parent, "TreeType", analysisType );
   } else {                                 // from 4.1.0 onwards
      gTools().ReadAttr( parent, "AnalysisType", analysisType );      
   }

   void* ch = gTools().GetChild(parent);
   i=0;
   while(ch) {
      fForest.push_back( dynamic_cast<DecisionTree*>( DecisionTree::CreateFromXML(ch, GetTrainingTMVAVersionCode()) ) );
      fForest.back()->SetAnalysisType(Types::EAnalysisType(analysisType));
      fForest.back()->SetTreeID(i++);
      gTools().ReadAttr(ch,"boostWeight",boostWeight);
      fBoostWeights.push_back(boostWeight);
      ch = gTools().GetNextChild(ch);
   }
}

//_______________________________________________________________________
void  TMVA::MethodBDT::ReadWeightsFromStream( istream& istr )
{
   // read the weights (BDT coefficients)
   TString dummy;
   //   Types::EAnalysisType analysisType;
   Int_t analysisType(0);

   // coverity[tainted_data_argument]
   istr >> dummy >> fNTrees;
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
      fForest.back()->Read(istr, GetTrainingTMVAVersionCode());
      fBoostWeights.push_back(boostWeight);
   }
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue( Double_t* err, Double_t* errUpper ){
   return this->GetMvaValue( err, errUpper, 0 );
}

//_______________________________________________________________________
Double_t TMVA::MethodBDT::GetMvaValue( Double_t* err, Double_t* errUpper, UInt_t useNTrees )
{
   // Return the MVA value (range [-1;1]) that classifies the
   // event according to the majority vote from the total number of
   // decision trees.
   const Event* ev = GetEvent();
   return PrivateGetMvaValue(const_cast<TMVA::Event&>(*ev), err, errUpper, useNTrees);

}
//_______________________________________________________________________
   Double_t TMVA::MethodBDT::PrivateGetMvaValue(TMVA::Event& ev, Double_t* err, Double_t* errUpper, UInt_t useNTrees )
{
   // Return the MVA value (range [-1;1]) that classifies the
   // event according to the majority vote from the total number of
   // decision trees.

   // cannot determine error
   NoErrorCalc(err, errUpper);
   
   // allow for the possibility to use less trees in the actual MVA calculation
   // than have been originally trained.
   UInt_t nTrees = fForest.size();

   if (useNTrees > 0 ) nTrees = useNTrees;

   if (fBoostType=="Grad") return GetGradBoostMVA(ev,nTrees);
   
   Double_t myMVA = 0;
   Double_t norm  = 0;
   for (UInt_t itree=0; itree<nTrees; itree++) {
      //
      if (fUseWeightedTrees) {
         myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(ev,fUseYesNoLeaf);
         norm  += fBoostWeights[itree];
      }
      else {
         myMVA += fForest[itree]->CheckEvent(ev,fUseYesNoLeaf);
         norm  += 1;
      }
   }
   return ( norm > std::numeric_limits<double>::epsilon() ) ? myMVA /= norm : 0 ;
}

//_______________________________________________________________________
const std::vector<Float_t>& TMVA::MethodBDT::GetMulticlassValues()
{
   // get the multiclass MVA response for the BDT classifier

   const TMVA::Event& e = *GetEvent();
   if (fMulticlassReturnVal == NULL) fMulticlassReturnVal = new std::vector<Float_t>();
   fMulticlassReturnVal->clear();

   std::vector<double> temp;

   UInt_t nClasses = DataInfo().GetNClasses();
   for(UInt_t iClass=0; iClass<nClasses; iClass++){
      temp.push_back(0.0);
      for(UInt_t itree = iClass; itree<fForest.size(); itree+=nClasses){
         temp[iClass] += fForest[itree]->CheckEvent(e,kFALSE);
      }
   }    

   for(UInt_t iClass=0; iClass<nClasses; iClass++){
      Double_t norm = 0.0;
      for(UInt_t j=0;j<nClasses;j++){
         if(iClass!=j)
            norm+=exp(temp[j]-temp[iClass]);
      }
      (*fMulticlassReturnVal).push_back(1.0/(1.0+norm));
   }

   
   return *fMulticlassReturnVal;
}




//_______________________________________________________________________
const std::vector<Float_t> & TMVA::MethodBDT::GetRegressionValues()
{
   // get the regression value generated by the BDTs


   if (fRegressionReturnVal == NULL) fRegressionReturnVal = new std::vector<Float_t>();
   fRegressionReturnVal->clear();

   const Event * ev = GetEvent();
   Event * evT = new Event(*ev);

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
         response[itree]    = fForest[itree]->CheckEvent(*ev,kFALSE);
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
//      fRegressionReturnVal->push_back( rVal/Double_t(count));
      evT->SetTarget(0, rVal/Double_t(count) );
   }
   else if(fBoostType=="Grad"){
      for (UInt_t itree=0; itree<fForest.size(); itree++) {
         myMVA += fForest[itree]->CheckEvent(*ev,kFALSE);
      }
//      fRegressionReturnVal->push_back( myMVA+fBoostWeights[0]);
      evT->SetTarget(0, myMVA+fBoostWeights[0] );
   }
   else{
      for (UInt_t itree=0; itree<fForest.size(); itree++) {
         //
         if (fUseWeightedTrees) {
            myMVA += fBoostWeights[itree] * fForest[itree]->CheckEvent(*ev,kFALSE);
            norm  += fBoostWeights[itree];
         }
         else {
            myMVA += fForest[itree]->CheckEvent(*ev,kFALSE);
            norm  += 1;
         }
      }
//      fRegressionReturnVal->push_back( ( norm > std::numeric_limits<double>::epsilon() ) ? myMVA /= norm : 0 );
      evT->SetTarget(0, ( norm > std::numeric_limits<double>::epsilon() ) ? myMVA /= norm : 0 );
   }



   const Event* evT2 = GetTransformationHandler().InverseTransform( evT );
   fRegressionReturnVal->push_back( evT2->GetTarget(0) );

   delete evT;


   return *fRegressionReturnVal;
}

//_______________________________________________________________________
void  TMVA::MethodBDT::WriteMonitoringHistosToFile( void ) const
{
   // Here we could write some histograms created during the processing
   // to the output file.
   Log() << kINFO << "Write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;

   //Results* results = Data()->GetResults(GetMethodName(), Types::kTraining, Types::kMaxAnalysisType);
   //results->GetStorage()->Write();
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
   for (UInt_t ivar = 0; ivar < GetNvar(); ivar++) {
      fVariableImportance[ivar]=0;
   }
   Double_t  sum=0;
   for (int itree = 0; itree < fNTrees; itree++) {
      vector<Double_t> relativeImportance(fForest[itree]->GetVariableImportance());
      for (UInt_t i=0; i< relativeImportance.size(); i++) {
         fVariableImportance[i] += relativeImportance[i];
      }
   }
   
   for (UInt_t ivar=0; ivar< fVariableImportance.size(); ivar++){
      fVariableImportance[ivar] = TMath::Sqrt(fVariableImportance[ivar]);
      sum += fVariableImportance[ivar];
   }
   for (UInt_t ivar=0; ivar< fVariableImportance.size(); ivar++) fVariableImportance[ivar] /= sum;

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

   TString nodeName = className;
   nodeName.ReplaceAll("Read","");
   nodeName.Append("Node");
   // write BDT-specific classifier response
   fout << "   std::vector<"<<nodeName<<"*> fForest;       // i.e. root nodes of decision trees" << endl;
   fout << "   std::vector<double>                fBoostWeights; // the weights applied in the individual boosts" << endl;
   fout << "};" << endl << endl;
   fout << "double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double myMVA = 0;" << endl;
   if (fBoostType!="Grad"){
      fout << "   double norm  = 0;" << endl;
   }
   fout << "   for (unsigned int itree=0; itree<fForest.size(); itree++){" << endl;
   fout << "      "<<nodeName<<" *current = fForest[itree];" << endl;
   fout << "      while (current->GetNodeType() == 0) { //intermediate node" << endl;
   fout << "         if (current->GoesRight(inputValues)) current=("<<nodeName<<"*)current->GetRight();" << endl;
   fout << "         else current=("<<nodeName<<"*)current->GetLeft();" << endl;
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
void TMVA::MethodBDT::MakeClassSpecificHeader(  std::ostream& fout, const TString& className) const
{
   // specific class header
   TString nodeName = className;
   nodeName.ReplaceAll("Read","");
   nodeName.Append("Node");
   //fout << "#ifndef NN" << endl; commented out on purpose see next line
   fout << "#define NN new "<<nodeName << endl; // NN definition depends on individual methods. Important to have NO #ifndef if several BDT methods compile together
   //fout << "#endif" << endl; commented out on purpose see previous line
   fout << "   " << endl;
   fout << "#ifndef "<<nodeName<<"__def" << endl;
   fout << "#define "<<nodeName<<"__def" << endl;
   fout << "   " << endl;
   fout << "class "<<nodeName<<" {" << endl;
   fout << "   " << endl;
   fout << "public:" << endl;
   fout << "   " << endl;
   fout << "   // constructor of an essentially \"empty\" node floating in space" << endl;
   fout << "   "<<nodeName<<" ( "<<nodeName<<"* left,"<<nodeName<<"* right," << endl;
   if (fUseFisherCuts){
      fout << "                          int nFisherCoeff," << endl;
      for (UInt_t i=0;i<GetNVariables()+1;i++){
         fout << "                          double fisherCoeff"<<i<<"," << endl;
      }
   }
   fout << "                          int selector, double cutValue, bool cutType, " << endl;
   fout << "                          int nodeType, double purity, double response ) :" << endl;
   fout << "   fLeft         ( left         )," << endl;
   fout << "   fRight        ( right        )," << endl;
   if (fUseFisherCuts) fout << "   fNFisherCoeff ( nFisherCoeff )," << endl;
   fout << "   fSelector     ( selector     )," << endl;
   fout << "   fCutValue     ( cutValue     )," << endl;
   fout << "   fCutType      ( cutType      )," << endl;
   fout << "   fNodeType     ( nodeType     )," << endl;
   fout << "   fPurity       ( purity       )," << endl;
   fout << "   fResponse     ( response     ){" << endl;
   if (fUseFisherCuts){
      for (UInt_t i=0;i<GetNVariables()+1;i++){
         fout << "     fFisherCoeff.push_back(fisherCoeff"<<i<<");" << endl;
      }
   }
   fout << "   }" << endl << endl;
   fout << "   virtual ~"<<nodeName<<"();" << endl << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;
   fout << "   virtual bool GoesRight( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   "<<nodeName<<"* GetRight( void )  {return fRight; };" << endl << endl;
   fout << "   // test event if it decends the tree at this node to the left " << endl;
   fout << "   virtual bool GoesLeft ( const std::vector<double>& inputValues ) const;" << endl;
   fout << "   "<<nodeName<<"* GetLeft( void ) { return fLeft; };   " << endl << endl;
   fout << "   // return  S/(S+B) (purity) at this node (from  training)" << endl << endl;
   fout << "   double GetPurity( void ) const { return fPurity; } " << endl;
   fout << "   // return the node type" << endl;
   fout << "   int    GetNodeType( void ) const { return fNodeType; }" << endl;
   fout << "   double GetResponse(void) const {return fResponse;}" << endl << endl;
   fout << "private:" << endl << endl;
   fout << "   "<<nodeName<<"*   fLeft;     // pointer to the left daughter node" << endl;
   fout << "   "<<nodeName<<"*   fRight;    // pointer to the right daughter node" << endl;
   if (fUseFisherCuts){
      fout << "   int                     fNFisherCoeff; // =0 if this node doesn use fisher, else =nvar+1 " << endl;
      fout << "   std::vector<double>     fFisherCoeff;  // the fisher coeff (offset at the last element)" << endl;
   }
   fout << "   int                     fSelector; // index of variable used in node selection (decision tree)   " << endl;
   fout << "   double                  fCutValue; // cut value appplied on this node to discriminate bkg against sig" << endl;
   fout << "   bool                    fCutType;  // true: if event variable > cutValue ==> signal , false otherwise" << endl;
   fout << "   int                     fNodeType; // Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal " << endl;
   fout << "   double                  fPurity;   // Purity of node from training"<< endl;
   fout << "   double                  fResponse; // Regression response value of node" << endl;
   fout << "}; " << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "   "<<nodeName<<"::~"<<nodeName<<"()" << endl;
   fout << "{" << endl;
   fout << "   if (fLeft  != NULL) delete fLeft;" << endl;
   fout << "   if (fRight != NULL) delete fRight;" << endl;
   fout << "}; " << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "bool "<<nodeName<<"::GoesRight( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   // test event if it decends the tree at this node to the right" << endl;
   fout << "   bool result;" << endl;
   if (fUseFisherCuts){
     fout << "   if (fNFisherCoeff == 0){" << endl;
     fout << "     result = (inputValues[fSelector] > fCutValue );" << endl;
     fout << "   }else{" << endl;
     fout << "     double fisher = fFisherCoeff.at(fFisherCoeff.size()-1);" << endl;
     fout << "     for (unsigned int ivar=0; ivar<fFisherCoeff.size()-1; ivar++)" << endl;
     fout << "       fisher += fFisherCoeff.at(ivar)*inputValues.at(ivar);" << endl;
     fout << "     result = fisher > fCutValue;" << endl;
     fout << "   }" << endl;
   }else{
     fout << "     result = (inputValues[fSelector] > fCutValue );" << endl;
   }
   fout << "   if (fCutType == true) return result; //the cuts are selecting Signal ;" << endl;
   fout << "   else return !result;" << endl;
   fout << "}" << endl;
   fout << "   " << endl;
   fout << "//_______________________________________________________________________" << endl;
   fout << "bool "<<nodeName<<"::GoesLeft( const std::vector<double>& inputValues ) const" << endl;
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
        << setprecision(6);
   if (fUseFisherCuts){
     fout << n->GetNFisherCoeff() << ", ";
     for (UInt_t i=0; i< GetNVariables()+1; i++) {
       if (n->GetNFisherCoeff() == 0 ){
         fout <<  "0, ";
       }else{
         fout << n->GetFisherCoeff(i) << ", ";
       }
     }
   }
   fout << n->GetSelector() << ", "
        << n->GetCutValue() << ", "
        << n->GetCutType() << ", "
        << n->GetNodeType() << ", "
        << n->GetPurity() << ","
        << n->GetResponse() << ") ";
}
