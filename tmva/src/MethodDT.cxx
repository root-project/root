// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDT (DT = Decision Trees)                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Analysis of Boosted Decision Trees                                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Or Cohen        <orcohenor@gmail.com>    - Weizmann Inst., Israel         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
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
// successive decision nodes are used to categorize the
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
// the idea behind the boosting is, that signal events from the training
// sample, that *end up in a background node (and vice versa) are given a
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
// stochasitc re-sampling of the initial training event sample.
//
// Analysis:
//
// applying an individual decision tree to a test event results in a
// classification of the event as either signal or background. For the
// boosted decision tree selection, an event is successively subjected to
// the whole set of decision trees and depending on how often it is
// classified as signal, a "likelihood" estimator is constructed for the
// event being signal or background. The value of this estimator is the
// one which is then used to select the events from an event sample, and
// the cut value on this estimator defines the efficiency and purity of
// the selection.
//*
//_______________________________________________________________________

#include <algorithm>
#include "Riostream.h"
#include "TRandom3.h"
#include "TMath.h"
#include "TObjString.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodDT.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Ranking.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/BinarySearchTree.h"
#include "TMVA/SeparationBase.h"
#include "TMVA/GiniIndex.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/MethodBoost.h"
#include "TMVA/CCPruner.h"

using std::vector;

REGISTER_METHOD(DT)

ClassImp(TMVA::MethodDT)

//_______________________________________________________________________
TMVA::MethodDT::MethodDT( const TString& jobName,
                          const TString& methodTitle,
                          DataSetInfo& theData,
                          const TString& theOption,
                          TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kDT, methodTitle, theData, theOption, theTargetDir )
   , fTree(0)
   , fNodeMinEvents(0)
   , fNCuts(0)
   , fUseYesNoLeaf(kFALSE)
   , fNodePurityLimit(0)
   , fNNodesMax(0)
   , fMaxDepth(0)
   , fErrorFraction(0)
   , fPruneStrength(0)
   , fPruneMethod(DecisionTree::kNoPruning)
   , fAutomatic(kFALSE)
   , fRandomisedTrees(kFALSE)
   , fUseNvars(0)
   , fPruneBeforeBoost(kFALSE)
   , fDeltaPruneStrength(0)
{
   // the standard constructor for just an ordinar "decision trees"
}

//_______________________________________________________________________
TMVA::MethodDT::MethodDT( DataSetInfo& dsi,
                          const TString& theWeightFile,
                          TDirectory* theTargetDir ) :
   TMVA::MethodBase( Types::kDT, dsi, theWeightFile, theTargetDir )
   , fTree(0)
   , fNodeMinEvents(0)
   , fNCuts(0)
   , fUseYesNoLeaf(kFALSE)
   , fNodePurityLimit(0)
   , fNNodesMax(0)
   , fMaxDepth(0)
   , fErrorFraction(0)
   , fPruneStrength(0)
   , fPruneMethod(DecisionTree::kNoPruning)
   , fAutomatic(kFALSE)
   , fRandomisedTrees(kFALSE)
   , fUseNvars(0)
   , fPruneBeforeBoost(kFALSE)
   , fDeltaPruneStrength(0)
{
   //constructor from Reader
}

//_______________________________________________________________________
Bool_t TMVA::MethodDT::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // FDA can handle classification with 2 classes and regression with one regression-target
   if( type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void TMVA::MethodDT::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // UseRandomisedTrees  choose at each node splitting a random set of variables 
   // UseNvars         use UseNvars variables in randomised trees
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
   // PruneMethod      The Pruning method: 
   //                  known: NoPruning  // switch off pruning completely
   //                         ExpectedError
   //                         CostComplexity 
   // PruneStrength    a parameter to adjust the amount of pruning. Should be large enouth such that overtraining is avoided");

   DeclareOptionRef(fRandomisedTrees,"UseRandomisedTrees","Choose at each node splitting a random set of variables and *bagging*");
   DeclareOptionRef(fUseNvars,"UseNvars","Number of variables used if randomised Tree option is chosen");
   DeclareOptionRef(fUseYesNoLeaf=kTRUE, "UseYesNoLeaf", 
                    "Use Sig or Bkg node type or the ratio S/B as classification in the leaf node");
   DeclareOptionRef(fNodePurityLimit=0.5, "NodePurityLimit", "In boosting/pruning, nodes with purity > NodePurityLimit are signal; background otherwise.");
   DeclareOptionRef(fPruneBeforeBoost=kFALSE, "PruneBeforeBoost", 
                    "Whether to perform the prune process right after the training or after the boosting");
   DeclareOptionRef(fSepTypeS="GiniIndex", "SeparationType", "Separation criterion for node splitting");
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   DeclareOptionRef(fNodeMinEvents, "nEventsMin", "Minimum number of events in a leaf node (default: max(20, N_train/(Nvar^2)/10) ) ");
   DeclareOptionRef(fNCuts, "nCuts", "Number of steps during node cut optimisation");
   DeclareOptionRef(fPruneStrength, "PruneStrength", "Pruning strength (negative value == automatic adjustment)");
   DeclareOptionRef(fPruneMethodS, "PruneMethod", "Pruning method: NoPruning (switched off), ExpectedError or CostComplexity");
   
   AddPreDefVal(TString("NoPruning"));
   AddPreDefVal(TString("ExpectedError"));
   AddPreDefVal(TString("CostComplexity"));

   DeclareOptionRef(fNNodesMax=100000,"NNodesMax","Max number of nodes in tree");
   if (DoRegression()) {
      DeclareOptionRef(fMaxDepth=50,"MaxDepth","Max depth of the decision tree allowed");
   }else{
      DeclareOptionRef(fMaxDepth=3,"MaxDepth","Max depth of the decision tree allowed");
   }
}

//_______________________________________________________________________
void TMVA::MethodDT::ProcessOptions() 
{
   // the option string is decoded, for available options see "DeclareOptions"
   fSepTypeS.ToLower();
   if      (fSepTypeS == "misclassificationerror") fSepType = new MisClassificationError();
   else if (fSepTypeS == "giniindex")              fSepType = new GiniIndex();
   else if (fSepTypeS == "crossentropy")           fSepType = new CrossEntropy();
   else if (fSepTypeS == "sdivsqrtsplusb")         fSepType = new SdivSqrtSplusB();
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown Separation Index option called" << Endl;
   }     

   //   std::cout << "fSeptypes " << fSepTypeS << "  fseptype " << fSepType << std::endl;

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror" )   fPruneMethod = DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity" )  fPruneMethod = DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "nopruning" )       fPruneMethod = DecisionTree::kNoPruning;
   else {
      Log() << kINFO << GetOptions() << Endl;
      Log() << kFATAL << "<ProcessOptions> unknown PruneMethod option called" << Endl;
   }

   if (fPruneStrength < 0) fAutomatic = kTRUE;
   else fAutomatic = kFALSE;
   if (fAutomatic && fPruneMethod==!DecisionTree::kCostComplexityPruning){
      Log() << kFATAL 
            <<  "Sorry autmoatic pruning strength determination is not implemented yet for ExpectedErrorPruning" << Endl;
   }


   if (this->Data()->HasNegativeEventWeights()){
      Log() << kINFO << " You are using a Monte Carlo that has also negative weights. "
              << "That should in principle be fine as long as on average you end up with "
              << "something positive. For this you have to make sure that the minimal number "
              << "of (unweighted) events demanded for a tree node (currently you use: nEventsMin="
              <<fNodeMinEvents<<", you can set this via the BDT option string when booking the "
              << "classifier) is large enough to allow for reasonable averaging!!! "
              << " If this does not help.. maybe you want to try the option: NoNegWeightsInTraining  "
              << "which ignores events with negative weight in the training. " << Endl
              << Endl << "Note: You'll get a WARNING message during the training if that should ever happen" << Endl;
   }
   
   if (fRandomisedTrees){
      Log() << kINFO << " Randomised trees should use *bagging* as *boost* method. Did you set this in the *MethodBoost* ? . Here I can enforce only the *no pruning*" << Endl;
      fPruneMethod = DecisionTree::kNoPruning;
      //      fBoostType   = "Bagging";
   }

}

//_______________________________________________________________________
void TMVA::MethodDT::Init( void )
{
   // common initialisation with defaults for the DT-Method
   fNodeMinEvents  = TMath::Max( 20, int( Data()->GetNTrainingEvents() / (10*GetNvar()*GetNvar())) );
   fNCuts          = 20; 
   fPruneMethod    = DecisionTree::kNoPruning;
   fPruneStrength  = 5;     // means automatic determination of the prune strength using a validation sample  
   fDeltaPruneStrength=0.1;
   fRandomisedTrees= kFALSE;
   fUseNvars       = GetNvar();

   // reference cut value to distingiush signal-like from background-like events   
   SetSignalReferenceCut( 0 );
   if (fAnalysisType == Types::kClassification || fAnalysisType == Types::kMulticlass ) {
      fMaxDepth        = 3;
   }else {
      fMaxDepth = 50;
   }
}

//_______________________________________________________________________
TMVA::MethodDT::~MethodDT( void )
{
   //destructor
   delete fTree;
}

//_______________________________________________________________________
void TMVA::MethodDT::Train( void )
{
   TMVA::DecisionTreeNode::fgIsTraining=true;
   fTree = new DecisionTree( fSepType, fNodeMinEvents, fNCuts, 0, 
                             fRandomisedTrees, fUseNvars, fNNodesMax, fMaxDepth,0 );
   if (fRandomisedTrees) Log()<<kWARNING<<" randomised Trees do not work yet in this framework," 
                                << " as I do not know how to give each tree a new random seed, now they"
                                << " will be all the same and that is not good " << Endl;
   fTree->SetAnalysisType( GetAnalysisType() );

   fTree->BuildTree(GetEventCollection(Types::kTraining));
   TMVA::DecisionTreeNode::fgIsTraining=false;
}

//_______________________________________________________________________
Bool_t TMVA::MethodDT::MonitorBoost( MethodBoost* booster )
{
   Int_t methodIndex = booster->GetMethodIndex();
   if (booster->GetBoostStage() == Types::kBoostProcBegin)
      {
         booster->AddMonitoringHist(new TH1I("NodesBeforePruning","nodes before pruning",booster->GetBoostNum(),0,booster->GetBoostNum()));
         booster->AddMonitoringHist(new TH1I("NodesAfterPruning","nodes after pruning",booster->GetBoostNum(),0,booster->GetBoostNum()));
         booster->AddMonitoringHist(new TH1D("PruneStrength","prune  strength",booster->GetBoostNum(),0,booster->GetBoostNum()));
      }

   if (booster->GetBoostStage() == Types::kBeforeTraining)
      {
         if (methodIndex == 0)
            {
               booster->GetMonitoringHist(2)->SetXTitle("#tree");
               booster->GetMonitoringHist(2)->SetYTitle("PruneStrength");
               //dividing the data set for pruning where strength is calculated automatically
               if (fAutomatic)
                  {
                     Data()->DivideTrainingSet(2);
                     Data()->MoveTrainingBlock(1,Types::kValidation,kTRUE);
                  }
            }
      }
   else if (booster->GetBoostStage() == Types::kBeforeBoosting)
      booster->GetMonitoringHist(0)->SetBinContent(booster->GetBoostNum()+1,fTree->GetNNodes());

   if (booster->GetBoostStage() == ((fPruneBeforeBoost)?Types::kBeforeBoosting:Types::kBoostValidation)
       && !(fPruneMethod == DecisionTree::kNoPruning)) {
      
      if (methodIndex==0 && fPruneBeforeBoost == kFALSE)
         Log() << kINFO << "Pruning "<< booster->GetBoostNum() << " Decision Trees ... patience please" << Endl;
         
      //reading the previous value
      if (fAutomatic && methodIndex > 0) {
         MethodDT* mdt = dynamic_cast<MethodDT*>(booster->GetPreviousMethod());
         if(mdt)
            fPruneStrength = mdt->GetPruneStrength();
      }

      booster->GetMonitoringHist(0)->SetBinContent(methodIndex+1,fTree->GetNNodes());
      booster->GetMonitoringHist(2)->SetBinContent(methodIndex+1,PruneTree(methodIndex));
      booster->GetMonitoringHist(1)->SetBinContent(methodIndex+1,fTree->GetNNodes());
   } // no pruning is performed
   else if (booster->GetBoostStage() != Types::kBoostProcEnd)
      return kFALSE;

   //finishing the pruning process, printing out everything
   if (booster->GetBoostStage() == Types::kBoostProcEnd)
      {
         if (fPruneMethod == DecisionTree::kNoPruning) {
            Log() << kINFO << "<Train> average number of nodes (w/o pruning) : "
                    <<  booster->GetMonitoringHist(0)->GetMean() << Endl;
         }
         else
            {
               Log() << kINFO << "<Train> average number of nodes before/after pruning : " 
                       << booster->GetMonitoringHist(0)->GetMean() << " / " 
                       << booster->GetMonitoringHist(1)->GetMean()
                       << Endl;
            }
      }

   return kTRUE;
}


//_______________________________________________________________________
Double_t TMVA::MethodDT::PruneTree(const Int_t methodIndex)
{
   if (fAutomatic && fPruneMethod == DecisionTree::kCostComplexityPruning) { // automatic cost complexity pruning
      CCPruner* pruneTool = new CCPruner(fTree, this->Data() , fSepType);
      pruneTool->Optimize();
      std::vector<DecisionTreeNode*> nodes = pruneTool->GetOptimalPruneSequence();
      fPruneStrength = pruneTool->GetOptimalPruneStrength();
      for(UInt_t i = 0; i < nodes.size(); i++) 
         fTree->PruneNode(nodes[i]);
      delete pruneTool;
   } 
   else if (fAutomatic &&  fPruneMethod != DecisionTree::kCostComplexityPruning){
      Int_t bla; 
      bla = methodIndex; //make the compiler quiet
      /*
      Double_t alpha = 0;
      Double_t delta = fDeltaPruneStrength;
      
      DecisionTree*  dcopy;
      vector<Double_t> q;
      multimap<Double_t,Double_t> quality;
      Int_t nnodes=fTree->GetNNodes();

      // find the maxiumum prune strength that still leaves some nodes 
      Bool_t forceStop = kFALSE;
      Int_t troubleCount=0, previousNnodes=nnodes;


      nnodes=fTree->GetNNodes();
      while (nnodes > 3 && !forceStop) {
         dcopy = new DecisionTree(*fTree);
         dcopy->SetPruneStrength(alpha+=delta);
         dcopy->PruneTree();
         q.push_back(TestTreeQuality(dcopy));
         quality.insert(pair<const Double_t,Double_t>(q.back(),alpha));
         nnodes=dcopy->GetNNodes();
         if (previousNnodes == nnodes) troubleCount++;
         else { 
            troubleCount=0; // reset counter
            if (nnodes < previousNnodes / 2 ) fDeltaPruneStrength /= 2.;
         }
         previousNnodes = nnodes;
         if (troubleCount > 20) {
            if (methodIndex == 0 && fPruneStrength <=0) {//maybe you need larger stepsize ??
               fDeltaPruneStrength *= 5;
               Log() << kINFO << "<PruneTree> trouble determining optimal prune strength"
                       << " for Tree " << methodIndex
                       << " --> first try to increase the step size"
                       << " currently Prunestrenght= " << alpha 
                       << " stepsize " << fDeltaPruneStrength << " " << Endl;
               troubleCount = 0;   // try again
               fPruneStrength = 1; // if it was for the first time.. 
            } else if (methodIndex == 0 && fPruneStrength <=2) {//maybe you need much larger stepsize ??
               fDeltaPruneStrength *= 5;
               Log() << kINFO << "<PruneTree> trouble determining optimal prune strength"
                       << " for Tree " << methodIndex
                       << " -->  try to increase the step size even more.. "
                       << " if that still didn't work, TRY IT BY HAND"  
                       << " currently Prunestrenght= " << alpha 
                       << " stepsize " << fDeltaPruneStrength << " " << Endl;
               troubleCount = 0;   // try again
               fPruneStrength = 3; // if it was for the first time.. 
            } else {
               forceStop=kTRUE;
               Log() << kINFO << "<PruneTree> trouble determining optimal prune strength"
                       << " for Tree " << methodIndex << " at tested prune strength: " << alpha << " --> abort forced, use same strength as for previous tree:"
                       << fPruneStrength << Endl;
            }
         }
         if (fgDebugLevel==1) Log() << kINFO << "Pruneed with ("<<alpha
                                      << ") give quality: " << q.back()
                                      << " and #nodes: " << nnodes  
                                      << Endl;
         delete dcopy;
      }
      if (!forceStop) {
         multimap<Double_t,Double_t>::reverse_iterator it=quality.rend();
         it++;
         fPruneStrength = it->second;
         // adjust the step size for the next tree.. think that 20 steps are sort of
         // fine enough.. could become a tunable option later..
         fDeltaPruneStrength *= Double_t(q.size())/20.;
      }

      fTree->SetPruneStrength(fPruneStrength);
      fTree->PruneTree();
      */
   } 
   else {
      fTree->SetPruneStrength(fPruneStrength);
      fTree->PruneTree();
   }
   return fPruneStrength;
}

//_______________________________________________________________________
Double_t TMVA::MethodDT::TestTreeQuality( DecisionTree *dt )
{
   Data()->SetCurrentType(Types::kValidation);
   // test the tree quality.. in terms of Miscalssification
   Double_t SumCorrect=0,SumWrong=0;
   for (Long64_t ievt=0; ievt<Data()->GetNEvents(); ievt++)
      {
         Event * ev = Data()->GetEvent(ievt);
         if ((dt->CheckEvent(*ev) > dt->GetNodePurityLimit() ) == DataInfo().IsSignal(ev)) SumCorrect+=ev->GetWeight();
         else SumWrong+=ev->GetWeight();
      }
   Data()->SetCurrentType(Types::kTraining);
   return  SumCorrect / (SumCorrect + SumWrong);
}

//_______________________________________________________________________
void TMVA::MethodDT::AddWeightsXMLTo( void* parent ) const 
{
   fTree->AddXMLTo(parent);
   //Log() << kFATAL << "Please implement writing of weights as XML" << Endl;
}

//_______________________________________________________________________
void TMVA::MethodDT::ReadWeightsFromXML( void* wghtnode)
{
   if(fTree)
      delete fTree;
   fTree = new DecisionTree();
   fTree->ReadXML(wghtnode,GetTrainingTMVAVersionCode());
}

//_______________________________________________________________________
void  TMVA::MethodDT::ReadWeightsFromStream( istream& istr )
{
   delete fTree;
   fTree = new DecisionTree();
   fTree->Read(istr);
}

//_______________________________________________________________________
Double_t TMVA::MethodDT::GetMvaValue( Double_t* err, Double_t* errUpper )
{
   // returns MVA value

   // cannot determine error
   NoErrorCalc(err, errUpper);

   return fTree->CheckEvent(*GetEvent(),fUseYesNoLeaf);
}

//_______________________________________________________________________
void TMVA::MethodDT::GetHelpMessage() const
{

}
//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodDT::CreateRanking()
{
   return 0;
}
