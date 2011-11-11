// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne, Jan Therhaag

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DecisionTree                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of a Decision Tree                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>          - U of Bonn, Germany        *
 *      Jan Therhaag          <Jan.Therhaag@cern.ch>   - U of Bonn, Germany       *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *               
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of a Decision Tree
//
// In a decision tree successive decision nodes are used to categorize the
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
//_______________________________________________________________________

#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <fstream>
#include <algorithm>
#include <cassert>

#include "TRandom3.h"
#include "TMath.h"
#include "TMatrix.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/DecisionTree.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/BinarySearchTree.h"

#include "TMVA/Tools.h"

#include "TMVA/GiniIndex.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/Event.h"
#include "TMVA/BDTEventWrapper.h"
#include "TMVA/IPruneTool.h"
#include "TMVA/CostComplexityPruneTool.h"
#include "TMVA/ExpectedErrorPruneTool.h"

const Int_t TMVA::DecisionTree::fgRandomSeed = 0; // set nonzero for debugging and zero for random seeds

using std::vector;

ClassImp(TMVA::DecisionTree)

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree():
   BinaryTree(),
   fNvars          (0),
   fNCuts          (-1),
   fUseFisherCuts  (kFALSE),
   fMinLinCorrForFisher (1),
   fUseExclusiveVars (kTRUE),
   fSepType        (NULL),
   fRegType        (NULL),
   fMinSize        (0),
   fMinSepGain (0),
   fUseSearchTree(kFALSE),
   fPruneStrength(0),
   fPruneMethod    (kNoPruning),
   fNodePurityLimit(0.5),
   fRandomisedTree (kFALSE),
   fUseNvars       (0),
   fUsePoissonNvars(kFALSE),
   fMyTrandom (NULL), 
   fNNodesMax      (999999),
   fMaxDepth       (999999),
   fSigClass       (0),
   fPairNegWeightsInNode(kFALSE),
   fTreeID         (0),
   fAnalysisType   (Types::kClassification)
{
   // default constructor using the GiniIndex as separation criterion,
   // no restrictions on minium number of events in a leave note or the
   // separation gain in the node splitting
}

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( TMVA::SeparationBase *sepType, Int_t minSize, Int_t nCuts, UInt_t cls,
                                  Bool_t randomisedTree, Int_t useNvars, Bool_t usePoissonNvars, UInt_t nNodesMax,
                                  UInt_t nMaxDepth, Int_t iSeed, Float_t purityLimit, Int_t treeID):
   BinaryTree(),
   fNvars          (0),
   fNCuts          (nCuts),
   fUseFisherCuts  (kFALSE),
   fMinLinCorrForFisher (1),
   fUseExclusiveVars (kTRUE),
   fSepType        (sepType),
   fRegType        (NULL),
   fMinSize        (minSize),
   fMinSepGain     (0),
   fUseSearchTree  (kFALSE),
   fPruneStrength  (0),
   fPruneMethod    (kNoPruning),
   fNodePurityLimit(purityLimit),
   fRandomisedTree (randomisedTree),
   fUseNvars       (useNvars),
   fUsePoissonNvars(usePoissonNvars),
   fMyTrandom      (new TRandom3(iSeed)),
   fNNodesMax      (nNodesMax),
   fMaxDepth       (nMaxDepth),
   fSigClass       (cls),
   fPairNegWeightsInNode(kFALSE),
   fTreeID         (treeID)
{
   // constructor specifying the separation type, the min number of
   // events in a no that is still subjected to further splitting, the
   // number of bins in the grid used in applying the cut for the node
   // splitting.

   if (sepType == NULL) { // it is interpreted as a regression tree, where
                          // currently the separation type (simple least square)
                          // cannot be chosen freely)
      fAnalysisType = Types::kRegression;
      fRegType = new RegressionVariance();
      if ( nCuts <=0 ) {
         fNCuts = 200;
         Log() << kWARNING << " You had choosen the training mode using optimal cuts, not\n"
               << " based on a grid of " << fNCuts << " by setting the option NCuts < 0\n"
               << " as this doesn't exist yet, I set it to " << fNCuts << " and use the grid"
               << Endl;
      }
   }else{
      fAnalysisType = Types::kClassification;
   }
}

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( const DecisionTree &d ):
   BinaryTree(),
   fNvars      (d.fNvars),
   fNCuts      (d.fNCuts),
   fUseFisherCuts  (d.fUseFisherCuts),
   fMinLinCorrForFisher (d.fMinLinCorrForFisher),
   fUseExclusiveVars (d.fUseExclusiveVars),
   fSepType    (d.fSepType),
   fRegType    (d.fRegType),
   fMinSize    (d.fMinSize),
   fMinSepGain (d.fMinSepGain),
   fUseSearchTree  (d.fUseSearchTree),
   fPruneStrength  (d.fPruneStrength),
   fPruneMethod    (d.fPruneMethod),
   fNodePurityLimit(d.fNodePurityLimit),
   fRandomisedTree (d.fRandomisedTree),
   fUseNvars       (d.fUseNvars),
   fUsePoissonNvars(d.fUsePoissonNvars),
   fMyTrandom      (new TRandom3(fgRandomSeed)),  // well, that means it's not an identical copy. But I only ever intend to really copy trees that are "outgrown" already. 
   fNNodesMax  (d.fNNodesMax),
   fMaxDepth   (d.fMaxDepth),
   fSigClass   (d.fSigClass),
   fPairNegWeightsInNode(d.fPairNegWeightsInNode),
   fTreeID     (d.fTreeID),
   fAnalysisType(d.fAnalysisType)
{
   // copy constructor that creates a true copy, i.e. a completely independent tree
   // the node copy will recursively copy all the nodes
   this->SetRoot( new TMVA::DecisionTreeNode ( *((DecisionTreeNode*)(d.GetRoot())) ) );
   this->SetParentTreeInNodes();
   fNNodes = d.fNNodes;
}


//_______________________________________________________________________
TMVA::DecisionTree::~DecisionTree()
{
   // destructor

   // destruction of the tree nodes done in the "base class" BinaryTree

   if (fMyTrandom) delete fMyTrandom;
}

//_______________________________________________________________________
void TMVA::DecisionTree::SetParentTreeInNodes( Node *n )
{
   // descend a tree to find all its leaf nodes, fill max depth reached in the
   // tree at the same time.

   if (n == NULL) { //default, start at the tree top, then descend recursively
      n = this->GetRoot();
      if (n == NULL) {
         Log() << kFATAL << "SetParentTreeNodes: started with undefined ROOT node" <<Endl;
         return ;
      }
   }

   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) != NULL) ) {
      Log() << kFATAL << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   }  else if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) == NULL) ) {
      Log() << kFATAL << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   }
   else {
      if (this->GetLeftDaughter(n) != NULL) {
         this->SetParentTreeInNodes( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         this->SetParentTreeInNodes( this->GetRightDaughter(n) );
      }
   }
   n->SetParentTree(this);
   if (n->GetDepth() > this->GetTotalTreeDepth()) this->SetTotalTreeDepth(n->GetDepth());
   return;
}

//_______________________________________________________________________
TMVA::DecisionTree* TMVA::DecisionTree::CreateFromXML(void* node, UInt_t tmva_Version_Code ) {
   // re-create a new tree (decision tree or search tree) from XML
   std::string type("");
   gTools().ReadAttr(node,"type", type);
   DecisionTree* dt = new DecisionTree();

   dt->ReadXML( node, tmva_Version_Code );
   return dt;
}


//_______________________________________________________________________
UInt_t TMVA::DecisionTree::BuildTree( const vector<TMVA::Event*> & eventSample,
                                      TMVA::DecisionTreeNode *node)
{
   // building the decision tree by recursively calling the splitting of
   // one (root-) node into two daughter nodes (returns the number of nodes)

   // Bool_t IsRootNode=kFALSE;
   if (node==NULL) {
      // IsRootNode = kTRUE;
      //start with the root node
      node = new TMVA::DecisionTreeNode();
      fNNodes = 1;
      this->SetRoot(node);
      // have to use "s" for start as "r" for "root" would be the same as "r" for "right"
      this->GetRoot()->SetPos('s');
      this->GetRoot()->SetDepth(0);
      this->GetRoot()->SetParentTree(this);
   }

   UInt_t nevents = eventSample.size();

   if (nevents > 0 ) {
      fNvars = eventSample[0]->GetNVariables();
      fVariableImportance.resize(fNvars);
   }
   else Log() << kFATAL << ":<BuildTree> eventsample Size == 0 " << Endl;

   Double_t s=0, b=0;
   Double_t suw=0, buw=0;
   Double_t target=0, target2=0;
   Float_t *xmin = new Float_t[fNvars];
   Float_t *xmax = new Float_t[fNvars];
   for (UInt_t ivar=0; ivar<fNvars; ivar++) {
      xmin[ivar]=xmax[ivar]=0;
   }
   for (UInt_t iev=0; iev<eventSample.size(); iev++) {
      const TMVA::Event* evt = eventSample[iev];
      const Double_t weight = evt->GetWeight();
      if (evt->GetClass() == fSigClass) {
         s += weight;
         suw += 1;
      }
      else {
         b += weight;
         buw += 1;
      }
      if ( DoRegression() ) {
         const Double_t tgt = evt->GetTarget(0);
         target +=weight*tgt;
         target2+=weight*tgt*tgt;
      }

      for (UInt_t ivar=0; ivar<fNvars; ivar++) {
         const Double_t val = evt->GetValue(ivar);
         if (iev==0) xmin[ivar]=xmax[ivar]=val;
         if (val < xmin[ivar]) xmin[ivar]=val;
         if (val > xmax[ivar]) xmax[ivar]=val;
      }
   }

   if (s+b < 0) {
      Log() << kWARNING << " One of the Decision Tree nodes has negative total number of signal or background events. "
            << "(Nsig="<<s<<" Nbkg="<<b<<" Probaby you use a Monte Carlo with negative weights. That should in principle "
            << "be fine as long as on average you end up with something positive. For this you have to make sure that the "
            << "minimul number of (unweighted) events demanded for a tree node (currently you use: nEventsMin="<<fMinSize
            << ", you can set this via the BDT option string when booking the classifier) is large enough to allow for "
            << "reasonable averaging!!!" << Endl
            << " If this does not help.. maybe you want to try the option: NoNegWeightsInTraining which ignores events "
            << "with negative weight in the training." << Endl;
      double nBkg=0.;
      for (UInt_t i=0; i<eventSample.size(); i++) {
         if (eventSample[i]->GetClass() != fSigClass) {
            nBkg += eventSample[i]->GetWeight();
            Log() << kDEBUG << "Event "<< i<< " has (original) weight: " <<  eventSample[i]->GetWeight()/eventSample[i]->GetBoostWeight() 
                  << " boostWeight: " << eventSample[i]->GetBoostWeight() << Endl;
         }
      }
      Log() << kDEBUG << " that gives in total: " << nBkg<<Endl;
   }

   node->SetNSigEvents(s);
   node->SetNBkgEvents(b);
   node->SetNSigEvents_unweighted(suw);
   node->SetNBkgEvents_unweighted(buw);
   node->SetPurity();
   if (node == this->GetRoot()) {
      node->SetNEvents(s+b);
      node->SetNEvents_unweighted(suw+buw);
   }
   for (UInt_t ivar=0; ivar<fNvars; ivar++) {
      node->SetSampleMin(ivar,xmin[ivar]);
      node->SetSampleMax(ivar,xmax[ivar]);
   }
   delete[] xmin;
   delete[] xmax;

   // I now demand the minimum number of events for both daughter nodes. Hence if the number
   // of events in the parent node is not at least two times as big, I don't even need to try
   // splitting

   //HHVTEST
   //   if (fNNodes < fNNodesMax && node->GetDepth() < fMaxDepth 
   if (eventSample.size() >= 2*fMinSize && fNNodes < fNNodesMax && node->GetDepth() < fMaxDepth 
       && ( ( s!=0 && b !=0 && !DoRegression()) || ( (s+b)!=0 && DoRegression()) ) ) {
      Double_t separationGain;
      if (fNCuts > 0){
         separationGain = this->TrainNodeFast(eventSample, node);
      } else {
         separationGain = this->TrainNodeFull(eventSample, node);
      }
      if (separationGain < std::numeric_limits<double>::epsilon()) { // we could not gain anything, e.g. all events are in one bin,
         /// if (separationGain < 0.00000001) { // we could not gain anything, e.g. all events are in one bin,
         // no cut can actually do anything to improve the node
         // hence, naturally, the current node is a leaf node
         if (DoRegression()) {
            node->SetSeparationIndex(fRegType->GetSeparationIndex(s+b,target,target2));
            node->SetResponse(target/(s+b));
            node->SetRMS(TMath::Sqrt(target2/(s+b) - target/(s+b)*target/(s+b)));
         }
         else {
            node->SetSeparationIndex(fSepType->GetSeparationIndex(s,b));
         }
         if (node->GetPurity() > fNodePurityLimit) node->SetNodeType(1);
         else node->SetNodeType(-1);
         if (node->GetDepth() > this->GetTotalTreeDepth()) this->SetTotalTreeDepth(node->GetDepth());

      } else {

         vector<TMVA::Event*> leftSample; leftSample.reserve(nevents);
         vector<TMVA::Event*> rightSample; rightSample.reserve(nevents);

         Double_t nRight=0, nLeft=0;

         for (UInt_t ie=0; ie< nevents ; ie++) {
            if (node->GoesRight(*eventSample[ie])) {
               rightSample.push_back(eventSample[ie]);
               nRight += eventSample[ie]->GetWeight();
            }
            else {
               leftSample.push_back(eventSample[ie]);
               nLeft += eventSample[ie]->GetWeight();
            }
         }

         // sanity check
         if (leftSample.size() == 0 || rightSample.size() == 0) {
            Log() << kFATAL << "<TrainNode> all events went to the same branch" << Endl
                  << "---                       Hence new node == old node ... check" << Endl
                  << "---                         left:" << leftSample.size()
                  << " right:" << rightSample.size() << Endl
                  << "--- this should never happen, please write a bug report to Helge.Voss@cern.ch"
                  << Endl;
         }

         // continue building daughter nodes for the left and the right eventsample
         TMVA::DecisionTreeNode *rightNode = new TMVA::DecisionTreeNode(node,'r');
         fNNodes++;
         rightNode->SetNEvents(nRight);
         rightNode->SetNEvents_unweighted(rightSample.size());

         TMVA::DecisionTreeNode *leftNode = new TMVA::DecisionTreeNode(node,'l');

         fNNodes++;
         leftNode->SetNEvents(nLeft);
         leftNode->SetNEvents_unweighted(leftSample.size());

         node->SetNodeType(0);
         node->SetLeft(leftNode);
         node->SetRight(rightNode);

         this->BuildTree(rightSample, rightNode);
         this->BuildTree(leftSample,  leftNode );
      }
   }
   else{ // it is a leaf node
      if (DoRegression()) {
         node->SetSeparationIndex(fRegType->GetSeparationIndex(s+b,target,target2));
         node->SetResponse(target/(s+b));
         node->SetRMS(TMath::Sqrt(target2/(s+b) - target/(s+b)*target/(s+b)));
      }
      else {
         node->SetSeparationIndex(fSepType->GetSeparationIndex(s,b));
         if   (node->GetPurity() > fNodePurityLimit) node->SetNodeType(1);
         else node->SetNodeType(-1);
         // loop through the event sample ending up in this node and check for events with negative weight
         // those "cannot" be boosted normally. Hence, if there is one of those
         // is misclassified, find randomly as many events with positive weights in this
         // node as needed to get the same absolute number of weight, and mark them as 
         // "not to be boosted" in order to make up for not boosting the negative weight event
         if (fPairNegWeightsInNode){
            Double_t sumOfNegWeights = 0;
            UInt_t    iClassID=99;  // the event class that misClassified in the current node
            for (UInt_t iev=0; iev<eventSample.size(); iev++) {
               if (eventSample[iev]->GetWeight() < 0) {
                  if (eventSample[iev]->GetClass() == fSigClass){
                     if (node->GetNodeType() != 1) { // classification is wrong
                        sumOfNegWeights+=eventSample[iev]->GetWeight();
                        iClassID=eventSample[iev]->GetClass();
                     }
                  } else {
                     if (node->GetNodeType() == 1) { // classification is wrong
                        sumOfNegWeights+=eventSample[iev]->GetWeight();
                        iClassID=eventSample[iev]->GetClass();
                     }
                  }
               }
            }
            if (iClassID == 99 && sumOfNegWeights < 0) Log() << kFATAL << " sorry.. something went wrong in treatment of neg. events" << Endl;
            // I need to find "misclassified" events whose positive weights add up to "sumOfNegWeights"
            while (sumOfNegWeights < 0 && 
                   ( ( TMath::Abs(sumOfNegWeights) < node->GetNBkgEvents() && iClassID != fSigClass) || 
                     ( TMath::Abs(sumOfNegWeights) < node->GetNSigEvents() && iClassID == fSigClass) ) ){
               UInt_t iev=fMyTrandom->Integer(eventSample.size());

               Log() << kWARNING
                     << "  so far...  I have still " << sumOfNegWeights 
                     << " now event " << iev << "("<<eventSample.size()<<") has " << eventSample[iev]->GetWeight()
                     << " class " << eventSample[iev]->GetClass() << "("<<iClassID<<")"
                     << " sig " << node->GetNSigEvents() << "("<<node->GetNSigEvents_unweighted()<< ")"
                     << " bkg " << node->GetNBkgEvents() << "("<<node->GetNBkgEvents_unweighted()<< ")"
                     << Endl;
               if (eventSample[iev]->GetWeight() > 0  && iClassID==eventSample[iev]->GetClass() ){               
                  sumOfNegWeights+=eventSample[iev]->GetWeight();
                  eventSample[iev]->SetDoNotBoost();

                  // Double_t dist=0, minDist=10E270;
                  // for (UInt_t ivar=0; ivar < GetNvar(); ivar++){
                  //    for (UInt_t jvar=0; jvar<GetNvar(); jvar++){
                  //       dist += (negEvents[nev]->GetValue(ivar)-fEventSample[iev]->GetValue(ivar))*
                  //          //                           (*invCov)[ivar][jvar]*
                  //          (negEvents[nev]->GetValue(jvar)-fEventSample[iev]->GetValue(jvar));
                  //    }
                  // }
                  // Log() << kWARNING << "pair with event in dist^2="<<dist << "  dist="<<TMath::Sqrt(dist) << Endl;


               }
            }
         }
      }
      
      
      if (node->GetDepth() > this->GetTotalTreeDepth()) this->SetTotalTreeDepth(node->GetDepth());
   }
   
   //   if (IsRootNode) this->CleanTree();
   return fNNodes;
}

//_______________________________________________________________________
void TMVA::DecisionTree::FillTree( vector<TMVA::Event*> & eventSample )
  
{
   // fill the existing the decision tree structure by filling event
   // in from the top node and see where they happen to end up
   for (UInt_t i=0; i<eventSample.size(); i++) {
      this->FillEvent(*(eventSample[i]),NULL);
   }
}

//_______________________________________________________________________
void TMVA::DecisionTree::FillEvent( TMVA::Event & event,  
                                    TMVA::DecisionTreeNode *node )
{
   // fill the existing the decision tree structure by filling event
   // in from the top node and see where they happen to end up
  
   if (node == NULL) { // that's the start, take the Root node
      node = this->GetRoot();
   }
  
   node->IncrementNEvents( event.GetWeight() );
   node->IncrementNEvents_unweighted( );
  
   if (event.GetClass() == fSigClass) {
      node->IncrementNSigEvents( event.GetWeight() );
      node->IncrementNSigEvents_unweighted( );
   } 
   else {
      node->IncrementNBkgEvents( event.GetWeight() );
      node->IncrementNBkgEvents_unweighted( );
   }
   node->SetSeparationIndex(fSepType->GetSeparationIndex(node->GetNSigEvents(),
                                                         node->GetNBkgEvents()));
  
   if (node->GetNodeType() == 0) { //intermediate node --> go down
      if (node->GoesRight(event))
         this->FillEvent(event,dynamic_cast<TMVA::DecisionTreeNode*>(node->GetRight())) ;
      else
         this->FillEvent(event,dynamic_cast<TMVA::DecisionTreeNode*>(node->GetLeft())) ;
   }
  
  
}

//_______________________________________________________________________
void TMVA::DecisionTree::ClearTree()
{
   // clear the tree nodes (their S/N, Nevents etc), just keep the structure of the tree
  
   if (this->GetRoot()!=NULL) this->GetRoot()->ClearNodeAndAllDaughters();
  
}

//_______________________________________________________________________
UInt_t TMVA::DecisionTree::CleanTree( DecisionTreeNode *node )
{
   // remove those last splits that result in two leaf nodes that
   // are both of the type (i.e. both signal or both background)
   // this of course is only a reasonable thing to do when you use
   // "YesOrNo" leafs, while it might loose s.th. if you use the
   // purity information in the nodes.
   // --> hence I don't call it automatically in the tree building

   if (node==NULL) {
      node = this->GetRoot();
   }

   DecisionTreeNode *l = node->GetLeft();
   DecisionTreeNode *r = node->GetRight();

   if (node->GetNodeType() == 0) {
      this->CleanTree(l);
      this->CleanTree(r);
      if (l->GetNodeType() * r->GetNodeType() > 0) {

         this->PruneNode(node);
      }
   }
   // update the number of nodes after the cleaning
   return this->CountNodes();
   
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::PruneTree( vector<Event*>* validationSample )
{
   // prune (get rid of internal nodes) the Decision tree to avoid overtraining
   // serveral different pruning methods can be applied as selected by the 
   // variable "fPruneMethod". 
  
   //   std::ofstream logfile("dt_pruning.log");
  
   IPruneTool* tool(NULL);
   PruningInfo* info(NULL);

   if( fPruneMethod == kNoPruning ) return 0.0;

   if      (fPruneMethod == kExpectedErrorPruning) 
      //      tool = new ExpectedErrorPruneTool(logfile);
      tool = new ExpectedErrorPruneTool();
   else if (fPruneMethod == kCostComplexityPruning) 
      {
         tool = new CostComplexityPruneTool();
      }
   else {
      Log() << kFATAL << "Selected pruning method not yet implemented "
            << Endl;
   }
   if(!tool) return 0.0;

   tool->SetPruneStrength(GetPruneStrength());
   if(tool->IsAutomatic()) {
      if(validationSample == NULL) 
         Log() << kFATAL << "Cannot automate the pruning algorithm without an "
               << "independent validation sample!" << Endl;
      if(validationSample->size() == 0) 
         Log() << kFATAL << "Cannot automate the pruning algorithm with "
               << "independent validation sample of ZERO events!" << Endl;
   }

   info = tool->CalculatePruningInfo(this,validationSample);
   if(!info) {
      delete tool;
      Log() << kFATAL << "Error pruning tree! Check prune.log for more information." 
            << Endl;
   }
   Double_t pruneStrength = info->PruneStrength;

   //   Log() << kDEBUG << "Optimal prune strength (alpha): " << pruneStrength
   //           << " has quality index " << info->QualityIndex << Endl;
   

   for (UInt_t i = 0; i < info->PruneSequence.size(); ++i) {
      
      PruneNode(info->PruneSequence[i]);
   }
   // update the number of nodes after the pruning
   this->CountNodes();

   delete tool;
   delete info;

   return pruneStrength;
};


//_______________________________________________________________________
void TMVA::DecisionTree::ApplyValidationSample( const EventList* validationSample ) const
{
   // run the validation sample through the (pruned) tree and fill in the nodes
   // the variables NSValidation and NBValidadtion (i.e. how many of the Signal
   // and Background events from the validation sample. This is then later used
   // when asking for the "tree quality" .. 
   GetRoot()->ResetValidationData();
   for (UInt_t ievt=0; ievt < validationSample->size(); ievt++) {
      CheckEventWithPrunedTree(*(*validationSample)[ievt]);
   }
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::TestPrunedTreeQuality( const DecisionTreeNode* n, Int_t mode ) const
{
   // return the misclassification rate of a pruned tree
   // a "pruned tree" may have set the variable "IsTerminal" to "arbitrary" at
   // any node, hence this tree quality testing will stop there, hence test
   // the pruned tree (while the full tree is still in place for normal/later use)
   
   if (n == NULL) { // default, start at the tree top, then descend recursively
      n = this->GetRoot();
      if (n == NULL) {
         Log() << kFATAL << "TestPrunedTreeQuality: started with undefined ROOT node" <<Endl;
         return 0;
      }
   } 

   if( n->GetLeft() != NULL && n->GetRight() != NULL && !n->IsTerminal() ) {
      return (TestPrunedTreeQuality( n->GetLeft(), mode ) +
              TestPrunedTreeQuality( n->GetRight(), mode ));
   }
   else { // terminal leaf (in a pruned subtree of T_max at least)
      if (DoRegression()) {
         Double_t sumw = n->GetNSValidation() + n->GetNBValidation();
         return n->GetSumTarget2() - 2*n->GetSumTarget()*n->GetResponse() + sumw*n->GetResponse()*n->GetResponse();
      } 
      else {
         if (mode == 0) {
            if (n->GetPurity() > this->GetNodePurityLimit()) // this is a signal leaf, according to the training
               return n->GetNBValidation();
            else
               return n->GetNSValidation();
         }
         else if ( mode == 1 ) {
            // calculate the weighted error using the pruning validation sample
            return (n->GetPurity() * n->GetNBValidation() + (1.0 - n->GetPurity()) * n->GetNSValidation());
         }
         else {
            throw std::string("Unknown ValidationQualityMode");
         }
      }
   }
}

//_______________________________________________________________________
void TMVA::DecisionTree::CheckEventWithPrunedTree( const Event& e ) const
{
   // pass a single validation event throught a pruned decision tree
   // on the way down the tree, fill in all the "intermediate" information
   // that would normally be there from training.

   DecisionTreeNode* current =  this->GetRoot();
   if (current == NULL) {
      Log() << kFATAL << "CheckEventWithPrunedTree: started with undefined ROOT node" <<Endl;
   }

   while(current != NULL) {
      if(e.GetClass() == fSigClass)
         current->SetNSValidation(current->GetNSValidation() + e.GetWeight());
      else
         current->SetNBValidation(current->GetNBValidation() + e.GetWeight());

      if (e.GetNTargets() > 0) {
         current->AddToSumTarget(e.GetWeight()*e.GetTarget(0));
         current->AddToSumTarget2(e.GetWeight()*e.GetTarget(0)*e.GetTarget(0));
      }

      if (current->GetRight() == NULL || current->GetLeft() == NULL) {
         current = NULL;
      }
      else {
         if (current->GoesRight(e))
            current = (TMVA::DecisionTreeNode*)current->GetRight();
         else
            current = (TMVA::DecisionTreeNode*)current->GetLeft();
      }
   }
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetSumWeights( const EventList* validationSample ) const
{
   // calculate the normalization factor for a pruning validation sample
   Double_t sumWeights = 0.0;
   for( EventList::const_iterator it = validationSample->begin();
        it != validationSample->end(); ++it ) {
      sumWeights += (*it)->GetWeight();
   }
   return sumWeights;
}



//_______________________________________________________________________
UInt_t TMVA::DecisionTree::CountLeafNodes( TMVA::Node *n )
{
   // return the number of terminal nodes in the sub-tree below Node n
  
   if (n == NULL) { // default, start at the tree top, then descend recursively
      n =  this->GetRoot();
      if (n == NULL) {
         Log() << kFATAL << "CountLeafNodes: started with undefined ROOT node" <<Endl;
         return 0;
      }
   } 
  
   UInt_t countLeafs=0;
  
   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {
      countLeafs += 1;
   } 
   else { 
      if (this->GetLeftDaughter(n) != NULL) {
         countLeafs += this->CountLeafNodes( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         countLeafs += this->CountLeafNodes( this->GetRightDaughter(n) );
      }
   }
   return countLeafs;
}

//_______________________________________________________________________
void TMVA::DecisionTree::DescendTree( Node* n )
{
   // descend a tree to find all its leaf nodes
  
   if (n == NULL) { // default, start at the tree top, then descend recursively
      n =  this->GetRoot();
      if (n == NULL) {
         Log() << kFATAL << "DescendTree: started with undefined ROOT node" <<Endl;
         return ;
      }
   } 
  
   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {
      // do nothing
   } 
   else if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) != NULL) ) {
      Log() << kFATAL << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   }  
   else if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) == NULL) ) {
      Log() << kFATAL << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   } 
   else { 
      if (this->GetLeftDaughter(n) != NULL) {
         this->DescendTree( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         this->DescendTree( this->GetRightDaughter(n) );
      }
   }
}

//_______________________________________________________________________
void TMVA::DecisionTree::PruneNode( DecisionTreeNode* node )
{
   // prune away the subtree below the node 
   DecisionTreeNode *l = node->GetLeft();
   DecisionTreeNode *r = node->GetRight();

   node->SetRight(NULL);
   node->SetLeft(NULL);
   node->SetSelector(-1);
   node->SetSeparationGain(-1);
   if (node->GetPurity() > fNodePurityLimit) node->SetNodeType(1);
   else node->SetNodeType(-1);
   this->DeleteNode(l);
   this->DeleteNode(r);
   // update the stored number of nodes in the Tree
   this->CountNodes();
  
}

//_______________________________________________________________________
void TMVA::DecisionTree::PruneNodeInPlace( DecisionTreeNode* node ) {
   // prune a node temporaily (without actually deleting its decendants
   // which allows testing the pruned tree quality for many different
   // pruning stages without "touching" the tree.

   if(node == NULL) return;
   node->SetNTerminal(1);
   node->SetSubTreeR( node->GetNodeR() );
   node->SetAlpha( std::numeric_limits<double>::infinity( ) );
   node->SetAlphaMinSubtree( std::numeric_limits<double>::infinity( ) );
   node->SetTerminal(kTRUE); // set the node to be terminal without deleting its descendants FIXME not needed
}

//_______________________________________________________________________
TMVA::Node* TMVA::DecisionTree::GetNode( ULong_t sequence, UInt_t depth )
{
   // retrieve node from the tree. Its position (up to a maximal tree depth of 64)
   // is coded as a sequence of left-right moves starting from the root, coded as
   // 0-1 bit patterns stored in the "long-integer"  (i.e. 0:left ; 1:right
  
   Node* current = this->GetRoot();
  
   for (UInt_t i =0;  i < depth; i++) {
      ULong_t tmp = 1 << i;
      if ( tmp & sequence) current = this->GetRightDaughter(current);
      else current = this->GetLeftDaughter(current);
   }
  
   return current;
}


//_______________________________________________________________________
void TMVA::DecisionTree::GetRandomisedVariables(Bool_t *useVariable, UInt_t *mapVariable, UInt_t &useNvars){
  //
   for (UInt_t ivar=0; ivar<fNvars; ivar++) useVariable[ivar]=kFALSE;
   if (fUseNvars==0) { // no number specified ... choose s.th. which hopefully works well 
      // watch out, should never happen as it is initialised automatically in MethodBDT already!!!
      fUseNvars        =  UInt_t(TMath::Sqrt(fNvars)+0.6);
   }
   if (fUsePoissonNvars) useNvars=TMath::Min(fNvars,TMath::Max(UInt_t(1),(UInt_t) fMyTrandom->Poisson(fUseNvars)));
   else useNvars = fUseNvars;

   UInt_t nSelectedVars = 0;
   while (nSelectedVars < useNvars) {
      Double_t bla = fMyTrandom->Rndm()*fNvars;
      useVariable[Int_t (bla)] = kTRUE;
      nSelectedVars = 0;
      for (UInt_t ivar=0; ivar < fNvars; ivar++) {
         if (useVariable[ivar] == kTRUE) { 
            mapVariable[nSelectedVars] = ivar;
            nSelectedVars++;
         }
      }
   }
   if (nSelectedVars != useNvars) { std::cout << "Bug in TrainNode - GetRandisedVariables()... sorry" << std::endl; std::exit(1);}
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::TrainNodeFast( const vector<TMVA::Event*> & eventSample,
                                           TMVA::DecisionTreeNode *node )
{
   // Decide how to split a node using one of the variables that gives
   // the best separation of signal/background. In order to do this, for each 
   // variable a scan of the different cut values in a grid (grid = fNCuts) is 
   // performed and the resulting separation gains are compared.
   // in addition to the individual variables, one can also ask for a fisher
   // discriminant being built out of (some) of the variables and used as a
   // possible multivariate split.

   Double_t  separationGainTotal = -1, sepTmp;
   Double_t *separationGain    = new Double_t[fNvars+1];
   for (UInt_t ivar=0; ivar <= fNvars; ivar++) {
      separationGain[ivar]=-1;
   }
   Double_t cutValue=-999;
   Int_t mxVar= -1;
   Int_t cutIndex=-1;
   Bool_t cutType=kTRUE;
   Double_t  nTotS, nTotB;
   Int_t     nTotS_unWeighted, nTotB_unWeighted; 
   UInt_t nevents = eventSample.size();


   // the +1 comes from the fact that I treat later on the Fisher output as an 
   // additional possible variable.
   Bool_t *useVariable = new Bool_t[fNvars+1];   // for performance reasons instead of vector<Bool_t> useVariable(fNvars);
   UInt_t *mapVariable = new UInt_t[fNvars+1];    // map the subset of variables used in randomised trees to the original variable number (used in the Event() ) 

   std::vector<Double_t> fisherCoeff;
 
   if (fRandomisedTree) { // choose for each node splitting a random subset of variables to choose from
      UInt_t tmp=fUseNvars;
      GetRandomisedVariables(useVariable,mapVariable,tmp);
   } 
   else {
      for (UInt_t ivar=0; ivar < fNvars; ivar++) {
         useVariable[ivar] = kTRUE;
         mapVariable[ivar] = ivar;
      }
   }
   useVariable[fNvars] = kFALSE; //by default fisher is not used..

   if (fUseFisherCuts) {
      useVariable[fNvars] = kTRUE; // that's were I store the "fisher MVA"

      //use for the Fisher discriminant ONLY those variables that show
      //some reasonable linear correlation in either Signal or Background
      Bool_t *useVarInFisher = new Bool_t[fNvars];   // for performance reasons instead of vector<Bool_t> useVariable(fNvars);
      UInt_t *mapVarInFisher = new UInt_t[fNvars];   // map the subset of variables used in randomised trees to the original variable number (used in the Event() ) 
      for (UInt_t ivar=0; ivar < fNvars; ivar++) {
         useVarInFisher[ivar] = kFALSE;
         mapVarInFisher[ivar] = ivar;
      }
      
      std::vector<TMatrixDSym*>* covMatrices;
      covMatrices = gTools().CalcCovarianceMatrices( eventSample, 2 ); // currently for 2 classes only
      TMatrixD *ss = new TMatrixD(*(covMatrices->at(0)));
      TMatrixD *bb = new TMatrixD(*(covMatrices->at(1)));
      const TMatrixD *s = gTools().GetCorrelationMatrix(ss);
      const TMatrixD *b = gTools().GetCorrelationMatrix(bb);
      
      for (UInt_t ivar=0; ivar < fNvars; ivar++) {
         for (UInt_t jvar=ivar+1; jvar < fNvars; jvar++) {
            if (  ( TMath::Abs( (*s)(ivar, jvar)) > fMinLinCorrForFisher) ||
                  ( TMath::Abs( (*b)(ivar, jvar)) > fMinLinCorrForFisher) ){
               useVarInFisher[ivar] = kTRUE;
               useVarInFisher[jvar] = kTRUE;
            }
         }
      }
      // now as you know which variables you want to use, count and map them:
      // such that you can use an array/matrix filled only with THOSE variables
      // that you used
      UInt_t nFisherVars = 0;
      for (UInt_t ivar=0; ivar < fNvars; ivar++) {
         //now .. pick those variables that are used in the FIsher and are also
         //  part of the "allowed" variables in case of Randomized Trees)
         if (useVarInFisher[ivar] && useVariable[ivar]) {
            mapVarInFisher[nFisherVars++]=ivar;
            // now exclud the the variables used in the Fisher cuts, and don't 
            // use them anymore in the individual variable scan
            if (fUseExclusiveVars) useVariable[ivar] = kFALSE;
         }
      }
      
      
      fisherCoeff = this->GetFisherCoefficients(eventSample, nFisherVars, mapVarInFisher);
      delete [] useVarInFisher;
      delete [] mapVarInFisher;
   }


   const UInt_t nBins = fNCuts+1;
   UInt_t cNvars = fNvars;
   if (fUseFisherCuts) cNvars++;  // use the Fisher output simple as additional variable

   Double_t** nSelS = new Double_t* [cNvars];
   Double_t** nSelB = new Double_t* [cNvars];
   Double_t** nSelS_unWeighted = new Double_t* [cNvars];
   Double_t** nSelB_unWeighted = new Double_t* [cNvars];
   Double_t** target = new Double_t* [cNvars];
   Double_t** target2 = new Double_t* [cNvars];
   Double_t** cutValues = new Double_t* [cNvars];

   for (UInt_t i=0; i<cNvars; i++) {
      nSelS[i] = new Double_t [nBins];
      nSelB[i] = new Double_t [nBins];
      nSelS_unWeighted[i] = new Double_t [nBins];
      nSelB_unWeighted[i] = new Double_t [nBins];
      target[i] = new Double_t [nBins];
      target2[i] = new Double_t [nBins];
      cutValues[i] = new Double_t [nBins];
   }

   Double_t *xmin = new Double_t[cNvars]; 
   Double_t *xmax = new Double_t[cNvars];

   for (UInt_t ivar=0; ivar < cNvars; ivar++) {
      if (ivar < fNvars){
         xmin[ivar]=node->GetSampleMin(ivar);
         xmax[ivar]=node->GetSampleMax(ivar);
      } else { // the fisher variable
         xmin[ivar]=999;
         xmax[ivar]=-999;
         // too bad, for the moment I don't know how to do this without looping
         // once to get the "min max" and then AGAIN to fill the histogram
         for (UInt_t iev=0; iev<nevents; iev++) {
            // returns the Fisher value (no fixed range)
            Double_t result = fisherCoeff[fNvars]; // the fisher constant offset
            for (UInt_t jvar=0; jvar<fNvars; jvar++)
               result += fisherCoeff[jvar]*(eventSample[iev])->GetValue(jvar);
            if (result > xmax[ivar]) xmax[ivar]=result;
            if (result < xmin[ivar]) xmin[ivar]=result;
         }
      }
      for (UInt_t ibin=0; ibin<nBins; ibin++) {
         nSelS[ivar][ibin]=0;
         nSelB[ivar][ibin]=0;
         nSelS_unWeighted[ivar][ibin]=0;
         nSelB_unWeighted[ivar][ibin]=0;
         target[ivar][ibin]=0;
         target2[ivar][ibin]=0;
         cutValues[ivar][ibin]=0;
      }
   }

   // fill the cut values for the scan:
   for (UInt_t ivar=0; ivar < cNvars; ivar++) {

      if ( useVariable[ivar] ) {
         
         //set the grid for the cut scan on the variables like this:
         // 
         //  |       |        |         |         |   ...      |        |  
         // xmin                                                       xmax
         //
         // cut      0        1         2         3   ...     fNCuts-1 (counting from zero)
         // bin  0       1         2         3       .....      nBins-1=fNCuts (counting from zero)
         // --> nBins = fNCuts+1
         // (NOTE, the cuts at xmin or xmax would just give the whole sample and
         //  hence can be safely omitted
         
         Double_t istepSize =( xmax[ivar] - xmin[ivar] ) / Double_t(nBins);
         // std::cout << "min="<<xmin[ivar]  
         //           << " max="<<xmax[ivar] 
         //           << " widht=" << istepSize 
         //           << std::endl;
         for (Int_t icut=0; icut<fNCuts; icut++) {
            cutValues[ivar][icut]=xmin[ivar]+(Double_t(icut+1))*istepSize;
            //            std::cout << " cutValues["<<ivar<<"]["<<icut<<"]=" <<  cutValues[ivar][icut] << std::endl;
         }
      }
   }
  
   nTotS=0; nTotB=0;
   nTotS_unWeighted=0; nTotB_unWeighted=0;   
   for (UInt_t iev=0; iev<nevents; iev++) {

      Double_t eventWeight =  eventSample[iev]->GetWeight(); 
      if (eventSample[iev]->GetClass() == fSigClass) {
         nTotS+=eventWeight;
         nTotS_unWeighted++;
      }
      else {
         nTotB+=eventWeight;
         nTotB_unWeighted++;
      }
      
      Int_t iBin=-1;
      for (UInt_t ivar=0; ivar < cNvars; ivar++) {
         // now scan trough the cuts for each varable and find which one gives
         // the best separationGain at the current stage.
         if ( useVariable[ivar] ) {
            Double_t eventData;
            if (ivar < fNvars) eventData = eventSample[iev]->GetValue(ivar); 
            else { // the fisher variable
               eventData = fisherCoeff[fNvars];
               for (UInt_t jvar=0; jvar<fNvars; jvar++)
                  eventData += fisherCoeff[jvar]*(eventSample[iev])->GetValue(jvar);
               
            }
            // "maximum" is nbins-1 (the "-1" because we start counting from 0 !!
            iBin = TMath::Min(Int_t(nBins-1),TMath::Max(0,int (nBins*(eventData-xmin[ivar])/(xmax[ivar]-xmin[ivar]) ) ));
            if (eventSample[iev]->GetClass() == fSigClass) {
               nSelS[ivar][iBin]+=eventWeight;
               nSelS_unWeighted[ivar][iBin]++;
            } 
            else {
               nSelB[ivar][iBin]+=eventWeight;
               nSelB_unWeighted[ivar][iBin]++;
            }
            if (DoRegression()) {
               target[ivar][iBin] +=eventWeight*eventSample[iev]->GetTarget(0);
               target2[ivar][iBin]+=eventWeight*eventSample[iev]->GetTarget(0)*eventSample[iev]->GetTarget(0);
            }
         }
      }
   }   
   // now turn the "histogram" into a cummulative distribution
   for (UInt_t ivar=0; ivar < cNvars; ivar++) {
      if (useVariable[ivar]) {
         for (UInt_t ibin=1; ibin < nBins; ibin++) {
            nSelS[ivar][ibin]+=nSelS[ivar][ibin-1];
            nSelS_unWeighted[ivar][ibin]+=nSelS_unWeighted[ivar][ibin-1];
            nSelB[ivar][ibin]+=nSelB[ivar][ibin-1];
            nSelB_unWeighted[ivar][ibin]+=nSelB_unWeighted[ivar][ibin-1];
            if (DoRegression()) {
               target[ivar][ibin] +=target[ivar][ibin-1] ;
               target2[ivar][ibin]+=target2[ivar][ibin-1];
            }
         }
         if (nSelS_unWeighted[ivar][nBins-1] +nSelB_unWeighted[ivar][nBins-1] != eventSample.size()) {
            Log() << kFATAL << "Helge, you have a bug ....nSelS_unw..+nSelB_unw..= "
                  << nSelS_unWeighted[ivar][nBins-1] +nSelB_unWeighted[ivar][nBins-1] 
                  << " while eventsample size = " << eventSample.size()
                  << Endl;
         }
         double lastBins=nSelS[ivar][nBins-1] +nSelB[ivar][nBins-1];
         double totalSum=nTotS+nTotB;
         if (TMath::Abs(lastBins-totalSum)/totalSum>0.01) {
            Log() << kFATAL << "Helge, you have another bug ....nSelS+nSelB= "
                  << lastBins
                  << " while total number of events = " << totalSum
                  << Endl;
         }
      }
   }
   // now select the optimal cuts for each varable and find which one gives
   // the best separationGain at the current stage
   for (UInt_t ivar=0; ivar < cNvars; ivar++) {
      if (useVariable[ivar]) {
         for (UInt_t iBin=0; iBin<nBins-1; iBin++) { // the last bin contains "all events" -->skip
            // the separationGain is defined as the various indices (Gini, CorssEntropy, e.t.c)
            // calculated by the "SamplePurities" fom the branches that would go to the
            // left or the right from this node if "these" cuts were used in the Node:
            // hereby: nSelS and nSelB would go to the right branch
            //        (nTotS - nSelS) + (nTotB - nSelB)  would go to the left branch;

            // only allow splits where both daughter nodes match the specified miniumum number
            // for this use the "unweighted" events, as you are interested in statistically 
            // significant splits, which is determined by the actual number of entries
            // for a node, rather than the sum of event weights.

            Double_t sl = nSelS_unWeighted[ivar][iBin];
            Double_t bl = nSelB_unWeighted[ivar][iBin];
            Double_t s  = nTotS_unWeighted;
            Double_t b  = nTotB_unWeighted;
            // HHVTEST ... see if that's the reason why neg.even weight boosting still behave different...
            // Double_t sl = nSelS[ivar][iBin];
            // Double_t bl = nSelB[ivar][iBin];
            // Double_t s  = nTotS;
            // Double_t b  = nTotB;
            Double_t sr = s-sl;
            Double_t br = b-bl;
            if ( (sl+bl)>=fMinSize && (sr+br)>=fMinSize ) {

               if (DoRegression()) {
                  sepTmp = fRegType->GetSeparationGain(nSelS[ivar][iBin]+nSelB[ivar][iBin], 
                                                       target[ivar][iBin],target2[ivar][iBin],
                                                       nTotS+nTotB,
                                                       target[ivar][nBins-1],target2[ivar][nBins-1]);
               } else {
                  sepTmp = fSepType->GetSeparationGain(nSelS[ivar][iBin], nSelB[ivar][iBin], nTotS, nTotB);
               }
               if (separationGain[ivar] < sepTmp) {
                  separationGain[ivar] = sepTmp;  // used for variable importance calculation
                  if (separationGainTotal < sepTmp) {
                     separationGainTotal = sepTmp;
                     mxVar = ivar;
                     cutIndex = iBin;
                     if (cutIndex >= fNCuts) Log()<<kFATAL<<"ibin for cut " << iBin << Endl; 
                  }
               }
            }
         }
      }
   }
   
   if (DoRegression()) {
      node->SetSeparationIndex(fRegType->GetSeparationIndex(nTotS+nTotB,target[0][nBins-1],target2[0][nBins-1]));
      node->SetResponse(target[0][nBins-1]/(nTotS+nTotB));
      node->SetRMS(TMath::Sqrt(target2[0][nBins-1]/(nTotS+nTotB) - target[0][nBins-1]/(nTotS+nTotB)*target[0][nBins-1]/(nTotS+nTotB)));
   }
   else {
      node->SetSeparationIndex(fSepType->GetSeparationIndex(nTotS,nTotB));
   }
   if (mxVar >= 0) { 
      if (nSelS[mxVar][cutIndex]/nTotS > nSelB[mxVar][cutIndex]/nTotB) cutType=kTRUE;
      else cutType=kFALSE;      
      cutValue = cutValues[mxVar][cutIndex];
    
      node->SetSelector((UInt_t)mxVar);
      node->SetCutValue(cutValue);
      node->SetCutType(cutType);
      node->SetSeparationGain(separationGainTotal);
      if (mxVar < (Int_t) fNvars){ // the fisher cut is actually not used in this node, hence don't need to store fisher components
         node->SetNFisherCoeff(0);
         fVariableImportance[mxVar] += separationGainTotal*separationGainTotal * (nTotS+nTotB) * (nTotS+nTotB) ;
         //for (UInt_t ivar=0; ivar<fNvars; ivar++) fVariableImportance[ivar] += separationGain[ivar]*separationGain[ivar] * (nTotS+nTotB) * (nTotS+nTotB) ;
      }else{
         // allocate Fisher coefficients (use fNvars, and set the non-used ones to zero. Might
         // be even less storage space on average than storing also the mapping used otherwise
         // can always be changed relatively easy
         node->SetNFisherCoeff(fNvars+1);     
         for (UInt_t ivar=0; ivar<=fNvars; ivar++) {
            node->SetFisherCoeff(ivar,fisherCoeff[ivar]);
            // take 'fisher coeff. weighted estimate as variable importance, "Don't fill the offset coefficient though :) 
            if (ivar<fNvars){
               fVariableImportance[ivar] += fisherCoeff[ivar]*fisherCoeff[ivar]*separationGainTotal*separationGainTotal * (nTotS+nTotB) * (nTotS+nTotB) ;
            }
         }
      } 
   }
   else {
      separationGainTotal = 0;
   }
  

   for (UInt_t i=0; i<cNvars; i++) {
      delete [] nSelS[i];
      delete [] nSelB[i];
      delete [] nSelS_unWeighted[i];
      delete [] nSelB_unWeighted[i];
      delete [] target[i];
      delete [] target2[i];
      delete [] cutValues[i];
   }
   delete [] nSelS;
   delete [] nSelB;
   delete [] nSelS_unWeighted;
   delete [] nSelB_unWeighted;
   delete [] target;
   delete [] target2;
   delete [] cutValues;

   delete [] xmin;
   delete [] xmax;

   delete [] useVariable;
   delete [] mapVariable;

   delete [] separationGain;

   return separationGainTotal;

}



//_______________________________________________________________________
std::vector<Double_t>  TMVA::DecisionTree::GetFisherCoefficients(const EventList &eventSample, UInt_t nFisherVars, UInt_t *mapVarInFisher){ 
  // calculate the fisher coefficients for the event sample and the variables used

   std::vector<Double_t> fisherCoeff(fNvars+1);

   // initializaton of global matrices and vectors
   // average value of each variables for S, B, S+B
   TMatrixD* meanMatx = new TMatrixD( nFisherVars, 3 );
   
   // the covariance 'within class' and 'between class' matrices
   TMatrixD* betw = new TMatrixD( nFisherVars, nFisherVars );
   TMatrixD* with = new TMatrixD( nFisherVars, nFisherVars );
   TMatrixD* cov  = new TMatrixD( nFisherVars, nFisherVars );

   //
   // compute mean values of variables in each sample, and the overall means
   //

   // initialize internal sum-of-weights variables
   Double_t sumOfWeightsS = 0;
   Double_t sumOfWeightsB = 0;
   
   
   // init vectors
   Double_t* sumS = new Double_t[nFisherVars];
   Double_t* sumB = new Double_t[nFisherVars];
   for (UInt_t ivar=0; ivar<nFisherVars; ivar++) { sumS[ivar] = sumB[ivar] = 0; }   

   UInt_t nevents = eventSample.size();   
   // compute sample means
   for (UInt_t ievt=0; ievt<nevents; ievt++) {
      
      // read the Training Event into "event"
      const Event * ev = eventSample[ievt];

      // sum of weights
      Double_t weight = ev->GetWeight();
      if (ev->GetClass() == fSigClass) sumOfWeightsS += weight;
      else                             sumOfWeightsB += weight;

      Double_t* sum = ev->GetClass() == fSigClass ? sumS : sumB;
      for (UInt_t ivar=0; ivar<nFisherVars; ivar++) sum[ivar] += ev->GetValue( mapVarInFisher[ivar] )*weight;
   }

   for (UInt_t ivar=0; ivar<nFisherVars; ivar++) {   
      (*meanMatx)( ivar, 2 ) = sumS[ivar];
      (*meanMatx)( ivar, 0 ) = sumS[ivar]/sumOfWeightsS;
      
      (*meanMatx)( ivar, 2 ) += sumB[ivar];
      (*meanMatx)( ivar, 1 ) = sumB[ivar]/sumOfWeightsB;
      
      // signal + background
      (*meanMatx)( ivar, 2 ) /= (sumOfWeightsS + sumOfWeightsB);
   }  
   delete [] sumS;
   delete [] sumB;

   // the matrix of covariance 'within class' reflects the dispersion of the
   // events relative to the center of gravity of their own class  

   // assert required

   assert( sumOfWeightsS > 0 && sumOfWeightsB > 0 );

   // product matrices (x-<x>)(y-<y>) where x;y are variables

   const Int_t nFisherVars2 = nFisherVars*nFisherVars;
   Double_t *sum2Sig  = new Double_t[nFisherVars2];
   Double_t *sum2Bgd  = new Double_t[nFisherVars2];
   Double_t *xval    = new Double_t[nFisherVars2];
   memset(sum2Sig,0,nFisherVars2*sizeof(Double_t));
   memset(sum2Bgd,0,nFisherVars2*sizeof(Double_t));
   
   // 'within class' covariance
   for (UInt_t ievt=0; ievt<nevents; ievt++) {

      // read the Training Event into "event"
      const Event* ev = eventSample[ievt];

      Double_t weight = ev->GetWeight(); // may ignore events with negative weights

      for (UInt_t x=0; x<nFisherVars; x++) xval[x] = ev->GetValue( mapVarInFisher[x] );
      Int_t k=0;
      for (UInt_t x=0; x<nFisherVars; x++) {
         for (UInt_t y=0; y<nFisherVars; y++) {            
            Double_t v = ( (xval[x] - (*meanMatx)(x, 0))*(xval[y] - (*meanMatx)(y, 0)) )*weight;
            if ( ev->GetClass() == fSigClass ) sum2Sig[k] += v;
            else                               sum2Bgd[k] += v;
            k++;
         }
      }
   }
   Int_t k=0;
   for (UInt_t x=0; x<nFisherVars; x++) {
      for (UInt_t y=0; y<nFisherVars; y++) {
         (*with)(x, y) = (sum2Sig[k] + sum2Bgd[k])/(sumOfWeightsS + sumOfWeightsB);
         k++;
      }
   }

   delete [] sum2Sig;
   delete [] sum2Bgd;
   delete [] xval;


   // the matrix of covariance 'between class' reflects the dispersion of the
   // events of a class relative to the global center of gravity of all the class
   // hence the separation between classes


   Double_t prodSig, prodBgd;

   for (UInt_t x=0; x<nFisherVars; x++) {
      for (UInt_t y=0; y<nFisherVars; y++) {

         prodSig = ( ((*meanMatx)(x, 0) - (*meanMatx)(x, 2))*
                     ((*meanMatx)(y, 0) - (*meanMatx)(y, 2)) );
         prodBgd = ( ((*meanMatx)(x, 1) - (*meanMatx)(x, 2))*
                     ((*meanMatx)(y, 1) - (*meanMatx)(y, 2)) );

         (*betw)(x, y) = (sumOfWeightsS*prodSig + sumOfWeightsB*prodBgd) / (sumOfWeightsS + sumOfWeightsB);
      }
   }



   // compute full covariance matrix from sum of within and between matrices
   for (UInt_t x=0; x<nFisherVars; x++) 
      for (UInt_t y=0; y<nFisherVars; y++) 
         (*cov)(x, y) = (*with)(x, y) + (*betw)(x, y);
        
   // Fisher = Sum { [coeff]*[variables] }
   //
   // let Xs be the array of the mean values of variables for signal evts
   // let Xb be the array of the mean values of variables for backgd evts
   // let InvWith be the inverse matrix of the 'within class' correlation matrix
   //
   // then the array of Fisher coefficients is 
   // [coeff] =TMath::Sqrt(fNsig*fNbgd)/fNevt*transpose{Xs-Xb}*InvWith
   TMatrixD* theMat = with; // Fishers original
   //   TMatrixD* theMat = cov; // Mahalanobis
      
   TMatrixD invCov( *theMat );
   if ( TMath::Abs(invCov.Determinant()) < 10E-24 ) {
      Log() << kWARNING << "FisherCoeff matrix is almost singular with deterninant="
              << TMath::Abs(invCov.Determinant()) 
              << " did you use the variables that are linear combinations or highly correlated?" 
              << Endl;
   }
   if ( TMath::Abs(invCov.Determinant()) < 10E-120 ) {
      Log() << kFATAL << "FisherCoeff matrix is singular with determinant="
              << TMath::Abs(invCov.Determinant())  
              << " did you use the variables that are linear combinations?" 
              << Endl;
   }

   invCov.Invert();
   
   // apply rescaling factor
   Double_t xfact = TMath::Sqrt( sumOfWeightsS*sumOfWeightsB ) / (sumOfWeightsS + sumOfWeightsB);

   // compute difference of mean values
   std::vector<Double_t> diffMeans( nFisherVars );

   for (UInt_t ivar=0; ivar<=fNvars; ivar++) fisherCoeff[ivar] = 0;
   for (UInt_t ivar=0; ivar<nFisherVars; ivar++) {
      for (UInt_t jvar=0; jvar<nFisherVars; jvar++) {
         Double_t d = (*meanMatx)(jvar, 0) - (*meanMatx)(jvar, 1);
         fisherCoeff[mapVarInFisher[ivar]] += invCov(ivar, jvar)*d;
      }    
    
      // rescale
      fisherCoeff[mapVarInFisher[ivar]] *= xfact;
   }

   // offset correction
   Double_t f0 = 0.0;
   for (UInt_t ivar=0; ivar<nFisherVars; ivar++){ 
      f0 += fisherCoeff[mapVarInFisher[ivar]]*((*meanMatx)(ivar, 0) + (*meanMatx)(ivar, 1));
   }
   f0 /= -2.0;  

   fisherCoeff[fNvars] = f0;  //as we start counting variables from "zero", I store the fisher offset at the END
   
   return fisherCoeff;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::TrainNodeFull( const vector<TMVA::Event*> & eventSample,
                                           TMVA::DecisionTreeNode *node )
{
  
   // train a node by finding the single optimal cut for a single variable
   // that best separates signal and background (maximizes the separation gain)
  
   Double_t nTotS = 0.0, nTotB = 0.0;
   Int_t nTotS_unWeighted = 0, nTotB_unWeighted = 0;  
  
   vector<TMVA::BDTEventWrapper> bdtEventSample;
  
   // List of optimal cuts, separation gains, and cut types (removed background or signal) - one for each variable
   vector<Double_t> lCutValue( fNvars, 0.0 );
   vector<Double_t> lSepGain( fNvars, -1.0e6 );
   vector<Char_t> lCutType( fNvars ); // <----- bool is stored (for performance reasons, no vector<bool>  has been taken)
   lCutType.assign( fNvars, Char_t(kFALSE) );
  
   // Initialize (un)weighted counters for signal & background
   // Construct a list of event wrappers that point to the original data
   for( vector<TMVA::Event*>::const_iterator it = eventSample.begin(); it != eventSample.end(); ++it ) {
      if((*it)->GetClass() == fSigClass) { // signal or background event
         nTotS += (*it)->GetWeight();
         ++nTotS_unWeighted;
      }
      else {
         nTotB += (*it)->GetWeight();
         ++nTotB_unWeighted;
      }
      bdtEventSample.push_back(TMVA::BDTEventWrapper(*it));
   }
  
   vector<Char_t> useVariable(fNvars); // <----- bool is stored (for performance reasons, no vector<bool>  has been taken)
   useVariable.assign( fNvars, Char_t(kTRUE) );

   for (UInt_t ivar=0; ivar < fNvars; ivar++) useVariable[ivar]=Char_t(kFALSE);
   if (fRandomisedTree) { // choose for each node splitting a random subset of variables to choose from
      if (fUseNvars ==0 ) { // no number specified ... choose s.th. which hopefully works well 
         // watch out, should never happen as it is initialised automatically in MethodBDT already!!!
         fUseNvars        =  UInt_t(TMath::Sqrt(fNvars)+0.6);
      }
      Int_t nSelectedVars = 0;
      while (nSelectedVars < fUseNvars) {
         Double_t bla = fMyTrandom->Rndm()*fNvars;
         useVariable[Int_t (bla)] = Char_t(kTRUE);
         nSelectedVars = 0;
         for (UInt_t ivar=0; ivar < fNvars; ivar++) {
            if(useVariable[ivar] == Char_t(kTRUE)) nSelectedVars++;
         }
      }
   } 
   else {
      for (UInt_t ivar=0; ivar < fNvars; ivar++) useVariable[ivar] = Char_t(kTRUE);
   }
  
   for( UInt_t ivar = 0; ivar < fNvars; ivar++ ) { // loop over all discriminating variables
      if(!useVariable[ivar]) continue; // only optimze with selected variables
      TMVA::BDTEventWrapper::SetVarIndex(ivar); // select the variable to sort by
      std::sort( bdtEventSample.begin(),bdtEventSample.end() ); // sort the event data 
    
      Double_t bkgWeightCtr = 0.0, sigWeightCtr = 0.0;
      vector<TMVA::BDTEventWrapper>::iterator it = bdtEventSample.begin(), it_end = bdtEventSample.end();
      for( ; it != it_end; ++it ) {
         if((**it)->GetClass() == fSigClass ) // specify signal or background event
            sigWeightCtr += (**it)->GetWeight();
         else 
            bkgWeightCtr += (**it)->GetWeight(); 
         // Store the accumulated signal (background) weights
         it->SetCumulativeWeight(false,bkgWeightCtr); 
         it->SetCumulativeWeight(true,sigWeightCtr);
      }
    
      const Double_t fPMin = 1.0e-6;
      Bool_t cutType = kFALSE;
      Long64_t index = 0;
      Double_t separationGain = -1.0, sepTmp = 0.0, cutValue = 0.0, dVal = 0.0, norm = 0.0;
      // Locate the optimal cut for this (ivar-th) variable
      for( it = bdtEventSample.begin(); it != it_end; ++it ) {
         if( index == 0 ) { ++index; continue; }
         if( *(*it) == NULL ) {
            Log() << kFATAL << "In TrainNodeFull(): have a null event! Where index=" 
                  << index << ", and parent node=" << node->GetParent() << Endl;
            break;
         }
         dVal = bdtEventSample[index].GetVal() - bdtEventSample[index-1].GetVal();
         norm = TMath::Abs(bdtEventSample[index].GetVal() + bdtEventSample[index-1].GetVal());
         // Only allow splits where both daughter nodes have the specified miniumum number of events
         // Splits are only sensible when the data are ordered (eg. don't split inside a sequence of 0's)
         if( index >= fMinSize && (nTotS_unWeighted + nTotB_unWeighted) - index >= fMinSize && TMath::Abs(dVal/(0.5*norm + 1)) > fPMin ) {
            sepTmp = fSepType->GetSeparationGain( it->GetCumulativeWeight(true), it->GetCumulativeWeight(false), sigWeightCtr, bkgWeightCtr );
            if( sepTmp > separationGain ) {
               separationGain = sepTmp;
               cutValue = it->GetVal() - 0.5*dVal; 
               Double_t nSelS = it->GetCumulativeWeight(true);
               Double_t nSelB = it->GetCumulativeWeight(false);
               // Indicate whether this cut is improving the node purity by removing background (enhancing signal)
               // or by removing signal (enhancing background)
               if( nSelS/sigWeightCtr > nSelB/bkgWeightCtr ) cutType = kTRUE; 
               else cutType = kFALSE; 
            }
         }
         ++index;
      }
      lCutType[ivar] = Char_t(cutType);
      lCutValue[ivar] = cutValue;
      lSepGain[ivar] = separationGain;
   }
  
   Double_t separationGain = -1.0;
   Int_t iVarIndex = -1;
   for( UInt_t ivar = 0; ivar < fNvars; ivar++ ) {
      if( lSepGain[ivar] > separationGain ) {
         iVarIndex = ivar;
         separationGain = lSepGain[ivar];
      }
   }
  
   if(iVarIndex >= 0) {
      node->SetSelector(iVarIndex);
      node->SetCutValue(lCutValue[iVarIndex]);
      node->SetSeparationGain(lSepGain[iVarIndex]);
      node->SetCutType(lCutType[iVarIndex]);
      fVariableImportance[iVarIndex] += separationGain*separationGain * (nTotS+nTotB) * (nTotS+nTotB);
   }
   else {
      separationGain = 0.0;
   }
  
   return separationGain;
}

//___________________________________________________________________________________
TMVA::DecisionTreeNode* TMVA::DecisionTree::GetEventNode(const TMVA::Event & e) const
{
   // get the pointer to the leaf node where a particular event ends up in...
   // (used in gradient boosting)

   TMVA::DecisionTreeNode *current = (TMVA::DecisionTreeNode*)this->GetRoot();
   while(current->GetNodeType() == 0) { // intermediate node in a tree
      current = (current->GoesRight(e)) ?
         (TMVA::DecisionTreeNode*)current->GetRight() :
         (TMVA::DecisionTreeNode*)current->GetLeft();
   }
   return current;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::CheckEvent( const TMVA::Event & e, Bool_t UseYesNoLeaf ) const
{
   // the event e is put into the decision tree (starting at the root node)
   // and the output is NodeType (signal) or (background) of the final node (basket)
   // in which the given events ends up. I.e. the result of the classification if
   // the event for this decision tree.
  
   TMVA::DecisionTreeNode *current = this->GetRoot();
   if (!current)
      Log() << kFATAL << "CheckEvent: started with undefined ROOT node" <<Endl;

   while (current->GetNodeType() == 0) { // intermediate node in a (pruned) tree
      current = (current->GoesRight(e)) ? 
         current->GetRight() :
         current->GetLeft();
      if (!current) {
         Log() << kFATAL << "DT::CheckEvent: inconsistent tree structure" <<Endl;
      }

   }
  
   if ( DoRegression() ){
      return current->GetResponse();
   } 
   else {
      if (UseYesNoLeaf) return Double_t ( current->GetNodeType() );
      else              return current->GetPurity();
   }
}

//_______________________________________________________________________
Double_t  TMVA::DecisionTree::SamplePurity( vector<TMVA::Event*> eventSample )
{
   // calculates the purity S/(S+B) of a given event sample
  
   Double_t sumsig=0, sumbkg=0, sumtot=0;
   for (UInt_t ievt=0; ievt<eventSample.size(); ievt++) {
      if (eventSample[ievt]->GetClass() != fSigClass) sumbkg+=eventSample[ievt]->GetWeight();
      else sumsig+=eventSample[ievt]->GetWeight();
      sumtot+=eventSample[ievt]->GetWeight();
   }
   // sanity check
   if (sumtot!= (sumsig+sumbkg)){
      Log() << kFATAL << "<SamplePurity> sumtot != sumsig+sumbkg"
            << sumtot << " " << sumsig << " " << sumbkg << Endl;
   }
   if (sumtot>0) return sumsig/(sumsig + sumbkg);
   else return -1;
}

//_______________________________________________________________________
vector< Double_t >  TMVA::DecisionTree::GetVariableImportance()
{
   // Return the relative variable importance, normalized to all
   // variables together having the importance 1. The importance in
   // evaluated as the total separation-gain that this variable had in
   // the decision trees (weighted by the number of events)
  
   vector<Double_t> relativeImportance(fNvars);
   Double_t  sum=0;
   for (UInt_t i=0; i< fNvars; i++) {
      sum += fVariableImportance[i];
      relativeImportance[i] = fVariableImportance[i];
   } 
  
   for (UInt_t i=0; i< fNvars; i++) {
      if (sum > std::numeric_limits<double>::epsilon())
         relativeImportance[i] /= sum;
      else 
         relativeImportance[i] = 0;
   } 
   return relativeImportance;
}

//_______________________________________________________________________
Double_t  TMVA::DecisionTree::GetVariableImportance( UInt_t ivar )
{
   // returns the relative improtance of variable ivar
  
   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar < fNvars) return relativeImportance[ivar];
   else {
      Log() << kFATAL << "<GetVariableImportance>" << Endl
            << "---                     ivar = " << ivar << " is out of range " << Endl;
   }
  
   return -1;
}

