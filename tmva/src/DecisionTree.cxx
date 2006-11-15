// @(#)root/tmva $Id: DecisionTree.cxx,v 1.45 2006/11/14 15:21:00 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: DecisionTree.cxx,v 1.45 2006/11/14 15:21:00 stelzer Exp $
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

#include "TMVA/DecisionTree.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/BinarySearchTree.h"

#include "TMVA/Tools.h"

#include "TMVA/GiniIndex.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/SdivSqrtSplusB.h"
#include "TMVA/Event.h"

using std::vector;

#define USE_HELGESCODE 1    // the other one is Dougs implementation of the TrainNode
#define USE_HELGE_V1  0     // out loop is over NVAR in TrainNode, inner loop is Eventloop

ClassImp(TMVA::DecisionTree)
   ;

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( void )
   : BinaryTree(),
     fNvars      (0),
     fNCuts      (-1),
     fSepType    (NULL),
     fMinSize    (0),
     fPruneMethod(kExpectedErrorPruning),
     fDepth (0),
     fQualityIndex(NULL)
{
   // default constructor using the GiniIndex as separation criterion, 
   // no restrictions on minium number of events in a leave note or the
   // separation gain in the node splitting

   fLogger.SetSource( "DecisionTree" );
}

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( DecisionTreeNode* n )
   : BinaryTree(),
     fNvars      (0),
     fNCuts      (-1),
     fSepType    (NULL),
     fMinSize    (0),
     fPruneMethod(kExpectedErrorPruning),
     fDepth (0),
     fQualityIndex(NULL)
{
   // default constructor using the GiniIndex as separation criterion, 
   // no restrictions on minium number of events in a leave note or the
   // separation gain in the node splitting

   fLogger.SetSource( "DecisionTree" );

   this->SetRoot( n );
   this->SetParentTreeInNodes();
   fLogger.SetSource( "DecisionTree" );
}



//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( TMVA::SeparationBase *sepType,Int_t minSize,
                                  Int_t nCuts, TMVA::SeparationBase *qtype):
   BinaryTree(),
   fNvars      (0),
   fNCuts      (nCuts),
   fSepType    (sepType),
   fMinSize    (minSize),
   fPruneMethod(kExpectedErrorPruning),
   fDepth (0),
   fQualityIndex(qtype)
{
   // constructor specifying the separation type, the min number of
   // events in a no that is still subjected to further splitting, the
   // number of bins in the grid used in applying the cut for the node
   // splitting.

   fLogger.SetSource( "DecisionTree" );
}

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( const DecisionTree &d):
   BinaryTree(),
   fNvars      (d.fNvars),
   fNCuts      (d.fNCuts),
   fSepType    (d.fSepType),
   fMinSize    (d.fMinSize),
   fPruneMethod(d.fPruneMethod),
   fDepth      (d.fDepth),
   fQualityIndex(d.fQualityIndex)
{
   // copy constructor that creates a true copy, i.e. a completely independent tree 
   // the node copy will recursively copy all the nodes 
   this->SetRoot( new DecisionTreeNode ( *((DecisionTreeNode*)(d.GetRoot())) ) );
   this->SetParentTreeInNodes();
   fNNodes = d.fNNodes;
   fLogger.SetSource( "DecisionTree" );
}


//_______________________________________________________________________
TMVA::DecisionTree::~DecisionTree( void )
{
   // destructor
   // too bad, I cannot delete this as I don't know how to make a proper copy
   // when copying a decsiont Tree.. hence so far I just copy the pointer, which
   // means I cannot delet it here...
   //   delete fQualityIndex; 
}

//_______________________________________________________________________
void TMVA::DecisionTree::SetParentTreeInNodes( DecisionTreeNode *n){
   // descend a tree to find all its leaf nodes, fill max depth reached in the
   // tree at the same time. 

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      if (n == NULL) return ;
   } 

   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) != NULL) ) {
      fLogger << kFatal << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   }  else if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) == NULL) ) {
      fLogger << kFatal << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   } else { 
      if (this->GetLeftDaughter(n) != NULL){
         this->SetParentTreeInNodes( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         this->SetParentTreeInNodes( this->GetRightDaughter(n) );
      }
   }
   n->SetParentTree(this);
   if (n->GetDepth() > fDepth) fDepth = n->GetDepth();
   return;
}



//_______________________________________________________________________
Int_t TMVA::DecisionTree::BuildTree( vector<TMVA::Event*> & eventSample,
                                     TMVA::DecisionTreeNode *node )
{
   // building the decision tree by recursively calling the splitting of 
   // one (root-) node into two daughter nodes (returns the number of nodes)
   
   if (node==NULL) {
      //start with the root node
      node = new TMVA::DecisionTreeNode();
      fNNodes = 1;   
      this->SetRoot(node);
      // have to use "s" for start as "r" for "root" would be the same as "r" for "right"
      this->GetRoot()->SetPos('s');
      this->GetRoot()->SetDepth(0);
   }

   UInt_t nevents = eventSample.size();
   if (nevents > 0 ) {
      fNvars = eventSample[0]->GetNVars();
      fVariableImportance.resize(fNvars);
   }
   else fLogger << kFATAL << ":<BuildTree> eventsample Size == 0 " << Endl;

   Double_t s=0, b=0;
   for (UInt_t i=0; i<eventSample.size(); i++){
      if (eventSample[i]->IsSignal())
         s += eventSample[i]->GetWeight();
      else
         b += eventSample[i]->GetWeight();
   }
   node->SetNSigEvents(s);
   node->SetNBkgEvents(b);
   if (node == this->GetRoot()) node->SetNEvents(s+b);
   node->SetSeparationIndex(fSepType->GetSeparationIndex(s,b));

   //   if ( eventSample.size() > fMinSize  &&
   //        node->GetSoverSB() < fSoverSBUpperThreshold      &&
   //        node->GetSoverSB() > fSoverSBLowerThreshold  ) {
   if ( eventSample.size() > fMinSize &&
        node->GetSoverSB()*eventSample.size() > fMinSize     &&
        node->GetSoverSB()*eventSample.size() < eventSample.size()-fMinSize ) {

      Double_t separationGain;
      separationGain = this->TrainNode(eventSample, node);
      if (separationGain == 0) {//we could not gain anything, e.g. all events are in one bin, 
         // hence no cut can actually do anything. Happens for Integer Variables
         // Hence natuarlly the current node is a leaf node
         if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
         else node->SetNodeType(-1);
         if (node->GetDepth() > fDepth) fDepth = node->GetDepth();
      }else{
         vector<TMVA::Event*> leftSample; leftSample.reserve(nevents);
         vector<TMVA::Event*> rightSample; rightSample.reserve(nevents);
         Double_t nRight=0, nLeft=0;
         for (UInt_t ie=0; ie< nevents ; ie++){
            if (node->GoesRight(*eventSample[ie])){
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
            fLogger << kFATAL << "<TrainNode> all events went to the same branch" << Endl
                    << "---                       Hence new node == old node ... check" << Endl
                    << "---                         left:" << leftSample.size()
                    << " right:" << rightSample.size() << Endl
                    << "--- this should never happen, please write a bug report to Helge.Voss@cern.ch"
                    << Endl;
         }
    
         // continue building daughter nodes for the left and the right eventsample
         TMVA::DecisionTreeNode *rightNode = new TMVA::DecisionTreeNode(node,'r');
         fNNodes++;
         //         rightNode->SetPos('r');
         //         rightNode->SetDepth( node->GetDepth() + 1 );
         rightNode->SetNEvents(nRight);

         TMVA::DecisionTreeNode *leftNode = new TMVA::DecisionTreeNode(node,'l');
         fNNodes++;
         //         leftNode->SetPos('l');
         //         leftNode->SetDepth( node->GetDepth() + 1 );
         leftNode->SetNEvents(nLeft);
         
         node->SetNodeType(0);
         node->SetLeft(leftNode);
         node->SetRight(rightNode);
         this->BuildTree(rightSample, rightNode);
         this->BuildTree(leftSample,  leftNode );
      }
   } 
   else{ // it is a leaf node
      if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
      else node->SetNodeType(-1);
      if (node->GetDepth() > fDepth) fDepth = node->GetDepth();
   }
   
   return fNNodes;
}



//_______________________________________________________________________
void TMVA::DecisionTree::FillTree( vector<TMVA::Event*> & eventSample)

{
   // fill the existing the decision tree structure by filling event
   // in from the top node and see where they happen to end up
   for (UInt_t i=0; i<eventSample.size(); i++){
      this->FillEvent(*(eventSample[i]),NULL);
   }
}


//_______________________________________________________________________
void TMVA::DecisionTree::FillEvent( TMVA::Event & event,  
                                     TMVA::DecisionTreeNode *node  )
{
   // fill the existing the decision tree structure by filling event
   // in from the top node and see where they happen to end up
   
   if (node == NULL) { // that's the start, take the Root node
      node = (TMVA::DecisionTreeNode*)this->GetRoot();
   }

   node->IncrementNEvents( event.GetWeight() );
                         
   if (event.IsSignal())
      node->IncrementNSigEvents( event.GetWeight() );
   else
      node->IncrementNBkgEvents( event.GetWeight() );

   node->SetSeparationIndex(fSepType->GetSeparationIndex(node->GetNSigEvents(),
                                                         node->GetNBkgEvents()));
   
   if (node->GetNodeType() == 0){ //intermediate node --> go down
      if (node->GoesRight(event))
         this->FillEvent(event,(TMVA::DecisionTreeNode*)(node->GetRight())) ;
      else
         this->FillEvent(event,(TMVA::DecisionTreeNode*)(node->GetLeft())) ;
   }


}


//_______________________________________________________________________
void TMVA::DecisionTree::ClearTree(){
   // clear the tree nodes (their S/N, Nevents etc), just keep the structure of the tree

   if (this->GetRoot()!=NULL) 
      ((DecisionTreeNode*)(this->GetRoot()))->ClearNodeAndAllDaughters();

}


//_______________________________________________________________________
void TMVA::DecisionTree::PruneTree(){
   // prune (get rid of internal nodes) the Decision tree to avoid overtraining
   // serveral different pruning methods can be applied as selected by the 
   // variable "fPruneMethod". Currently however only the Expected Error Pruning
   // is implemented
   // 
   if (fPruneMethod == kExpectedErrorPruning) {
      this->PruneTreeEEP((DecisionTreeNode *)this->GetRoot());
   } else if (fPruneMethod == kCostComplexityPruning) {
      this->PruneTreeCC();
   } else if (fPruneMethod == kMCC) {
      this->PruneTreeMCC();
   } else {
      fLogger << kFatal << "Selected pruning method not yet implemented "
              << Endl;
   }
   //update the number of nodes after the pruning
   this->CountNodes();
};
      


//_______________________________________________________________________
void TMVA::DecisionTree::PruneTreeEEP(DecisionTreeNode *node){
   // recursive prunig of nodes using the Expected Error Pruning (EEP)
   // if internal node, then prune
   DecisionTreeNode *l = (DecisionTreeNode*)node->GetLeft();
   DecisionTreeNode *r = (DecisionTreeNode*)node->GetRight();
   if (node->GetNodeType() == 0){
      this->PruneTreeEEP(l);
      this->PruneTreeEEP(r);
      if (this->GetSubTreeError(node) >= this->GetNodeError(node)) { 
//          cout << "bla: prune away node at depth " << node->GetDepth()
//               << " and sequence " << node->GetSequence() <<endl;
         this->PruneNode(node);
      }
   } 
}

//_______________________________________________________________________
void TMVA::DecisionTree::PruneTreeCC(){
   // prunig of nodes using the Cost Complexity criteria. The Pruning is performed
   // until a minimum in the cost complexity CC(alpha) is reached.
   // CC(alpha) = alpha*NLeafs + sum_over_leafs[ N*Quality(leaf) ]
   // where Quality(leaf) is given by the 1-purity (for Misclassification Error)
   // purity(1-purity) for Gini-Index..e.t.c.. typically the Misclassification Error
   // is used for guiding the pruning.
   
   // you keep pruning the nodes that give the smallest gain in quality until the
   // CC has reached its minimum. I.e...you keep pruning nodes until the next pruning
   // step would result in a CC that is larger than the CC of the current tree.

   Double_t currentCC = this->GetCostComplexity(fPruneStrength);
   Double_t nextCC    = this->GetCostComplexityIfNextPruneStep(fPruneStrength);

   while (currentCC > nextCC &&  this->GetNNodes() > 3 ){//find the weakest node and prune it away
      this->PruneNode( this->FindCCPruneCandidate() );
      currentCC = this->GetCostComplexity(fPruneStrength);
      nextCC    = this->GetCostComplexityIfNextPruneStep(fPruneStrength);
   }
   return;
}


//_______________________________________________________________________
void TMVA::DecisionTree::PruneTreeMCC(){
   // Similar to the CostCoplexity pruning, only here I calculate immediately
   // the "prunestrength" (= alpha, the regularisation parameter in the CostComplexity)
   // for which the respective subtree below a node would be pruned. Then I continue
   // pruning until all nodes have such a value larger than the specified prunestrength

   this->FillLinkStrengthMap();
   Double_t currentG = fLinkStrengthMap.begin()->first;

   
   while (currentG < fPruneStrength &&  this->GetNNodes() > 3 ){//find the weakest node and prune it away
      //      this->PruneNode( fLinkStrengthMap.begin()->second );
      this->PruneNode( this->GetWeakestLink() );
      currentG = fLinkStrengthMap.begin()->first;
      //      cout << "bla: currentG " << currentG << " NNodes: " << this->GetNNodes() << endl;
   }
   return;
}



//_______________________________________________________________________
TMVA::DecisionTreeNode*  TMVA::DecisionTree::GetWeakestLink() 
{
   // get the weakest link for the pruning step
   this->FillLinkStrengthMap();
   return fLinkStrengthMap.begin()->second;
}

//_______________________________________________________________________
void TMVA::DecisionTree::FillLinkStrengthMap(TMVA::DecisionTreeNode *n) 
{
   // loop over all non-leaf nodes of the tree and calculate for each
   // of these nodes the prunestrenght ("alpha") at which this node
   // would win against its subtree.(hence it's subtree would be pruned)
   // this is given by:
   //          R(t) - R(T_t)       where R(t)  : MisClassificationCost of the node t
   // alpha < --------------             R(T_t): MisClassificationCost of subtree blow t
   //          |T_t| - 1                 |T|   : # nodes in Tree T
   //



   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      fLinkStrengthMap.clear();
      if (n == NULL) return ;
   } 
   if (this->GetLeftDaughter(n) != NULL){
      this->FillLinkStrengthMap( this->GetLeftDaughter(n)); 
   }
   if (this->GetRightDaughter(n) != NULL) {
      this->FillLinkStrengthMap( this->GetRightDaughter(n));
   }
   
   //now you are intrested  non-leaf nodes only
   if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) != NULL) ) {   
      // R(t) = 
      Double_t alpha = ( this->MisClassificationCostOfNode(n)  -
                         this->MisClassificationCostOfSubTree(n) ) /  
         (n->CountMeAndAllDaughters() - 1);

      fLinkStrengthMap.insert(pair<const Double_t, TMVA::DecisionTreeNode* > ( alpha, n ));
   }
}


//_______________________________________________________________________
Double_t TMVA::DecisionTree::MisClassificationCostOfNode(TMVA::DecisionTreeNode *n)
{
   // get the misclassificationCost of the subTree
   return (1 - n->GetPurity()) * n->GetNEvents(); //  / this->GetNNodes() ;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::MisClassificationCostOfSubTree(TMVA::DecisionTreeNode *n)
{
   // get the misclassificationCost of the subTree
   Double_t tmp=0;

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      if (n == NULL) return 0;
   } 
   if (this->GetLeftDaughter(n) != NULL){
      tmp += this->MisClassificationCostOfSubTree( this->GetLeftDaughter(n)); 
   }
   if (this->GetRightDaughter(n) != NULL) {
      tmp += this->MisClassificationCostOfSubTree( this->GetRightDaughter(n));
   }
   
   //now you are interested  leaf nodes only
   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {   
      tmp = this->MisClassificationCostOfNode(n);
   }

   return tmp;
}

//_______________________________________________________________________
UInt_t TMVA::DecisionTree::CountLeafNodes(TMVA::DecisionTreeNode *n)
{
   // return the number of terminal nodes in the sub-tree below Node n

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      if (n == NULL) return 0 ;
   } 

   UInt_t countLeafs=0;

   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {
      //      cout << " bla: found a leaf lode, right ?? " << n->GetNodeType() << endl; 
      countLeafs += 1;
   } else { 
      if (this->GetLeftDaughter(n) != NULL){
         countLeafs += this->CountLeafNodes( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         countLeafs += this->CountLeafNodes( this->GetRightDaughter(n) );
      }
   }
   return countLeafs;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetCostComplexity(Double_t alpha) 
{
   // returns the cost complexity criterion for the decision tree
   // see "L.Breiman, J.H.Friedman, R.A.Olshen, C.J.Stone; "Classification and
   // Regression Trees", Wadsworth International Group (1984), Chapman & Hall/CRC (1984)
   
   // even though for guiding the cross complexity pruning, any index (gini index,
   // cross entropy or miscalssifiaction error can be used, typically one
   // chooses the "misclassification error" .. so will I.
   // (taken from "Elements of Statistical Learning" page 271)

   // find all leaf nodes
   
   Double_t cc=0.;

   this->FillQualityMap();
   multimap<Double_t, TMVA::DecisionTreeNode* >::iterator it=fQualityMap.begin();
   Int_t count=0;
   for (;it!=fQualityMap.end(); it++){
      Double_t s=it->second->GetNSigEvents();
      Double_t b=it->second->GetNBkgEvents();
      cc += (s+b) * it->first ;  
      count++;
   }

   //   cout << "bla  Tree size:" << this->GetNNodes() << " Leaf Nodes: " << count << "  " << fQualityMap.size() 
   //     << " Cost Complexity " << cc << "  " << cc+alpha*count <<endl;
   return cc+alpha * count;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetCostComplexityIfNextPruneStep(Double_t alpha) 
{
   // returns the cost complexity criterion for the decision tree
   // see "L.Breiman, J.H.Friedman, R.A.Olshen, C.J.Stone; "Classification and
   // Regression Trees", Wadsworth International Group (1984), Chapman & Hall/CRC (1984)
   
   // even though for guiding the cross complexity pruning, any index (gini index,
   // cross entropy or miscalssifiaction error can be used, typically one
   // chooses the "misclassification error" 
   // (taken from "Elements of Statistical Learning" page 271)
   // This however for me shows strange behaviours. In particulary, there are nodes 
   // which have daughters that both have a purity > 0.5. Hence for the actual 
   // Misclassificaion error of 1-max(p,1-p), the gain in quality:
   // (S + B) Q(S,B) - (s+b)Q(s,b) - ( (S-s)+(B-s) Q(S-s,B-b)) turns out to be == 0
   // which leads to non-convex behaviour of "CostComplexity vs tree-size"

   // find all leaf nodes
   
   Double_t cc=0.;

   this->FillQualityMap();
   this->FillQualityGainMap();

   multimap<Double_t, TMVA::DecisionTreeNode* >::iterator it=fQualityMap.begin();
   Int_t count=0;
   for (;it!=fQualityMap.end(); it++){
      if (it->second->GetParent() != fQualityGainMap.begin()->second ) {
         Double_t s=it->second->GetNSigEvents();
         Double_t b=it->second->GetNBkgEvents();
         cc += (s+b) * it->first ;  
         count++;
      } 
   }
   //now add the pruning candidates contribution as if it were pruned
   Double_t s=fQualityGainMap.begin()->second->GetNSigEvents();
   Double_t b=fQualityGainMap.begin()->second->GetNBkgEvents();
   
   cc += (s+b) * fQualityIndex->GetSeparationIndex(s,b);
   count++;
   
   cc+=alpha*count;
  
   return cc;
}


//_______________________________________________________________________
void TMVA::DecisionTree::FillQualityGainMap(DecisionTreeNode* n )
{
   // traverse the whole tree and fill the map of QualityGain - Node
   // for pre-leaf nodes, deciding which node is the next prune candidate


   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      fQualityGainMap.clear();
      if (n == NULL) return ;
   } 
   
   if (this->GetLeftDaughter(n) != NULL){
      this->FillQualityGainMap( this->GetLeftDaughter(n)); 
   }
   if (this->GetRightDaughter(n) != NULL) {
      this->FillQualityGainMap( this->GetRightDaughter(n));
   }
   
   //quality gain of course exists for internal nodes only:
   if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) != NULL) ) {   
      //but you want to fill it for pre-leaf nodes only
      if ((this->GetLeftDaughter(n)->GetLeft() == NULL) && 
          (this->GetLeftDaughter(n)->GetRight() == NULL) && 
          (this->GetRightDaughter(n)->GetLeft() == NULL) && 
          (this->GetRightDaughter(n)->GetRight() == NULL) ){
         
         fQualityGainMap.insert(pair<const Double_t, TMVA::DecisionTreeNode* > 
                                ( fQualityIndex->GetSeparationGain (this->GetRightDaughter(n)->GetNSigEvents(),
                                                                    this->GetRightDaughter(n)->GetNBkgEvents(),
                                                                    n->GetNSigEvents(), n->GetNBkgEvents()),
                                  n));
      }
   }      
   return;
}

//_______________________________________________________________________
void TMVA::DecisionTree::FillQualityMap(DecisionTreeNode* n )
{
   // traverse the whole tree and find the leaf nodes, and then fill the Quality Criterion
   // for the leaf nodes (Used in the Pruning)

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      fQualityMap.clear();
      if (n == NULL) return ;
   } 
   
   if (this->GetLeftDaughter(n) != NULL){
      this->FillQualityMap( this->GetLeftDaughter(n)); 
   }
   if (this->GetRightDaughter(n) != NULL) {
      this->FillQualityMap( this->GetRightDaughter(n));
   }
   
   //now you are intrested in leaf nodes only
   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {   
      
      fQualityMap.insert(pair<const Double_t, TMVA::DecisionTreeNode* > 
                         ( fQualityIndex->GetSeparationIndex (n->GetNSigEvents(), 
                                                              n->GetNBkgEvents()),
                           n));
   }
   return;
}

   
//_______________________________________________________________________
void TMVA::DecisionTree::DescendTree( DecisionTreeNode *n)
{
   // descend a tree to find all its leaf nodes

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (DecisionTreeNode*) this->GetRoot();
      if (n == NULL) return ;
   } 

   if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) == NULL) ) {
      //      cout << " bla: found a leaf lode, right ?? " << n->GetNodeType() << endl; 
   } else if ((this->GetLeftDaughter(n) == NULL) && (this->GetRightDaughter(n) != NULL) ) {
      fLogger << kFatal << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   }  else if ((this->GetLeftDaughter(n) != NULL) && (this->GetRightDaughter(n) == NULL) ) {
      fLogger << kFatal << " Node with only one daughter?? Something went wrong" << Endl;
      return;
   } else { 
      if (this->GetLeftDaughter(n) != NULL){
         this->DescendTree( this->GetLeftDaughter(n) );
      }
      if (this->GetRightDaughter(n) != NULL) {
         this->DescendTree( this->GetRightDaughter(n) );
      }
   }
   return;
}


//_______________________________________________________________________
TMVA::DecisionTreeNode* TMVA::DecisionTree::FindCCPruneCandidate()
{
   // get the  pruning candidate node as the one pre-leaf node that gives
   // the smalest quality gain
   this->FillQualityGainMap();
   return fQualityGainMap.begin()->second;
}

//_______________________________________________________________________
void TMVA::DecisionTree::PruneNode(DecisionTreeNode *node)
{
   // prune away the subtree below the node 
   
   DecisionTreeNode *l = (DecisionTreeNode*)node->GetLeft();
   DecisionTreeNode *r = (DecisionTreeNode*)node->GetRight();
   
   node->SetRight(NULL);
   node->SetLeft(NULL);
   if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
   else node->SetNodeType(-1);
   this->DeleteNode(l);
   this->DeleteNode(r);
   //update the stored number of Nodes in the Tree
   this->CountNodes();

}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetNodeError(DecisionTreeNode *node)
{
   // calculate an UPPER limit on the error made by the classification done
   // by this node. If the S/S+B of the node is f, then according to the
   // training sample, the error rate (fraction of misclassified events by
   // this node) is (1-f)
   // now f has a statistical error according to the binomial distribution
   // hence the error on f can be estimated (same error as the binomial error
   // for efficency calculations ( sigma = sqrt(eff(1-eff)/N) ) 
   
   
   Double_t errorRate=0;

   Double_t N=node->GetNEvents();
   
   //fraction of correctly classified events by this node:
   Double_t f=0;
   if (node->GetSoverSB() > 0.5) f = node->GetSoverSB();
   else  f = (1-node->GetSoverSB());

   Double_t df = sqrt(f*(1-f)/N);
   
   errorRate = std::min(1.,(1 - (f-fPruneStrength*df) ));
   
   // -------------------------------------------------------------------
   // Minimum Error Pruning (MEP) accordig to Niblett/Bratko
   //# of correctly classified events by this node:
   //Double_t n=f*N;
   //Double_t p_apriori = 0.5, m=100;
   //errorRate = (N - n + (1-p_apriori) * m ) / (N + m);
   
   // Pessimistic error Pruing (proposed by Quinlan (error estimat with continuity approximation)
   //# of correctly classified events by this node:
   //Double_t n=f*N;
   //errorRate = (N - n + 0.5) / N;
          
   //const Double Z=.65;
   //# of correctly classified events by this node:
   //Double_t n=f*N;
   //errorRate = (f + Z*Z/(2*N) + Z*sqrt(f/N - f*f/N + Z*Z/4/N/N) ) / (1 + Z*Z/N); 
   //errorRate = (n + Z*Z/2 + Z*sqrt(n - n*n/N + Z*Z/4) )/ (N + Z*Z);
   //errorRate = 1 - errorRate;
   // -------------------------------------------------------------------
   
   return errorRate;
}
//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetSubTreeError(DecisionTreeNode *node)
{
   // calculate the expected statistical error on the subtree below "node"
   // which is used in the expected error pruning
   DecisionTreeNode *l = (DecisionTreeNode*)node->GetLeft();
   DecisionTreeNode *r = (DecisionTreeNode*)node->GetRight();
   if (node->GetNodeType() == 0) {
      Double_t subTreeError = 
         (l->GetNEvents() * this->GetSubTreeError(l) +
          r->GetNEvents() * this->GetSubTreeError(r)) /
         node->GetNEvents();
      return subTreeError;
   }else{
      return this->GetNodeError(node);
   }
}


//_______________________________________________________________________
TMVA::DecisionTreeNode* TMVA::DecisionTree::GetLeftDaughter( DecisionTreeNode *n)
{
   // get left daughter node current node "n"
   return (DecisionTreeNode*) n->GetLeft();
}

//_______________________________________________________________________
TMVA::DecisionTreeNode* TMVA::DecisionTree::GetRightDaughter( DecisionTreeNode *n)
{
   // get right daughter node current node "n"
   return (DecisionTreeNode*) n->GetRight();
}
   

//_______________________________________________________________________
TMVA::DecisionTreeNode* TMVA::DecisionTree::GetNode(ULong_t sequence, UInt_t depth)
{
   // retrieve node from the tree. Its position (up to a maximal tree depth of 64)
   // is coded as a sequence of left-right moves starting from the root, coded as
   // 0-1 bit patterns stored in the "long-integer"  (i.e. 0:left ; 1:right

   DecisionTreeNode* current = (DecisionTreeNode*) this->GetRoot();
   
   for (UInt_t i =0;  i < depth; i++){
      ULong_t tmp = 1 << i;
      if ( tmp & sequence) current = this->GetRightDaughter(current);
      else current = this->GetLeftDaughter(current);
   }

   
   return current;
}

//_______________________________________________________________________
void TMVA::DecisionTree::FindMinAndMax(vector<TMVA::Event*> & eventSample,
                                       vector<Double_t> & xmin,
                                       vector<Double_t> & xmax)
{
   // helper function which calculates gets the Min and Max value
   // of the event variables in the current event sample

   UInt_t num_events = eventSample.size();
  
   for (Int_t ivar=0; ivar < fNvars; ivar++){
      xmin[ivar]=xmax[ivar]=eventSample[0]->GetVal(ivar);
   }
  
   for (UInt_t i=1;i<num_events;i++){
      for (Int_t ivar=0; ivar < fNvars; ivar++){
         if (xmin[ivar]>eventSample[i]->GetVal(ivar))
            xmin[ivar]=eventSample[i]->GetVal(ivar);
         if (xmax[ivar]<eventSample[i]->GetVal(ivar))
            xmax[ivar]=eventSample[i]->GetVal(ivar);
      }
   }
  
};

//_______________________________________________________________________
void  TMVA::DecisionTree::SetCutPoints(vector<Double_t> & cut_points,
                                       Double_t xmin,
                                       Double_t xmax,
                                       Int_t num_gridpoints)
{
   // helper function which calculates the grid points used for
   // the cut scan
   Double_t step = (xmax - xmin)/num_gridpoints;
   Double_t x = xmin + step/2; 
   for (Int_t j=0; j < num_gridpoints; j++){
      cut_points[j] = x;
      x += step;
   }
};

//_______________________________________________________________________

#if USE_HELGESCODE==1
Double_t TMVA::DecisionTree::TrainNode(vector<TMVA::Event*> & eventSample,
                                       TMVA::DecisionTreeNode *node)
{
   // decide how to split a node. At each node, ONE of the variables is
   // choosen, which gives the best separation between signal and bkg on
   // the sample which enters the Node.  
   // In order to do this, for each variable a scan of the different cut
   // values in a grid (grid = fNCuts) is performed and the resulting separation
   // gains are compared.. This cut scan uses either a binary search tree
   // or a simple loop over the events depending on the number of events
   // in the sample 

   vector<Double_t> *xmin  = new vector<Double_t>( fNvars );
   vector<Double_t> *xmax  = new vector<Double_t>( fNvars );

   Double_t separationGain = -1, sepTmp;
   Double_t cutValue=-999;
   Int_t mxVar=-1, cutIndex=0;
   Bool_t cutType=kTRUE;
   Double_t  nTotS, nTotB;
   UInt_t nevents = eventSample.size();

   //find min and max value of the variables in the sample
   for (int ivar=0; ivar < fNvars; ivar++){
      (*xmin)[ivar]=(*xmax)[ivar]=eventSample[0]->GetVal(ivar);
   }
   for (UInt_t iev=1;iev<nevents;iev++){
      for (Int_t ivar=0; ivar < fNvars; ivar++){
         Double_t eventData = eventSample[iev]->GetVal(ivar); 
         if ((*xmin)[ivar]>eventData)(*xmin)[ivar]=eventData;
         if ((*xmax)[ivar]<eventData)(*xmax)[ivar]=eventData;
      }
   }

   vector< vector<Double_t> > nSelS (fNvars);
   vector< vector<Double_t> > nSelB (fNvars);
   vector< vector<Double_t> > significance (fNvars);
   vector< vector<Double_t> > cutValues(fNvars);
   vector< vector<Bool_t> > cutTypes(fNvars);

   for (int ivar=0; ivar < fNvars; ivar++){
      cutValues[ivar].resize(fNCuts);
      cutTypes[ivar].resize(fNCuts);
      nSelS[ivar].resize(fNCuts);
      nSelB[ivar].resize(fNCuts);
      significance[ivar].resize(fNCuts);

      //set the grid for the cut scan on the variables
      Double_t istepSize =( (*xmax)[ivar] - (*xmin)[ivar] ) / Double_t(fNCuts);
      for (Int_t icut=0; icut<fNCuts; icut++){
         cutValues[ivar][icut]=(*xmin)[ivar]+(Float_t(icut)+0.5)*istepSize;
      }
   }


 
#if USE_HELGE_V1==1

   // this is the alternative code, having as an outer loop the loop over the variables, and 
   // the inner loop over the event sample. I would like to keep this (it does not seem to be
   // any slower) as this would be necessary for any more clever cut optimisation algorithm
   // i can right now think of.

   nTotS=0; nTotB=0;
   for (int ivar=0; ivar < fNvars; ivar++){
      for (UInt_t iev=0; iev<nevents; iev++){

         Double_t eventData  = eventSample[iev]->GetData(ivar); 
         Int_t    eventType  = eventSample[iev]->GetType(); 
         Double_t eventWeight= eventSample[iev]->GetWeight(); 


         if (ivar==0){
            if (eventType==1){
               nTotS+=eventWeight;
            }else {
               nTotB+=eventWeight;
            }
         }
         // now scan trough the cuts for each varable and find which one gives
         // the best separationGain at the current stage.
         // just scan the possible cut values for this variable
         for (Int_t icut=0; icut<fNCuts; icut++){
            if (eventData > cutValues[ivar][icut]){
               if (eventType==1) nSelS[ivar][icut]+=eventWeight;
               else nSelB[ivar][icut]+=eventWeight;
            }
         }
      }
   }

#else 

   nTotS=0; nTotB=0;
   for (UInt_t iev=0; iev<nevents; iev++){
      Int_t eventType = eventSample[iev]->Type();
      Double_t eventWeight =  eventSample[iev]->GetWeight(); 
      if (eventType==1){
         nTotS+=eventWeight;
      }else {
         nTotB+=eventWeight;
      }

      for (int ivar=0; ivar < fNvars; ivar++){
         // now scan trough the cuts for each varable and find which one gives
         // the best separationGain at the current stage.
         // just scan the possible cut values for this variable
         Double_t eventData = eventSample[iev]->GetVal(ivar); 
         for (Int_t icut=0; icut<fNCuts; icut++){
            if (eventData > cutValues[ivar][icut]){
               if (eventType==1) nSelS[ivar][icut]+=eventWeight;
               else nSelB[ivar][icut]+=eventWeight;
            }
         }
      }
   }

#endif


   // now select the optimal cuts for each varable and find which one gives
   // the best separationGain at the current stage.
   for (int ivar=0; ivar < fNvars; ivar++){
      for (Int_t icut=0; icut<fNCuts; icut++){
         // now the separationGain is defined as the various indices (Gini, CorssEntropy, e.t.c)
         // calculated by the "SamplePurities" from the branches that would go to the
         // left or the right from this node if "these" cuts were used in the Node:
         // hereby: nSelS and nSelB would go to the right branch
         //        (nTotS - nSelS) + (nTotB - nSelB)  would go to the left branch;
       
         sepTmp = fSepType->GetSeparationGain(nSelS[ivar][icut], nSelB[ivar][icut], nTotS, nTotB);

         if (separationGain < sepTmp) {
            separationGain = sepTmp;
            mxVar = ivar;
            cutIndex = icut;
         }
      }
   }

   if (nSelS[mxVar][cutIndex]/nTotS > nSelB[mxVar][cutIndex]/nTotB) cutType=kTRUE;
   else cutType=kFALSE;
   cutValue = cutValues[mxVar][cutIndex];

   node->SetSelector((UInt_t)mxVar);
   node->SetCutValue(cutValue);
   node->SetCutType(cutType);
   node->SetSeparationGain(separationGain);

   fVariableImportance[mxVar] += separationGain*separationGain * (nTotS+nTotB) * (nTotS+nTotB) ;
 
   delete xmin;
   delete xmax;

   return separationGain;
}


#else 

//_______________________________________________________________________
Double_t TMVA::DecisionTree::TrainNode(vector<TMVA::Event*> & eventSample,
                                       TMVA::DecisionTreeNode *node)
{
   // decide how to split a node. At each node, ONE of the variables is
   // choosen, which gives the best separationGain between signal and bkg on
   // the sample which enters the Node.  
   // In order to do this, for each variable a scan of the different cut
   // values in a grid (grid = fNCuts) is performed and the resulting separationGain
   // gains are compared.. This cut scan uses a simple loop over events, 
   // but may be remodified to use binary search trees.

   vector<Double_t> xmin ( fNvars );
   vector<Double_t> xmax ( fNvars );

   Double_t separationGain = -1;
   Double_t cutValue=-999;
   Int_t mxVar=-1;
   Bool_t cutType=kTRUE;
   Double_t  nSelS=0., nSelB=0., nTotS=0., nTotB=0.;
   UInt_t num_events = eventSample.size();
   
   vector<vector<Double_t> > signal_counts (fNvars);
   vector<vector<Double_t> > background_counts (fNvars);
   vector<vector<Double_t> > cut_points (fNvars);
   vector<vector<Double_t> > significance (fNvars);
   
   this->FindMinAndMax(eventSample, xmin, xmax);

   
   for (Int_t i=0; i < fNvars; i++){
      signal_counts[i].resize(fNCuts);
      background_counts[i].resize(fNCuts);
      cut_points[i].resize(fNCuts);
      significance[i].resize(fNCuts);

      this->SetCutPoints(cut_points[i], xmin[i], xmax[i], fNCuts);
   }

   for (UInt_t event=0; event < num_events; event++){
     
      Int_t event_type = eventSample[event]->GetType();
      Double_t event_weight = eventSample[event]->GetWeight();
     
      if (event_type == 1){
         nTotS += event_weight;
      } else {
         nTotB += event_weight;
      }
     
      for (Int_t variable = 0; variable < fNvars; variable++){
         Double_t event_val = eventSample[event]->GetData(variable);
         for (Int_t cut=0; cut < fNCuts; cut++){
            if (event_val > cut_points[variable][cut]){
               if (event_type == 1){
                  signal_counts[variable][cut] += event_weight;
               } else {
                  background_counts[variable][cut] += event_weight;
               }
            }
         } 
      }
   }

   for (Int_t var = 0; var < fNvars; var++){
      for (Int_t cut=0; cut < fNCuts; cut++){
         Double_t cur_sep = fSepType->GetSeparationGain(signal_counts[var][cut],
                                                        background_counts[var][cut],
                                                        nTotS, nTotB);
         if (separationGain < cur_sep) {
            separationGain = cur_sep;
            cutValue=cut_points[var][cut];
            cutType= (nSelS/nTotS > nSelB/nTotB) ? kTRUE : kFALSE;
            mxVar = var;
         } 
      }
   }
   
   node->SetSelector(mxVar);
   node->SetCutValue(cutValue);
   node->SetCutType(cutType);
   node->SetSeparationGain(separationGain);

   fVariableImportance[mxVar] += separationGain*separationGain * (nTotS+nTotB)* (nTotS+nTotB);
  
   return separationGain;
}
#endif

//_______________________________________________________________________
Double_t TMVA::DecisionTree::CheckEvent(const TMVA::Event & e, Bool_t UseYesNoLeaf)
{
   // the event e is put into the decision tree (starting at the root node)
   // and the output is NodeType (signal) or (background) of the final node (basket)
   // in which the given events ends up. I.e. the result of the classification if
   // the event for this decision tree.

   TMVA::DecisionTreeNode *current = (TMVA::DecisionTreeNode*)this->GetRoot();

   while(current->GetNodeType() == 0){ //intermediate node
      if (current->GoesRight(e))
         current=(TMVA::DecisionTreeNode*)current->GetRight();
      else current=(TMVA::DecisionTreeNode*)current->GetLeft();
   }

   if (UseYesNoLeaf) return Double_t ( current->GetNodeType() );
   else return current->GetSoverSB();
}

//_______________________________________________________________________
Double_t  TMVA::DecisionTree::SamplePurity(vector<TMVA::Event*> eventSample)
{
   //calculates the purity S/(S+B) of a given event sample
   
   Double_t sumsig=0, sumbkg=0, sumtot=0;
   for (UInt_t ievt=0; ievt<eventSample.size(); ievt++) {
      if (eventSample[ievt]->Type()==0) sumbkg+=eventSample[ievt]->GetWeight();
      if (eventSample[ievt]->Type()==1) sumsig+=eventSample[ievt]->GetWeight();
      sumtot+=eventSample[ievt]->GetWeight();
   }
   //sanity check
   if (sumtot!= (sumsig+sumbkg)){
      fLogger << kFATAL << "<SamplePurity> sumtot != sumsig+sumbkg"
              << sumtot << " " << sumsig << " " << sumbkg << Endl;
   }
   if (sumtot>0) return sumsig/(sumsig + sumbkg);
   else return -1;
}

//_______________________________________________________________________
vector< Double_t >  TMVA::DecisionTree::GetVariableImportance()
{
   //return the relative variable importance, normalized to all
   //variables together having the importance 1. The importance in
   //evaluated as the total separation-gain that this variable had in
   //the decision trees (weighted by the number of events)
 
   vector<Double_t> relativeImportance(fNvars);
   Double_t  sum=0;
   for (int i=0; i< fNvars; i++) {
      sum += fVariableImportance[i];
      relativeImportance[i] = fVariableImportance[i];
   } 

   for (int i=0; i< fNvars; i++) {
      relativeImportance[i] /= sum;
   } 
   return relativeImportance;
  
}

//_______________________________________________________________________
Double_t  TMVA::DecisionTree::GetVariableImportance(Int_t ivar)
{
   // returns the relative improtance of variable ivar
 
   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar >= 0 && ivar < fNvars) return relativeImportance[ivar];
   else {
      fLogger << kFATAL << "<GetVariableImportance>" << Endl
              << "---                     ivar = " << ivar << " is out of range " << Endl;
   }

   return -1;
}




//_______________________________________________________________________
TH2D*  TMVA::DecisionTree::DrawTree(TString hname)
{
   // returns poiter to 2D histogram with tree drawing in it.

   Double_t xmax= 2*fDepth + 0.5;
   Double_t xmin= -xmax;


   ULong_t nbins =1; for (UInt_t i=0; i<fDepth; i++) nbins *= 2;  //  (2^depth)

   TH2D* h=new TH2D(hname,"",2*nbins+1, xmin, xmax,
                    2*fDepth+2, -0.5, 2*fDepth+0.5);

   this->DrawNode( h, (DecisionTreeNode*)this->GetRoot(), 2*fDepth, 0, Double_t(fDepth) );
  
   return h;

}   
   

   
//_______________________________________________________________________
void TMVA::DecisionTree::DrawNode( TH2D* h,  DecisionTreeNode *n, 
                                   Double_t y, Double_t x, Double_t scale)
{
   // recursively puts an entries in the histogram for the node and its daughters
   //

   if (this->GetLeftDaughter(n) != NULL){
      this->DrawNode( h, this->GetLeftDaughter(n), y-2, x-scale, scale/2.  );
   }
   if (this->GetRightDaughter(n) != NULL) {
      this->DrawNode( h, this->GetRightDaughter(n), y-2, x+scale, scale/2.);
   }
   h->Fill(x,y,n->GetNEvents());

   return;
}
