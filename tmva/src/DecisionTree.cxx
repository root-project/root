// @(#)root/tmva $Id: DecisionTree.cxx,v 1.11 2006/05/23 09:53:10 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DecisionTree                                                    *
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
 * $Id: DecisionTree.cxx,v 1.11 2006/05/23 09:53:10 stelzer Exp $
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

#include "TVirtualFitter.h"

#include "TMVA/DecisionTree.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/BinarySearchTree.h"

#include "TMVA/Tools.h"

#include "TMVA/GiniIndex.h"
#include "TMVA/CrossEntropy.h"
#include "TMVA/MisClassificationError.h"
#include "TMVA/SdivSqrtSplusB.h"

using std::vector;

ClassImp(TMVA::DecisionTree)

//_______________________________________________________________________
   TMVA::DecisionTree::DecisionTree( void ):
      fNvars      (0),
      fNCuts      (-1),
      fSepType    (new TMVA::GiniIndex()),
      fMinSize    (0)
                                   //   fSoverSBUpperThreshold (0),
                                   //   fSoverSBLowerThreshold (0)
{
   // default constructor using the GiniIndex as separation criterion, 
   // no restrictions on minium number of events in a leave note or the
   // separation gain in the node splitting
}

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( TMVA::SeparationBase *sepType,Int_t minSize,
                                  Int_t nCuts):
   fNvars      (0),
   fNCuts      (nCuts),
   fSepType    (sepType),
   fMinSize    (minSize)
                                //   fSoverSBUpperThreshold (0),
                                //   fSoverSBLowerThreshold (0)
{
   // constructor specifying the separation type, the min number of
   // events in a no that is still subjected to further splitting, the
   // min separation gain requested for actually splitting a node
   // (NEEDS TO BE SET TO ZERO, OTHERWISE I GET A STRANGE BEHAVIOUR
   // WHICH IS NOT YET COMPLETELY UNDERSTOOD) as well as the number of
   // bins in the grid used in applying the cut for the node splitting.
}

//_______________________________________________________________________
TMVA::DecisionTree::~DecisionTree( void )
{
   // destructor
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
      fNNodes++;
      fSumOfWeights+=1.;
      this->SetRoot(node);
   }

   UInt_t nevents = eventSample.size();
   if (nevents > 0 ) fNvars = eventSample[0]->GetEventSize();
   else{
      cout << "--- TMVA::DecisionTree::BuildTree:  Error, Eventsample Size == 0 " <<endl;
      exit(1);
   }

   Double_t s=0, b=0;
   for (UInt_t i=0; i<eventSample.size(); i++){
      if (eventSample[i]->GetType()==0) b+= eventSample[i]->GetWeight();
      else if (eventSample[i]->GetType()==1) s+= eventSample[i]->GetWeight();
   }
   node->SetSoverSB(s/(s+b));
   node->SetSeparationIndex(fSepType->GetSeparationIndex(s,b));

   //   if ( eventSample.size() > fMinSize  &&
   //        node->GetSoverSB() < fSoverSBUpperThreshold      &&
   //        node->GetSoverSB() > fSoverSBLowerThreshold  ) {
   if ( eventSample.size() > fMinSize &&
        node->GetSoverSB()*eventSample.size() > fMinSize     &&
        node->GetSoverSB()*eventSample.size() < eventSample.size()-fMinSize ) {

      Double_t separationGain;
      separationGain = this->TrainNode(eventSample, node);
      vector<TMVA::Event*> leftSample; leftSample.reserve(nevents);
      vector<TMVA::Event*> rightSample; rightSample.reserve(nevents);
      Double_t nRight=0, nLeft=0;
      for (UInt_t ie=0; ie< nevents ; ie++){
         if (node->GoesRight(eventSample[ie])){
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
         cout << "--- DecisionTree::TrainNode Error:  all events went to the same branch\n";
         cout << "---                         Hence new node == old node ... check\n";
         cout << "---                         left:" << leftSample.size()
              << " right:" << rightSample.size() << endl;
         cout << "--- this should never happen, please write a bug report to Helge.Voss@cern.ch"
              << endl;
         exit(1);
      }
    
      // continue building daughter nodes for the left and the right eventsample
      TMVA::DecisionTreeNode *rightNode = new TMVA::DecisionTreeNode(node);
      fNNodes++;
      fSumOfWeights += 1.0;
      rightNode->SetNEvents(nRight);
      TMVA::DecisionTreeNode *leftNode = new TMVA::DecisionTreeNode(node);
      fNNodes++;
      fSumOfWeights += 1.0;
      leftNode->SetNEvents(nLeft);
    
      node->SetNodeType(0);
      node->SetLeft(leftNode);
      node->SetRight(rightNode);
      this->BuildTree(rightSample, rightNode);
      this->BuildTree(leftSample,  leftNode );
   } else{ // it is a leaf node
      //    cout << "Found a leaf lode: " << eventSample.size() << " " <<
      //      node->GetSoverSB()*eventSample.size()  << endl;
      if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
      else node->SetNodeType(-1);
   }
  
   return fNNodes;
}

//_______________________________________________________________________
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

   Double_t separation = -1;
   Double_t cutValue=-999;
   Int_t mxVar=-1;
   Bool_t cutType=kTRUE;
   Double_t  nSelS, nSelB, nTotS, nTotB;

   TMVA::BinarySearchTree *sigBST=NULL;
   TMVA::BinarySearchTree *bkgBST=NULL;

   fUseSearchTree = kTRUE;
   if (eventSample.size() < 30000) fUseSearchTree = kFALSE;

   for (int ivar=0; ivar < fNvars; ivar++){
      (*xmin)[ivar]=(*xmax)[ivar]=eventSample[0]->GetData(ivar);
   }

   for (UInt_t i=1;i<eventSample.size();i++){
      for (Int_t ivar=0; ivar < fNvars; ivar++){
         if ((*xmin)[ivar]>eventSample[i]->GetData(ivar))(*xmin)[ivar]=eventSample[i]->GetData(ivar);
         if ((*xmax)[ivar]<eventSample[i]->GetData(ivar))(*xmax)[ivar]=eventSample[i]->GetData(ivar);
      }
   }

   for (int ivar=0; ivar < fNvars; ivar++){
      if (fUseSearchTree) {
         sigBST = new TMVA::BinarySearchTree();
         bkgBST = new TMVA::BinarySearchTree();
         vector<Int_t> theVars;
         theVars.push_back(ivar);
         sigBST->Fill( eventSample, theVars, 1 );
         bkgBST->Fill( eventSample, theVars, 0 );
      }

      // now optimist the cuts for each varable and find which one gives
      // the best separation at the current stage.
      // just scan the possible cut values for this variable
      Double_t istepSize =( (*xmax)[ivar] - (*xmin)[ivar] ) / Double_t(fNCuts);
      Int_t nCuts = fNCuts;
      vector<Double_t> cutValueTmp(nCuts);
      vector<Double_t> sep(nCuts);
      vector<Bool_t> cutTypeTmp(nCuts);

      for (Int_t istep=0; istep<fNCuts; istep++){
         cutValueTmp[istep]=(*xmin)[ivar]+(Float_t(istep)+0.5)*istepSize;
         if (fUseSearchTree){
            TMVA::Volume volume(cutValueTmp[istep], (*xmax)[ivar]);
            nSelS  = sigBST->SearchVolume( &volume );
            nSelB  = bkgBST->SearchVolume( &volume );

            nTotS  = sigBST->GetSumOfWeights();
            nTotB  = bkgBST->GetSumOfWeights();
         }else{
            nSelS=0; nSelB=0; nTotS=0; nTotB=0;
            for (UInt_t i=0; i<eventSample.size(); i++){
               if (eventSample[i]->GetType()==1){
                  nTotS+=eventSample[i]->GetWeight();
                  if (eventSample[i]->GetData(ivar) > cutValueTmp[istep]) nSelS+=eventSample[i]->GetWeight();
               }else if (eventSample[i]->GetType()==0){
                  nTotB+=eventSample[i]->GetWeight();
                  if (eventSample[i]->GetData(ivar) > cutValueTmp[istep]) nSelB+=eventSample[i]->GetWeight();
               }
            }
         }

         // now the separation is defined as the various indices (Gini, CorssEntropy, e.t.c)
         // calculated by the "SamplePurities" from the branches that would go to the
         // left or the right from this node if "these" cuts were used in the Node:
         // hereby: nSelS and nSelB would go to the right branch
         //        (nTotS - nSelS) + (nTotB - nSelB)  would go to the left branch;

         if (nSelS/nTotS > nSelB/nTotB) cutTypeTmp[istep]=kTRUE;
         else cutTypeTmp[istep]=kFALSE;

         sep[istep]= fSepType->GetSeparationGain(nSelS, nSelB, nTotS, nTotB);
      }

      //ich hab's versucht...aber das ist scheissee!!! Ich will ein INT!!!
      //    vector<Double_t>::iterator mxsep=max_element(sep.begin(),sep.end());
      Int_t pos = TMVA::Tools::GetIndexMaxElement(sep);

      //and now, choose the variable that gives the maximum separation
      if (separation < sep[pos]) {
         separation = sep[pos];
         cutValue=cutValueTmp[pos];
         cutType=cutTypeTmp[pos];
         mxVar = ivar;
      }
      if (fUseSearchTree) {
         if (sigBST!=NULL) delete sigBST;
         if (bkgBST!=NULL) delete bkgBST;
      }
   }

   node->SetSelector(mxVar);
   node->SetCutValue(cutValue);
   node->SetCutType(cutType);
   node->SetSeparationGain(separation);

   delete xmin;
   delete xmax;

   return separation;
}

//_______________________________________________________________________
Double_t TMVA::DecisionTree::CheckEvent(TMVA::Event* e)
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
   //  return Double_t ( current->GetNodeType() );
   return current->GetSoverSB();
}

//_______________________________________________________________________
Double_t  TMVA::DecisionTree::SamplePurity(vector<TMVA::Event*> eventSample)
{
  //calculates the purity S/(S+B) of a given event sample

  Double_t sumsig=0, sumbkg=0, sumtot=0;
  for (UInt_t ievt=0; ievt<eventSample.size(); ievt++) {
    if (eventSample[ievt]->GetType()==0) sumbkg+=eventSample[ievt]->GetWeight();
    if (eventSample[ievt]->GetType()==1) sumsig+=eventSample[ievt]->GetWeight();
    sumtot+=eventSample[ievt]->GetWeight();
  }
  //sanity check
  if (sumtot!= (sumsig+sumbkg)){
    cout << "--- TMVA::DecisionTree::Purity Error! sumtot != sumsig+sumbkg"
         << sumtot << " " << sumsig << " " << sumbkg << endl;
    exit(1);
  }
  if (sumtot>0) return sumsig/(sumsig + sumbkg);
  else return -1;
}
