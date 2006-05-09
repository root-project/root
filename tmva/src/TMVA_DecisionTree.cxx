// @(#)root/tmva $Id: TMVA_DecisionTree.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_DecisionTree                                                     *
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
 * $Id: TMVA_DecisionTree.cxx,v 1.1 2006/05/08 12:46:31 brun Exp $
 **********************************************************************************/

//_______________________________________________________________________
//
// Implementation of a Decision Tree
//
//_______________________________________________________________________

#include <iostream>
#include <algorithm>

#include "TVirtualFitter.h"

#include "TMVA_DecisionTree.h"
#include "TMVA_DecisionTreeNode.h"
#include "TMVA_BinarySearchTree.h"

#include "TMVA_Tools.h"

#include "TMVA_GiniIndex.h"
#include "TMVA_CrossEntropy.h"
#include "TMVA_MisClassificationError.h"
#include "TMVA_SdivSqrtSplusB.h"

using std::vector;

ClassImp(TMVA_DecisionTree)

//_______________________________________________________________________
TMVA_DecisionTree::TMVA_DecisionTree( void )
{
  fNvars        = 0;
  fSepType      = new TMVA_GiniIndex();
  fNCuts        = -1;

//   fSoverSBUpperThreshold = 0;
//   fSoverSBLowerThreshold = 0;
  fMinSize               = 0;
  fMinSepGain   = 0.0003;

}

//_______________________________________________________________________
TMVA_DecisionTree::TMVA_DecisionTree( TMVA_SeparationBase *sepType,Int_t minSize, Double_t mnsep,
                                      Int_t nCuts)
{
  fNvars        = 0;
  fSepType      = sepType;
  fNCuts        = nCuts;

//   fSoverSBUpperThreshold = mxp;
//   fSoverSBLowerThreshold = mnp;
  fMinSize               = minSize;
  fMinSepGain            = mnsep;
}

//_______________________________________________________________________
TMVA_DecisionTree::~TMVA_DecisionTree( void )
{}

//_______________________________________________________________________
void TMVA_DecisionTree::BuildTree( vector<TMVA_Event*> & eventSample,
                                   TMVA_DecisionTreeNode *node )
{
  if (node==NULL) {
    //start with the root node
    node = new TMVA_DecisionTreeNode();
    fNNodes++;
    fSumOfWeights+=1.;
    this->SetRoot(node);
  }

  UInt_t nevents = eventSample.size();
  if (nevents > 0 ) fNvars = eventSample[0]->GetEventSize();
  else{
    cout << "--- TMVA_DecisionTree::BuildTree:  Error, Eventsample Size == 0 " <<endl;
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

    this->TrainNode(eventSample, node);
    if (node->GetSeparationGain() > fMinSepGain) {
      vector<TMVA_Event*> leftSample; leftSample.reserve(nevents);
      vector<TMVA_Event*> rightSample; rightSample.reserve(nevents);
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
      TMVA_DecisionTreeNode *rightNode = new TMVA_DecisionTreeNode(node);
      fNNodes++;
      fSumOfWeights += 1.0;
      rightNode->SetNEvents(nRight);
      TMVA_DecisionTreeNode *leftNode = new TMVA_DecisionTreeNode(node);
      fNNodes++;
      fSumOfWeights += 1.0;
      leftNode->SetNEvents(nLeft);

      node->SetNodeType(0);
      node->SetLeft(leftNode);
      node->SetRight(rightNode);
      this->BuildTree(rightSample, rightNode);
      this->BuildTree(leftSample,  leftNode );
    } else { // it is a leaf node
      //      cout << "Found a leaf node: " << node->GetSeparationGain() << endl;

      if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
      else node->SetNodeType(-1);
    }
  } else{ // it is a leaf node
    //    cout << "Found a leaf lode: " << eventSample.size() << " " <<
    //      node->GetSoverSB()*eventSample.size()  << endl;
    if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
    else node->SetNodeType(-1);
  }

  return;
}

//_______________________________________________________________________
Double_t TMVA_DecisionTree::TrainNode(vector<TMVA_Event*> & eventSample,
                                  TMVA_DecisionTreeNode *node)
{
  Int_t dummy;
  // at each node, ONE of the variables is choosen, which gives the best
  // separation between sign and bkg on the sample which enters the Node.
  // --> first fill a binary search tree for "each" variable in order to
  // quickly find which one offers the best separation.

  TMVA_BinarySearchTree *sigBST=NULL;
  TMVA_BinarySearchTree *bkgBST=NULL;

  vector<Double_t> *xmin  = new vector<Double_t>( fNvars );
  vector<Double_t> *xmax  = new vector<Double_t>( fNvars );

  Double_t separation = -1;
  Double_t cutMin=-999, cutMax=-999;
  Int_t mxVar=-1;
  Bool_t cutType=kTRUE;
  Double_t  nSelS, nSelB, nTotS, nTotB;

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
      sigBST = new TMVA_BinarySearchTree();
      bkgBST = new TMVA_BinarySearchTree();
      vector<Int_t> theVars;
      theVars.push_back(ivar);
      sigBST->Fill( eventSample,theVars, dummy, 1 );
      bkgBST->Fill( eventSample, theVars, dummy, 0 );
    }

    // now optimist the cuts for each varable and find which one gives
    // the best separation at the current stage.
    // just scan the possible cut values for this variable
    Double_t istepSize =( (*xmax)[ivar] - (*xmin)[ivar] ) / Double_t(fNCuts);
    Int_t nCuts = fNCuts;
    vector<Double_t> cutMinTmp(nCuts), cutMaxTmp(nCuts);
    vector<Double_t> sep(nCuts);
    vector<Bool_t> cutTypeTmp(nCuts);

    for (Int_t istep=0; istep<fNCuts; istep++){
      cutMinTmp[istep]=(*xmin)[ivar]+(Float_t(istep)+0.5)*istepSize;
      cutMaxTmp[istep]=(*xmax)[ivar];
      if (fUseSearchTree){
        TMVA_Volume volume(cutMinTmp[istep], cutMaxTmp[istep]);
        nSelS  = sigBST->SearchVolume( &volume );
        nSelB  = bkgBST->SearchVolume( &volume );

        nTotS  = sigBST->GetSumOfWeights();
        nTotB  = bkgBST->GetSumOfWeights();
      }else{
        nSelS=0; nSelB=0; nTotS=0; nTotB=0;
        for (UInt_t i=0; i<eventSample.size(); i++){
          if (eventSample[i]->GetType()==1){
            nTotS+=eventSample[i]->GetWeight();
            if (eventSample[i]->GetData(ivar) > cutMinTmp[istep]) nSelS+=eventSample[i]->GetWeight();
          }else if (eventSample[i]->GetType()==0){
            nTotB+=eventSample[i]->GetWeight();
            if (eventSample[i]->GetData(ivar) > cutMinTmp[istep]) nSelB+=eventSample[i]->GetWeight();
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
    Int_t pos = TMVA_Tools::GetIndexMaxElement(sep);

    //and now, choose the variable that gives the maximum separation
    if (separation < sep[pos]) {
      separation = sep[pos];
      cutMin=cutMinTmp[pos];
      cutMax=cutMaxTmp[pos];
      cutType=cutTypeTmp[pos];
      mxVar = ivar;
    }
    if (fUseSearchTree) {
      if (sigBST!=NULL) delete sigBST;
      if (bkgBST!=NULL) delete bkgBST;
    }
  }

  node->SetSelector(mxVar);
  node->SetCutMin(cutMin);
  node->SetCutMax(cutMax);
  node->SetCutType(cutType);
  node->SetSeparationGain(separation);

  delete xmin;
  delete xmax;
  return separation;
}

//_______________________________________________________________________
Int_t TMVA_DecisionTree::CheckEvent(TMVA_Event* e)
{
  TMVA_DecisionTreeNode *current = (TMVA_DecisionTreeNode*)this->GetRoot();

  while(current->GetNodeType() == 0){ //intermediate node
    if (current->GoesRight(e))
        current=(TMVA_DecisionTreeNode*)current->GetRight();
    else current=(TMVA_DecisionTreeNode*)current->GetLeft();
  }
  return current->GetNodeType();
}

//_______________________________________________________________________
Double_t  TMVA_DecisionTree::SamplePurity(vector<TMVA_Event*> eventSample)
{
  Double_t sumsig=0, sumbkg=0, sumtot=0;
  for (UInt_t ievt=0; ievt<eventSample.size(); ievt++) {
    if (eventSample[ievt]->GetType()==0) sumbkg+=eventSample[ievt]->GetWeight();
    if (eventSample[ievt]->GetType()==1) sumsig+=eventSample[ievt]->GetWeight();
    sumtot+=eventSample[ievt]->GetWeight();
  }
  //sanity check
  if (sumtot!= (sumsig+sumbkg)){
    cout << "--- TMVA_DecisionTree::Purity Error! sumtot != sumsig+sumbkg"
         << sumtot << " " << sumsig << " " << sumbkg << endl;
    exit(1);
  }
  if (sumtot>0) return sumsig/(sumsig + sumbkg);
  else return -1;
}


