// @(#)root/tmva $Id: DecisionTree.cxx,v 1.26 2006/09/28 10:50:16 helgevoss Exp $
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
 * $Id: DecisionTree.cxx,v 1.26 2006/09/28 10:50:16 helgevoss Exp $
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

//_______________________________________________________________________
TMVA::DecisionTree::DecisionTree( void )
   : fNvars      (0),
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
   if (nevents > 0 ) {
      fNvars = eventSample[0]->GetNVars();
      fVariableImportance.resize(fNvars);
   }
   else{
      cout << "--- TMVA::DecisionTree::BuildTree:  Error, Eventsample Size == 0 " <<endl;
      exit(1);
   }

   Double_t s=0, b=0;
   for (UInt_t i=0; i<eventSample.size(); i++){
      if (eventSample[i]->IsSignal())
         s += eventSample[i]->GetWeight();
      else
         b += eventSample[i]->GetWeight();
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
      if (separationGain == 0) {//we could not gain anything, e.g. all events are in one bin, 
         // hence no cut can actually do anything. Happens for Integer Variables
         // Hence natuarlly the current node is a leaf node
         if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
         else node->SetNodeType(-1);
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
      }
   } else{ // it is a leaf node
      //    cout << "Found a leaf lode: " << eventSample.size() << " " <<
      //      node->GetSoverSB()*eventSample.size()  << endl;
      if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
      else node->SetNodeType(-1);
   }
   
   return fNNodes;
}


//_______________________________________________________________________
void TMVA::DecisionTree::PruneTree(DecisionTreeNode *node){
  // recursive prunig of nodes:  if internal node, then prune 
   DecisionTreeNode *l = (DecisionTreeNode*)node->GetLeft();
   DecisionTreeNode *r = (DecisionTreeNode*)node->GetRight();
   if (node->GetNodeType() == 0){
      this->PruneTree(l);
      this->PruneTree(r);
//       cout << "SubTreeError="<<this->GetSubTreeError(node)
//            << " and NodeError="<<this->GetNodeError(node) <<endl;
      if (this->GetSubTreeError(node) >= this->GetNodeError(node) ){ 
         //  ||	l->GetNEvents() < fMinSize || r->GetNEvents() < fMinSize ){
         //       cout << "Prune! Node error: = " << this->GetNodeError(node)
         // 	   << " Subtree error= " << this->GetSubTreeError(node) <<endl;
         this->PruneNode(node);
      }
   } 
}
//_______________________________________________________________________
void TMVA::DecisionTree::PruneNode(DecisionTreeNode *node){
   // prune away the subtree below the node if the expected statistical error at this node is
   // smaller than the expected statistical error of the subtree
   
   DecisionTreeNode *l = (DecisionTreeNode*)node->GetLeft();
   DecisionTreeNode *r = (DecisionTreeNode*)node->GetRight();
   
   if (l->GetNodeType() !=0  && r->GetNodeType() !=0) {
      // delete both daughters and make the node a leaf
      //     cout << " prune node with daughter types (nev): "<< l->GetNodeType()
      // 	 << " ("
      // 	 <<l->GetNEvents() << ")     "
      // 	 << r->GetNodeType() << " (" << r->GetNEvents() << ") " << endl;
      
      node->SetRight(NULL);
      node->SetLeft(NULL);
      if (node->GetSoverSB() > 0.5) node->SetNodeType(1);
      else node->SetNodeType(-1);
      delete l;
    delete r;
   }else if (l->GetNodeType() !=0  && r->GetNodeType() ==0) {
      //      DecisionTreeNode *oldNode = node;
      node=l;
      //      delete oldNode;   //give s segmentation fault if I try to clean up.. why ??
   }else if (l->GetNodeType() ==0  && r->GetNodeType() !=0) {
      //      DecisionTreeNode *oldNode = node;
      node=r;
      //      delete oldNode;
   }else{
      //    cout << "I do not know how to merge tow internal nodes.. just skip" <<endl;
   }
}



//_______________________________________________________________________
Double_t TMVA::DecisionTree::GetNodeError(DecisionTreeNode *node){
   // calculate an UPPER limit on the error made by the classification done
   // by this node. If the S/S+B of the node is f, then according to the
   // training sample, the error rate (fraction of misclassified events by
   // this node) is (1-f)
   // now as f has a statistical error according to the binomial distribution
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


   //   cout << errorRate << "  " <<endl;   
   
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
   UInt_t Nevents = eventSample.size();

   //find min and max value of the variables in the sample
   for (int ivar=0; ivar < fNvars; ivar++){
      (*xmin)[ivar]=(*xmax)[ivar]=eventSample[0]->GetVal(ivar);
   }
   for (UInt_t iev=1;iev<Nevents;iev++){
      for (Int_t ivar=0; ivar < fNvars; ivar++){
         Double_t eventData = eventSample[iev]->GetVal(ivar); 
         if ((*xmin)[ivar]>eventData)(*xmin)[ivar]=eventData;
         if ((*xmax)[ivar]<eventData)(*xmax)[ivar]=eventData;
      }
   }

//    for (int ivar=0; ivar < fNvars; ivar++){
//       cout << "xmin["<<ivar<<"]= " << (*xmin)[ivar]<< "   " 
//            << "xmax["<<ivar<<"]= " << (*xmax)[ivar]<< endl;
//    }
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
      for (UInt_t iev=0; iev<Nevents; iev++){

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
   for (UInt_t iev=0; iev<Nevents; iev++){
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

//          cout << "   nSelS["<<ivar<<"]["<<icut<<"]= " <<nSelS[ivar][icut] 
//               << "   nSelB["<<ivar<<"]["<<icut<<"]= " <<nSelB[ivar][icut]
//               <<"    Sep: " << sepTmp<<endl;

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
 
//    cout << "the winner is " <<  "nSelS["<<mxVar<<"]["<<cutIndex<<"]= " <<nSelS[mxVar][cutIndex] << "   nSelB["<<mxVar<<"]["<<cutIndex<<" ]= "<<nSelB[mxVar][cutIndex]<<"   Sep: " << separationGain<<endl;
//    cout << " nTotS= " << nTotS;
//    cout << " nSelS= " << nSelS[mxVar][cutIndex];  
//    cout << " nTotB= " << nTotB;
//    cout << " nSelB= " << nSelB[mxVar][cutIndex];  
//    cout << endl;

   delete xmin;
   delete xmax;

   return separationGain;
}


#else 

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
                                                        background_counts[var][cut],				                                        nTotS, nTotB);
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
      cout << "--- TMVA::DecisionTree::Purity Error! sumtot != sumsig+sumbkg"
           << sumtot << " " << sumsig << " " << sumbkg << endl;
      exit(1);
   }
   if (sumtot>0) return sumsig/(sumsig + sumbkg);
   else return -1;
}

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

Double_t  TMVA::DecisionTree::GetVariableImportance(Int_t ivar)
{
   vector<Double_t> relativeImportance = this->GetVariableImportance();
   if (ivar >= 0 && ivar < fNvars) return relativeImportance[ivar];
   else {
      cout << "--- TMVA::DecisionTree::GetVariableImportance(ivar)  ERROR!!" <<endl;
      cout << "---                     ivar = " << ivar << " is out of range " <<endl;
      exit(1);
   }
}
