// @(#)root/tmva $Id$    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : BinarySearchTree                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
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
 *                                                                                *
 **********************************************************************************/
      
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinarySearchTree                                                     //
//                                                                      //
// A simple Binary search tree including a volume search method         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if ROOT_VERSION_CODE >= 364802
#ifndef ROOT_TMathBase
#include "TMathBase.h"
#endif
#else
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#endif
#include "TMatrixDBase.h"
#include "TObjString.h"
#include "TTree.h"
#include "Riostream.h"
#include <stdexcept>
#include <stdlib.h>

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif

ClassImp(TMVA::BinarySearchTree)

//_______________________________________________________________________
TMVA::BinarySearchTree::BinarySearchTree( void ) :
   BinaryTree(),
   fPeriod      ( 1 ),
   fCurrentDepth( 0 ),
   fStatisticsIsValid( kFALSE ),
   fSumOfWeights( 0 ),
   fCanNormalize( kFALSE )
{
   // default constructor
   fLogger.SetSource( "BinarySearchTree" );
}

//_______________________________________________________________________
TMVA::BinarySearchTree::BinarySearchTree( const BinarySearchTree &b)
   : BinaryTree(), 
     fPeriod      ( b.fPeriod ),
     fCurrentDepth( 0 ),
     fStatisticsIsValid( kFALSE ),
     fSumOfWeights( b.fSumOfWeights ),
     fCanNormalize( kFALSE )
{
   // copy constructor that creates a true copy, i.e. a completely independent tree 
   fLogger.SetSource( "BinarySearchTree" );
   fLogger << kFATAL << " Copy constructor not implemented yet " << Endl;
}

//_______________________________________________________________________
TMVA::BinarySearchTree::~BinarySearchTree( void ) 
{
   // destructor

   for(std::vector< pair<Double_t, const TMVA::Event*> >::iterator pIt = fNormalizeTreeTable.begin();
       pIt != fNormalizeTreeTable.end(); pIt++) {
      delete pIt->second;
   }
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::Insert( const Event* event ) 
{
   // insert a new "event" in the binary tree
   fCurrentDepth=0;
   fStatisticsIsValid = kFALSE;

   if (this->GetRoot() == NULL) {           // If the list is empty...
      this->SetRoot( new BinarySearchTreeNode(event)); //Make the new node the root.
      // have to use "s" for start as "r" for "root" would be the same as "r" for "right"
      this->GetRoot()->SetPos('s'); 
      this->GetRoot()->SetDepth(0);
      fNNodes = 1;
      fSumOfWeights = event->GetWeight();
      ((BinarySearchTreeNode*)this->GetRoot())->SetSelector((UInt_t)0);
      this->SetPeriode(event->GetNVars());
   }
   else {
      // sanity check:
      if (event->GetNVars() != (UInt_t)this->GetPeriode()) {
         fLogger << kFATAL << "<Insert> event vector length != Periode specified in Binary Tree" << Endl
                 << "--- event size: " << event->GetNVars() << " Periode: " << this->GetPeriode() << Endl
                 << "--- and all this when trying filling the "<<fNNodes+1<<"th Node" << Endl;
      }
      // insert a new node at the propper position  
      this->Insert(event, this->GetRoot()); 
   }

   // normalise the tree to speed up searches
   if (fCanNormalize) fNormalizeTreeTable.push_back( make_pair(0.0,new const Event(*event)) );
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::Insert( const Event *event, 
                                     Node *node ) 
{
   // private internal function to insert a event (node) at the proper position
   fCurrentDepth++;
   fStatisticsIsValid = kFALSE;

   if (node->GoesLeft(*event)){    // If the adding item is less than the current node's data...
      if (node->GetLeft() != NULL){            // If there is a left node...
         // Add the new event to the left node
         this->Insert(event, node->GetLeft());
      } 
      else {                             // If there is not a left node...
         // Make the new node for the new event
         BinarySearchTreeNode* current = new BinarySearchTreeNode(event); 
         fNNodes++;
         fSumOfWeights += event->GetWeight();
         current->SetSelector(fCurrentDepth%((Int_t)event->GetNVars()));
         current->SetParent(node);          // Set the new node's previous node.
         current->SetPos('l');
         current->SetDepth( node->GetDepth() + 1 );
         node->SetLeft(current);            // Make it the left node of the current one.
      }  
   } 
   else if (node->GoesRight(*event)) { // If the adding item is less than or equal to the current node's data...
      if (node->GetRight() != NULL) {              // If there is a right node...
         // Add the new node to it.
         this->Insert(event, node->GetRight()); 
      } 
      else {                                 // If there is not a right node...
         // Make the new node.
         BinarySearchTreeNode* current = new BinarySearchTreeNode(event);   
         fNNodes++;
         fSumOfWeights += event->GetWeight();
         current->SetSelector(fCurrentDepth%((Int_t)event->GetNVars()));
         current->SetParent(node);              // Set the new node's previous node.
         current->SetPos('r');
         current->SetDepth( node->GetDepth() + 1 );
         node->SetRight(current);               // Make it the left node of the current one.
      }
   } 
   else fLogger << kFATAL << "<Insert> neither left nor right :)" << Endl;
}

//_______________________________________________________________________
TMVA::BinarySearchTreeNode* TMVA::BinarySearchTree::Search( Event* event ) const 
{ 
   //search the tree to find the node matching "event"
   return this->Search( event, this->GetRoot() );
}

//_______________________________________________________________________
TMVA::BinarySearchTreeNode* TMVA::BinarySearchTree::Search(Event* event, Node* node) const 
{ 
   // Private, recursive, function for searching.
   if (node != NULL) {               // If the node is not NULL...
      // If we have found the node...
      if (((BinarySearchTreeNode*)(node))->EqualsMe(*event)) 
         return (BinarySearchTreeNode*)node;                  // Return it
      if (node->GoesLeft(*event))      // If the node's data is greater than the search item...
         return this->Search(event, node->GetLeft());  //Search the left node.
      else                          //If the node's data is less than the search item...
         return this->Search(event, node->GetRight()); //Search the right node.
   }
   else return NULL; //If the node is NULL, return NULL.
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::GetSumOfWeights( void ) const
{
   //return the sum of event (node) weights
   if (fSumOfWeights <= 0) {
      fLogger << kWARNING << "you asked for the SumOfWeights, which is not filled yet"
              << " I call CalcStatistics which hopefully fixes things" 
              << Endl;
   }
   if (fSumOfWeights <= 0) fLogger << kFATAL << " Zero events in your Search Tree" <<Endl;

   return fSumOfWeights;
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::Fill( const MethodBase& callingMethod, TTree* theTree, Int_t theType )
{
   // create the search tree from the events in the DataSet
   // NOTE: the tree given as argument MUST NOT contain transformed variables, 
   // otherwise double transformation would occur
   Int_t nevents=0;
   if (fSumOfWeights != 0) {
      fLogger << kWARNING 
              << "You are filling a search three that is not empty.. "
              << " do you know what you are doing?"
              << Endl;
   }
   fPeriod = callingMethod.GetVarTransform().GetNVariables();

   // get the event: this implicitly takes into account variable transformation
   const Event& event = callingMethod.GetEvent(); 

   // insert all events into the tree
   for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

      callingMethod.ReadEvent(theTree, ievt);
      // insert event into binary tree
      if (theType==-1 || event.Type()==theType) {
         // create new event with pointer to event vector, and with a weight
         this->Insert( &event );
         nevents++;
         fSumOfWeights += event.GetWeight();
      }
   } // end of event loop

   // sanity check
   if (nevents == 0) {
      fLogger << kFATAL << "<Fill> number of events in tree is zero: " << nevents << Endl;
   }
   CalcStatistics();

   return fSumOfWeights;
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::Fill( std::vector<Event*> theTree, std::vector<Int_t> theVars, 
                                       Int_t theType )
{
   // create the search tree from the event collection 
   // using ONLY the variables specified in "theVars"
   fPeriod = (theVars).size();
   // the event loop
   Int_t nevents = 0;
   Int_t n=theTree.size();

   if (fSumOfWeights != 0) {
      fLogger << kWARNING 
              << "You are filling a search three that is not empty.. "
              << " do you know what you are doing?"
              << Endl;
   }

   for (Int_t ievt=0; ievt<n; ievt++) {
      // insert event into binary tree
      if (theType == -1 || theTree[ievt]->Type() == theType) {
         // create new event with pointer to event vector, and with a weight
         Event *e = new Event(*theTree[ievt]);
         this->Insert( e );
         nevents++;
         fSumOfWeights += e->GetWeight();
      }
   } // end of event loop

   // sanity check
   if (nevents == 0) {
      fLogger << kFATAL << "<Fill> number of events "
              << "that got filled into the tree is <= zero: " << nevents << Endl;
   }
   CalcStatistics();

   return fSumOfWeights;
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::Fill( std::vector<Event*> theTree, Int_t theType )
{
   // create the search tree from the events in a TTree
   // using ALL the variables specified included in the Event
   Int_t n=theTree.size();
  
   Int_t nevents = 0;
   if (fSumOfWeights != 0) {
      fLogger << kWARNING 
              << "You are filling a search three that is not empty.. "
              << " do you know what you are doing?"
              << Endl;
   }
   for (Int_t ievt=0; ievt<n; ievt++) {
      // insert event into binary tree
      if (theType == -1 || theTree[ievt]->Type() == theType) {
         this->Insert( theTree[ievt] );
         nevents++;
         fSumOfWeights += theTree[ievt]->GetWeight();
      }
   } // end of event loop
   CalcStatistics();

   return fSumOfWeights;
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::NormalizeTree ( std::vector< pair<Double_t, const TMVA::Event*> >::iterator leftBound, 
                                             std::vector< pair<Double_t, const TMVA::Event*> >::iterator rightBound, 
                                             UInt_t actDim )
{

   // normalises the binary-search tree to reduce the branch length and hence speed up the 
   // search procedure (in average)
   if (leftBound == rightBound) return;
   
   if (actDim == fPeriod)  actDim = 0;
   
   for (std::vector< pair<Double_t, const TMVA::Event*> >::iterator i=leftBound; i!=rightBound; i++) {
      i->first = i->second->GetVal( actDim );
   }
   
   std::sort( leftBound, rightBound );
   
   std::vector< pair<Double_t, const TMVA::Event*> >::iterator leftTemp  = leftBound;
   std::vector< pair<Double_t, const TMVA::Event*> >::iterator rightTemp = rightBound;
  
   // meet in the middle
   while (true) {
      rightTemp--; if (rightTemp == leftTemp ) break;
      leftTemp++;  if (leftTemp  == rightTemp) break;
   }
  
   std::vector< pair<Double_t, const TMVA::Event*> >::iterator mid     = leftTemp;
   std::vector< pair<Double_t, const TMVA::Event*> >::iterator midTemp = mid;

   //    int i=0;
   //    for(std::vector< pair<Double_t, const TMVA::Event*> >::iterator tmpIter = leftBound; tmpIter != rightBound; tmpIter++)
   //       cout << "...    " << i++ <<": " << mid->first << "   ->  " << mid->second->GetVal( actDim ) << endl;


   if (mid!=leftBound) midTemp--;

   while (mid != leftBound && mid->second->GetVal( actDim ) == midTemp->second->GetVal( actDim ))  {
      mid--; 
      midTemp--;
   }

   Insert( mid->second );

   //    Print(cout);
   //    cout << endl << endl;

   NormalizeTree( leftBound, mid, actDim+1 );
   mid++;
   //    Print(cout);
   //    cout << endl << endl;
   NormalizeTree( mid, rightBound, actDim+1 );


   return;  
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::NormalizeTree()
{
   // Normalisation of tree
   SetNormalize( kFALSE );
   Clear( NULL );
   this->SetRoot(NULL);
   NormalizeTree( fNormalizeTreeTable.begin(), fNormalizeTreeTable.end(), 0 ); 
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::Clear( Node* n )
{
   // clear nodes
   BinarySearchTreeNode* currentNode = (BinarySearchTreeNode*)(n == NULL ? this->GetRoot() : n);

   if (currentNode->GetLeft()  != 0) Clear( currentNode->GetLeft()  );
   if (currentNode->GetRight() != 0) Clear( currentNode->GetRight() );
   
   if (n != NULL) delete n;

   return;
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::SearchVolume( Volume* volume, 
                                               std::vector<const BinarySearchTreeNode*>* events )
{
   // search the whole tree and add up all weigths of events that 
   // lie within the given voluem
   return SearchVolume( this->GetRoot(), volume, 0, events );
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::SearchVolume( Node* t, Volume* volume, Int_t depth, 
                                               std::vector<const BinarySearchTreeNode*>* events )
{
   // recursively walk through the daughter nodes and add up all weigths of events that 
   // lie within the given volume

   if (t==NULL) return 0;  // Are we at an outer leave?
   
   BinarySearchTreeNode* st = (BinarySearchTreeNode*)t;

   Double_t count = 0.0;
   if (InVolume( st->GetEventV(), volume )) {
      count += st->GetWeight();
      if (NULL != events) events->push_back( st );
   }
   if (st->GetLeft()==NULL && st->GetRight()==NULL) {
      
      return count;  // Are we at an outer leave?
   }

   Bool_t tl, tr;
   Int_t  d = depth%this->GetPeriode();
   if (d != st->GetSelector()) {
      fLogger << kFATAL << "<SearchVolume> selector in Searchvolume " 
              << d << " != " << "node "<< st->GetSelector() << Endl;
   }
   tl = (*(volume->fLower))[d] <  st->GetEventV()[d];  // Should we descend left?
   tr = (*(volume->fUpper))[d] >= st->GetEventV()[d];  // Should we descend right?

   if (tl) count += SearchVolume( st->GetLeft(),  volume, (depth+1), events );
   if (tr) count += SearchVolume( st->GetRight(), volume, (depth+1), events );

   return count;
}

Bool_t TMVA::BinarySearchTree::InVolume(const std::vector<Float_t>& event, Volume* volume ) const 
{
   // test if the data points are in the given volume

   Bool_t result = false;
   for (UInt_t ivar=0; ivar< fPeriod; ivar++) {
      result = ( (*(volume->fLower))[ivar] <  event[ivar] &&
                 (*(volume->fUpper))[ivar] >= event[ivar] );
      if (!result) break;
   }
   return result;
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::CalcStatistics( Node* n )
{
   // calculate basic statistics (mean, rms for each variable)

   if (fStatisticsIsValid) return;

   BinarySearchTreeNode * currentNode = (BinarySearchTreeNode*)n;

   // default, start at the tree top, then descend recursively
   if (n == NULL) {
      fSumOfWeights = 0;
      for (Int_t sb=0; sb<2; sb++) {
         fNEventsW[sb]  = 0;
         fMeans[sb]     = std::vector<Float_t>(fPeriod);
         fRMS[sb]       = std::vector<Float_t>(fPeriod);
         fMin[sb]       = std::vector<Float_t>(fPeriod);
         fMax[sb]       = std::vector<Float_t>(fPeriod);
         fSum[sb]       = std::vector<Double_t>(fPeriod);
         fSumSq[sb]     = std::vector<Double_t>(fPeriod);
         for (UInt_t j=0; j<fPeriod; j++) {
            fMeans[sb][j] = fRMS[sb][j] = fSum[sb][j] = fSumSq[sb][j] = 0;
            fMin[sb][j] =  1.e25;
            fMax[sb][j] = -1.e25; 
         }
      }
      currentNode = (BinarySearchTreeNode*) this->GetRoot();
      if (currentNode == NULL) return; // no root-node
   }
      
   const std::vector<Float_t> & evtVec = currentNode->GetEventV();
   Double_t                     weight = currentNode->GetWeight();
   Int_t                        type   = currentNode->IsSignal() ? 0 : 1;
   fNEventsW[type] += weight;
   fSumOfWeights   += weight;

   for (UInt_t j=0; j<fPeriod; j++) {
      Float_t val = evtVec[j];
      fSum[type][j]   += val*weight;
      fSumSq[type][j] += val*val*weight;
      if (val < fMin[type][j]) fMin[type][j] = val; 
      if (val > fMax[type][j]) fMax[type][j] = val; 
   }

   if (currentNode->GetLeft()  != 0) CalcStatistics( currentNode->GetLeft()  );
   if (currentNode->GetRight() != 0) CalcStatistics( currentNode->GetRight() );

   if (n == NULL) { // i.e. the root node
      for (Int_t sb=0; sb<2; sb++) {
         for (UInt_t j=0; j<fPeriod; j++) {
            if (fNEventsW[sb] == 0) { fMeans[sb][j] = fRMS[sb][j] = 0; continue; }
            fMeans[sb][j] = fSum[sb][j]/fNEventsW[sb];
            fRMS[sb][j]   = TMath::Sqrt(fSumSq[sb][j]/fNEventsW[sb] - fMeans[sb][j]*fMeans[sb][j]);
         }
      }
      fStatisticsIsValid = kTRUE;
   }
   
   return;
}

Int_t TMVA::BinarySearchTree::SearchVolumeWithMaxLimit( Volume *volume, std::vector<const BinarySearchTreeNode*>* events,
                                                        Int_t max_points )
{
   // recursively walk through the daughter nodes and add up all weigths of events that 
   // lie within the given volume a maximum number of events can be given
   if (this->GetRoot() == NULL) return 0;  // Are we at an outer leave?

   std::queue< pair< const BinarySearchTreeNode*, Int_t > > queue;
   std::pair< const BinarySearchTreeNode*, Int_t > st = make_pair( (const BinarySearchTreeNode*)this->GetRoot(), 0 );
   //   std::pair< const BinarySearchTreeNode*, Int_t > st = make_pair( (BinarySearchTreeNode*)this->GetRoot(), 0 );
   queue.push( st );

   Int_t count = 0;
   
   while ( !queue.empty() ) {
      st = queue.front(); queue.pop();
      
      if (count == max_points)
         return count;

      if (InVolume( st.first->GetEventV(), volume )) {
         count++;
         if (NULL != events) events->push_back( st.first );
      }

      Bool_t tl, tr;
      Int_t d = st.second;
      if ( d == Int_t(this->GetPeriode()) ) d = 0;

      if (d != st.first->GetSelector()) {
         fLogger << kFATAL << "<SearchVolume> selector in Searchvolume "
                 << d << " != " << "node "<< st.first->GetSelector() << Endl;
      }

      tl = (*(volume->fLower))[d] <  st.first->GetEventV()[d] && st.first->GetLeft()  != NULL;  // Should we descend left?
      tr = (*(volume->fUpper))[d] >= st.first->GetEventV()[d] && st.first->GetRight() != NULL;  // Should we descend right?

      if (tl) queue.push( make_pair( (const BinarySearchTreeNode*)st.first->GetLeft(), d+1 ) );
      if (tr) queue.push( make_pair( (const BinarySearchTreeNode*)st.first->GetRight(), d+1 ) );
   }

   return count;
}

