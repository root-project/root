// @(#)root/tmva $Id: BinarySearchTree.cxx,v 1.24 2006/11/17 14:59:23 stelzer Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::BinarySearchTree                                                *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: BinarySearchTree.cxx,v 1.24 2006/11/17 14:59:23 stelzer Exp $        
 **********************************************************************************/
      
//_______________________________________________________________________
//                                                                      
// a simple Binary search tree including volume search method                    
//                                                                      
//_______________________________________________________________________

#include "TMatrixDBase.h"
#include "TObjString.h"
#include "TTree.h"
#include "Riostream.h"
#include <stdexcept>

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
   ;

using std::vector;

//_______________________________________________________________________
TMVA::BinarySearchTree::BinarySearchTree( void ) :
   TMVA::BinaryTree(),
   fSumOfWeights( 0 ),
   fPeriode ( 1 ),
   fCurrentDepth( 0 )
{
  // default constructor
   fLogger.SetSource( "BinarySearchTree" );
}

//_______________________________________________________________________
TMVA::BinarySearchTree::BinarySearchTree( const BinarySearchTree &b)
   : TMVA::BinaryTree(), 
     fSumOfWeights( b.fSumOfWeights ),
     fPeriode ( b.fPeriode ),
     fCurrentDepth( 0 )
{
   // copy constructor that creates a true copy, i.e. a completely independent tree 
   fLogger.SetSource( "BinarySearchTree" );
   fLogger << kFATAL << " Copy constructor not implemented yet " << Endl;

}

//_______________________________________________________________________
TMVA::BinarySearchTree::~BinarySearchTree( void ) 
{
   // destructor
}


//_______________________________________________________________________
void TMVA::BinarySearchTree::Insert( TMVA::Event* event, Bool_t eventOwnership ) 
{
   //insert a new "event" in the binary tree
   //   set "eventOwnershipt" to kTRUE if the event should be owned (deleted) by
   //   the tree
   fCurrentDepth=0;

   if (this->GetRoot() == NULL) {           //If the list is empty...
      this->SetRoot( new TMVA::BinarySearchTreeNode(event, eventOwnership)); //Make the new node the root.
      // have to use "s" for start as "r" for "root" would be the same as "r" for "right"
      this->GetRoot()->SetPos('s'); 
      this->GetRoot()->SetDepth(0);
      fNNodes++;
      fSumOfWeights+=event->GetWeight();
      ((BinarySearchTreeNode*)this->GetRoot())->SetSelector((UInt_t)0);
      this->SetPeriode(event->GetNVars());
   }
   else {
      //sanity check:
      if (event->GetNVars() != (UInt_t)this->GetPeriode()) {
         fLogger << kFATAL << "<Insert> event vector length != Periode specified in Binary Tree" << Endl
                 << "--- event size: " << event->GetNVars() << " Periode: " << this->GetPeriode() << Endl
                 << "--- and all this when trying filling the "<<fNNodes+1<<"th Node" << Endl;
      }
      // insert a new node at the propper position  
      this->Insert(event, this->GetRoot(), eventOwnership); 
   }
}

//_______________________________________________________________________
void TMVA::BinarySearchTree::Insert( TMVA::Event *event, 
                                     TMVA::Node *node, Bool_t eventOwnership ) 
{
   //private internal fuction to insert a event (node) at the proper position
   fCurrentDepth++;
   if (node->GoesLeft(*event)){    // If the adding item is less than the current node's data...
      if (node->GetLeft() != NULL){            // If there is a left node...
         // Add the new node to it.
         this->Insert(event, node->GetLeft(), eventOwnership);
      } 
      else {                             // If there is not a left node...
         // Make the new node.
         TMVA::BinarySearchTreeNode* current = 
            new TMVA::BinarySearchTreeNode(event, eventOwnership); 
         fNNodes++;
         fSumOfWeights+=event->GetWeight();
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
         this->Insert(event, node->GetRight(), eventOwnership); 
      } 
      else {                                 // If there is not a right node...
         // Make the new node.
         TMVA::BinarySearchTreeNode* current = 
            new TMVA::BinarySearchTreeNode(event, eventOwnership);   
         fNNodes++;
         fSumOfWeights+=event->GetWeight();
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
TMVA::BinarySearchTreeNode* TMVA::BinarySearchTree::Search( TMVA::Event* event ) const 
{ 
   //search the tree to find the node matching "event"
   return this->Search( event, this->GetRoot() );
}

//_______________________________________________________________________
TMVA::BinarySearchTreeNode* TMVA::BinarySearchTree::Search(TMVA::Event* event, TMVA::Node* node) const 
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
   return fSumOfWeights;
}

//_______________________________________________________________________
Int_t TMVA::BinarySearchTree::Fill( const DataSet& ds, TTree* theTree, Int_t theType, 
                                    Types::EPreprocessingMethod corr, Types::ESBType type  )
{
   // create the search tree from the events in the DataSet
   Int_t nevents=0;
   fPeriode = ds.GetNVariables();
   // the event loop
   Int_t n=theTree->GetEntries();
   for (Int_t ievt=0; ievt<n; ievt++) {
      ds.ReadEvent(theTree,ievt,corr,type);
      // insert event into binary tree
      if (theType==-1 || ds.Event().Type()==theType) {
         // create new event with pointer to event vector, and with a weight
         this->Insert( new TMVA::Event(ds.Event()), kTRUE );
         nevents++;
      }
   } // end of event loop

   // sanity check
   if (nevents <= 0) {
      fLogger << kFATAL << "<Fill> number of events in tree is zero: " << nevents << Endl;
   }
   return nevents;
}

//_______________________________________________________________________
Int_t TMVA::BinarySearchTree::Fill( vector<TMVA::Event*> theTree, vector<Int_t> theVars, 
                                    Int_t theType )
{
   // create the search tree from the event collection 
   // using ONLY the variables specified in "theVars"
   fPeriode = (theVars).size();
   // the event loop
   Int_t nevents = 0;
   Int_t n=theTree.size();

   for (Int_t ievt=0; ievt<n; ievt++) {
      // insert event into binary tree
      if (theType == -1 || theTree[ievt]->Type() == theType) {
         // create new event with pointer to event vector, and with a weight
         TMVA::Event *e = new TMVA::Event(*theTree[ievt]);
         this->Insert( e , kTRUE);
         nevents++;
      }
   } // end of event loop

   // sanity check
   if (nevents <= 0) {
      fLogger << kFATAL << "<Fill> number of events "
              << "that got filled into the tree is <= zero: " << nevents << Endl;
   }
   return nevents;
}

//_______________________________________________________________________
Int_t TMVA::BinarySearchTree::Fill( vector<TMVA::Event*> theTree, Int_t theType )
{
   // create the search tree from the events in a TTree
   // using ALL the variables specified included in the Event
   Int_t n=theTree.size();
  
   Int_t nevents = 0;
   for (Int_t ievt=0; ievt<n; ievt++) {
      // insert event into binary tree
      if (theType == -1 || theTree[ievt]->Type() == theType) {
         this->Insert( theTree[ievt] , kFALSE);
         nevents++;
      }
   } // end of event loop
   return nevents;
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::SearchVolume( TMVA::Volume* volume, 
                                               std::vector<TMVA::Event*>* events )
{
   //search the whole tree and add up all weigths of events that 
   // lie within the given voluem
   return SearchVolume( this->GetRoot(), volume, 0, events );
}

//_______________________________________________________________________
Double_t TMVA::BinarySearchTree::SearchVolume( TMVA::Node* t, TMVA::Volume* volume, Int_t depth, 
                                               std::vector<TMVA::Event*>* events )
{
   // recursively walk through the daughter nodes and add up all weigths of events that 
   // lie within the given volume
   
   BinarySearchTreeNode* st = (BinarySearchTreeNode*)t;

   if (st==NULL) return 0;  // Are we at an outer leave?

   Double_t count = 0.0;
   if (InVolume( st->GetEvent(), volume )) {
      count += st->GetEvent()->GetWeight();
      if (NULL != events) events->push_back( st->GetEvent() );
   }
   if (st->GetLeft()==NULL && st->GetRight()==NULL) return count;  // Are we at an outer leave?

   Bool_t tl, tr;
   Int_t  d = depth%this->GetPeriode();
   if (d != st->GetSelector()) {
      fLogger << kFATAL << "<SearchVolume> selector in Searchvolume " 
              << d << " != " << "node "<< st->GetSelector() << Endl;
   }
   tl = (*(volume->fLower))[d] <  (st->GetEvent()->GetVal(d));  // Should we descend left?
   tr = (*(volume->fUpper))[d] >= (st->GetEvent()->GetVal(d));  // Should we descend right?

   if (tl) count += SearchVolume( st->GetLeft(),  volume, (depth+1), events );
   if (tr) count += SearchVolume( st->GetRight(), volume, (depth+1), events );

   return count;
}

Bool_t TMVA::BinarySearchTree::InVolume( TMVA::Event* event, TMVA::Volume* volume ) const 
{
   // test if the data points are in the given volume

   Bool_t result = false;
   for (UInt_t ivar=0; ivar< fPeriode; ivar++) {
      result = ( (*(volume->fLower))[ivar] <  ((event->GetVal(ivar))) &&
                 (*(volume->fUpper))[ivar] >= ((event->GetVal(ivar))) );
      if (!result) break;
   }
   return result;
}
