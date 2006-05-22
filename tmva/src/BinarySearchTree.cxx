// @(#)root/tmva $Id: BinarySearchTree.cxx,v 1.5 2006/05/22 08:04:38 andreas.hoecker Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : TMVA::BinarySearchTree                                                *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
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
 * $Id: BinarySearchTree.cxx,v 1.5 2006/05/22 08:04:38 andreas.hoecker Exp $        
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
#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif

ClassImp(TMVA::BinarySearchTree)

using std::vector;

//_______________________________________________________________________
TMVA::BinarySearchTree::BinarySearchTree( void ) 
{}

//_______________________________________________________________________
TMVA::BinarySearchTree::~BinarySearchTree( void ) 
{}

//_______________________________________________________________________
Int_t TMVA::BinarySearchTree::Fill( TTree* theTree, vector<TString>* theVars, 
				    Int_t theType )
{
  // create the search tree from the events in a TTree 
  // using the variables specified in "theVars"
  Int_t nevents=0;
  fPeriode = (*theVars).size();
  // the event loop
  Int_t n=theTree->GetEntries();
  for (Int_t ievt=0; ievt<n; ievt++) {
    // insert event into binary tree
    if (theType == -1 || (int)TMVA::Tools::GetValue( theTree, ievt, "type" ) == theType) {

      // create new event with pointer to event vector, and with a weight
      TMVA::Event *e=new TMVA::Event(theTree, ievt, theVars);
      this->Insert( e , kTRUE);
      nevents++;
    }
  } // end of event loop

  // sanity check
  if (nevents <= 0) {
    cerr << "--- TMVA::BinarySearchTree::BinarySearchTree::Fill( TTree* ... Error: number of events "
         << "in tree is zero: " << nevents << endl;
    throw std::invalid_argument( "Abort" );
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
    if (theType == -1 || theTree[ievt]->GetType() == theType) {
      // create new event with pointer to event vector, and with a weight
      TMVA::Event *e=new TMVA::Event();
      for (Int_t j=0; j<fPeriode; j++){
         e->Insert(theTree[ievt]->GetData(theVars[j]) );
      }
      e->SetWeight(theTree[ievt]->GetWeight() );
      this->Insert( e , kTRUE);
      nevents++;
    }
  } // end of event loop

  // sanity check
  if (nevents <= 0) {
    cout << "--- TMVA::BinarySearchTree::BinarySearchTree:Fill(std::vector<TMVA::Event*> ... "
         << "Error: number of events "
         << "that got  actually filled into the tree is zero: " << nevents << endl;
    exit(1);
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
    if (theType == -1 || theTree[ievt]->GetType() == theType) {
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
  // lie within the given voluem
  if (t==NULL) return 0;  // Are we at an outer leave?

  Double_t count = 0.0;
  if (InVolume( t->GetData(), volume )) {
    count += t->GetData()->GetWeight();
    if (NULL != events) events->push_back( t->GetData() );
  }
  if (t->GetLeft()==NULL && t->GetRight()==NULL) return count;  // Are we at an outer leave?

  Bool_t tl, tr;
  Int_t  d = depth%this->GetPeriode();
  if (d != t->GetSelector()) {
    cout << "Fatal error in TMVA::BinarySearchTree::SearchVolume: selector in Searchvolume " 
         << d << " != " << "node "<< t->GetSelector() << " ==> abort" << endl;
    exit(1);
  }
  tl = (*(volume->fLower))[d] <  (t->GetData()->GetData(d));  // Should we descend left?
  tr = (*(volume->fUpper))[d] >= (t->GetData()->GetData(d));  // Should we descend right?

  if (tl) count += SearchVolume( t->GetLeft(),  volume, (depth+1), events );
  if (tr) count += SearchVolume( t->GetRight(), volume, (depth+1), events );

  return count;
}

Bool_t TMVA::BinarySearchTree::InVolume( TMVA::Event* event, TMVA::Volume* volume ) const 
{
  Bool_t result = false;
  for (Int_t ivar=0; ivar< fPeriode; ivar++) {
    result = ( (*(volume->fLower))[ivar] <  ((event->GetData(ivar))) &&
             (*(volume->fUpper))[ivar] >= ((event->GetData(ivar))) );
    if (!result) break;
  }
  return result;
}
