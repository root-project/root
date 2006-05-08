// @(#)root/tmva $Id: TMVA_BinarySearchTree.cpp,v 1.10 2006/05/03 19:45:38 helgevoss Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : TMVA_BinarySearchTree                                                 *
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
 * $Id: TMVA_BinarySearchTree.cpp,v 1.10 2006/05/03 19:45:38 helgevoss Exp $        
 **********************************************************************************/
      
//_______________________________________________________________________
//                                                                      
// Binary search tree including volume search method                    
//                                                                      
//_______________________________________________________________________

#include "TMVA_BinarySearchTree.h"
#include "TMVA_Tools.h"
#include "TMatrixDBase.h"
#include "TObjString.h"
#include "Riostream.h"
#include <stdexcept>

#define DEBUG_TMVA_BinarySearchTree kFALSE

using std::vector;

ClassImp(TMVA_BinarySearchTree)

//_______________________________________________________________________
TMVA_BinarySearchTree::TMVA_BinarySearchTree( void ) 
  : TMVA_BinaryTree(),
    fDbgcount( 0 )
{}

//_______________________________________________________________________
TMVA_BinarySearchTree::~TMVA_BinarySearchTree( void ) 
{}

//_______________________________________________________________________
void TMVA_BinarySearchTree::Fill( TTree* theTree, vector<TString>* theVars, 
				  int& nevents, Int_t theType )
{
  fPeriode = (*theVars).size();
  // the event loop
  nevents = 0;
  Int_t n=theTree->GetEntries();
  for (Int_t ievt=0; ievt<n; ievt++) {
    // insert event into binary tree
    if (theType == -1 || (int)TMVA_Tools::GetValue( theTree, ievt, "type" ) == theType) {

      // create new event with pointer to event vector, and with a weight
      TMVA_Event *e=new TMVA_Event(theTree, ievt, theVars);
      this->Insert( e , kTRUE);
      nevents++;
    }
  } // end of event loop
  
  // sanity check
  if (nevents <= 0) {
    cerr << "--- TMVA_BinarySearchTree::BinarySearchTree::Fill( TTree* ... Error: number of events "
	 << "in tree is zero: " << nevents << endl;
    throw std::invalid_argument( "Abort" );
  }
}

//_______________________________________________________________________
void TMVA_BinarySearchTree::Fill( vector<TMVA_Event*> theTree, Int_t theType )
{
  Int_t n=theTree.size();

  for (Int_t ievt=0; ievt<n; ievt++) {
    // insert event into binary tree
    if (theType == -1 || theTree[ievt]->GetType() == theType) 
      this->Insert( theTree[ievt] , kFALSE);
  } // end of event loop
}

//_______________________________________________________________________
void TMVA_BinarySearchTree::Fill( vector<TMVA_Event*> theTree, vector<Int_t> theVars, 
				  int& nevents, Int_t theType )
{
  fPeriode = (theVars).size();
  // the event loop
  nevents = 0;
  Int_t n=theTree.size();

  for (Int_t ievt=0; ievt<n; ievt++) {
    // insert event into binary tree
    if (theType == -1 || theTree[ievt]->GetType() == theType) {
      // create new event with pointer to event vector, and with a weight
      TMVA_Event *e=new TMVA_Event();
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
    cout << "--- TMVA_BinarySearchTree::BinarySearchTree:Fill(std::vector<TMVA_Event*> ... "
	 << "Error: number of events "
	 << "that got  actually filled into the tree is zero: " << nevents << endl;
    exit(1);
  }
}

//_______________________________________________________________________
Double_t TMVA_BinarySearchTree::SearchVolume( TMVA_Volume* volume, 
					      std::vector<TMVA_Event*>* events )
{
  return SearchVolume( this->GetRoot(), volume, 0, events );
}

//_______________________________________________________________________
Double_t TMVA_BinarySearchTree::SearchVolume( TMVA_Node* t, TMVA_Volume* volume, Int_t depth, 
					      std::vector<TMVA_Event*>* events )
{
  if (t==NULL) return 0;  // Are we at an outer leave?

  Double_t count = 0.0;
  if (InVolume( t->GetData(), volume )) {
    count += t->GetData()->GetWeight();
    if (NULL != events) events->push_back( t->GetData() );
  }
  if (t->GetLeft()==NULL && t->GetRight()==NULL) return count;  // Are we at an outer leave?

  Bool_t tl, tr;
  Int_t  d = depth%this->GetPeriode();
  if (d !=t->GetSelector()) {
    cout << "Fatal error in TMVA_BinarySearchTree::SearchVolume: selector in Searchvolume " 
	 << d << " != " << "node "<< t->GetSelector() << " ==> abort" << endl;
    exit(1);
  }
  tl = (*(volume->Lower))[d] <  (t->GetData()->GetData(d));  // Should we descend left?
  tr = (*(volume->Upper))[d] >= (t->GetData()->GetData(d));  // Should we descend right?

  if (tl) count += SearchVolume( t->GetLeft(),  volume, (depth+1), events );
  if (tr) count += SearchVolume( t->GetRight(), volume, (depth+1), events );
  
  return count;
}

Bool_t TMVA_BinarySearchTree::InVolume( TMVA_Event* event, TMVA_Volume* volume ) const 
{
  Bool_t result = false;
  for (Int_t ivar=0; ivar< fPeriode; ivar++) {
    result = ( (*(volume->Lower))[ivar] <  ((event->GetData(ivar))) &&
	       (*(volume->Upper))[ivar] >= ((event->GetData(ivar))) );
    if (!result) break;
  }
  return result;
}
