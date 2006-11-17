// @(#)root/tmva $Id: BinarySearchTree.h,v 1.19 2006/11/17 14:59:23 stelzer Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : BinarySearchTree                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      BinarySearchTree incl. volume Search method                               *
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
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_BinarySearchTree
#define ROOT_TMVA_BinarySearchTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinarySearchTree                                                     //
//                                                                      //
// A simple Binary search tree including volume search method           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Riostream.h"
#include <vector>
#include "time.h"

#include "TTree.h"

#ifndef ROOT_TMVA_Volume
#include "TMVA/Volume.h"
#endif
#ifndef ROOT_TMVA_BinaryTree
#include "TMVA/BinaryTree.h"
#endif
#ifndef ROOT_TMVA_BinarySearchTreeNode
#include "TMVA/BinarySearchTreeNode.h"
#endif


class TString;

// -----------------------------------------------------------------------------
// the binary search tree

using std::vector;

namespace TMVA {

  class DataSet;
  class Event;
  
  class BinarySearchTree : public BinaryTree {
    
  public:
    
    // constructor
    BinarySearchTree( void );
    
    // copy constructor
    BinarySearchTree (const BinarySearchTree &b);

    // destructor
    virtual ~BinarySearchTree( void );
    
    // Searches for a node with the specified data 
    // by calling  the private, recursive, function for searching
    BinarySearchTreeNode* Search( Event * event ) const;
    
    // Adds an item to the tree, 
    void Insert( Event * , Bool_t eventOwnership=kFALSE );
    
    //get sum of weights of the nodes;
    Double_t GetSumOfWeights( void ) const;
    
    //set the periode (number of variables)
    inline void SetPeriode( Int_t p )      { fPeriode = p; }
    // return periode (number of variables)
    inline UInt_t  GetPeriode( void ) const { return fPeriode; }
    
    
    
    
    // counts events (weights) within a given volume 
    Double_t SearchVolume( Volume*, std::vector<TMVA::Event*>* events = 0 );
    
    // create the search tree from the events in a TTree
    // using the variables specified in "theVars"
    Int_t Fill( const DataSet& ds, TTree* theTree, Int_t theType = -1, 
                Types::EPreprocessingMethod corr = Types::kNone, 
                Types::ESBType type = Types::kSignal );
    
    // Create the search tree from the event collection 
    // using ONLY the variables specified in "theVars"
    Int_t Fill( vector<TMVA::Event*>, vector<Int_t> theVars, Int_t theType = -1 );
    
    // create the search tree from the events in a TTree
    // using ALL the variables specified included in the Event
    Int_t Fill( vector<TMVA::Event*> theTree, Int_t theType = -1 );
    
  private:

    Double_t   fSumOfWeights;      //sum of the events (node) weights

    // add a new  node to the tree (as daughter) 
    void       Insert( Event*, Node* , Bool_t eventOwnership=kFALSE );
    // recursively search the nodes for Event
    BinarySearchTreeNode*      Search( Event*, Node *) const ;
    
    //check of Event variables lie with the volumde
    Bool_t   InVolume    ( TMVA::Event*, Volume* ) const;
    //
    void     DestroyNode ( BinarySearchTreeNode* );
    // recursive search through daughter nodes in weight counting
    Double_t SearchVolume( Node*, Volume*, Int_t, 
                           std::vector<Event*>* events );
    UInt_t fPeriode;              // periode (number of event variables)
    UInt_t fCurrentDepth;         // internal variable, counting the depth of the tree during insertion    
  
    ClassDef(BinarySearchTree,0) // Binary search tree including volume search method  
      ;
  };
  
  // -----------------------------------------------------------------------------
  
} // namespace TMVA

#endif
