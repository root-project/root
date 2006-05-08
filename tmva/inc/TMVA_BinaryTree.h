// @(#)root/tmva $Id: TMVA_BinaryTree.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : TMVA_BinaryTree                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      TMVA_BinaryTree: A base class for BinarySearch- or Decision-Trees         *
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_BinaryTree.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_BinaryTree
#define ROOT_TMVA_BinaryTree

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinaryTree                                                           //
//                                                                      //
// Base class for BinarySearch and Decision Trees                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include "Riostream.h"
#include "TROOT.h"
#ifndef ROOT_TMVA_Node
#include "TMVA_Node.h"
#endif

class TMVA_Event;

// -----------------------------------------------------------------------------

// the actual tree class
// Handles allocation, deallocation, and sorting of nodes.
// the Tree consists of a "root-node" wich might have  0 to 2 daughther nodes

class TMVA_BinaryTree {

  friend ostream& operator << ( ostream& os, const TMVA_BinaryTree& tree );
  
 public:
  // or a tree with Root node "n", any daughters of this node are automatically in the tree
  TMVA_BinaryTree( void );
  virtual ~TMVA_BinaryTree( void );
  
  void SetRoot( TMVA_Node* r ) { fRoot = r; }

  // Retrieves the address of the root node
  TMVA_Node* GetRoot( void ) const { return fRoot; }

  // Searches for a node with the specified data 
  // by calling  the private, recursive, function for searching
  TMVA_Node* Search( TMVA_Event* event ) const;
  
  // Adds an item to the tree, 
  void Insert( TMVA_Event* , Bool_t eventOwnership=kFALSE );

  // get number of Nodes in the Tree as counted while booking the nodes;
  Int_t GetNNodes( void );

  // cout the number of Nodes in the Tree by looping through the tree
  Int_t CountNodes( void );

  //get sum of weights of the nodes;
  Double_t GetSumOfWeights( void );

  inline void SetPeriode( Int_t p )      { fPeriode = p; }
  inline Int_t  GetPeriode( void ) const { return fPeriode; }

  void Print(ostream & os) const;

 protected:

  Int_t      fNNodes;
  Double_t   fSumOfWeights;

 private:
  
  TMVA_Node    *fRoot;
  // the tree only has it's root node, the "daughters" are taken car 
  // of by the "node" properties of the "root"
  
  void       Insert( TMVA_Event*, TMVA_Node* , Bool_t eventOwnership=kFALSE );
  TMVA_Node* Search( TMVA_Event*, TMVA_Node *) const ;
  void       DeleteNode( TMVA_Node* );
  ostream&   PrintOrdered( ostream & os, TMVA_Node* n ) const;

  Int_t fPeriode;
  Int_t fCurrentDepth;
  
  ClassDef(TMVA_BinaryTree,0) //Base class for BinarySearch and Decision Trees
};

#endif // ROOT_TMVA_BinaryTree

