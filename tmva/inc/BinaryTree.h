// @(#)root/tmva $Id: BinaryTree.h,v 1.4 2006/08/31 11:03:37 rdm Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : BinaryTree                                                            *
 *                                                                                *
 * Description:                                                                   *
 *      BinaryTree: A base class for BinarySearch- or Decision-Trees              *
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

#include <vector>
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#ifndef ROOT_TROOT
#include "TROOT.h"
#endif

#ifndef ROOT_TMVA_Node
#include "TMVA/Node.h"
#endif
#ifndef ROOT_TMVA_Node
#include "TMVA/Event.h"
#endif

// -----------------------------------------------------------------------------

// the actual tree class
// Handles allocation, deallocation, and sorting of nodes.
// the Tree consists of a "root-node" wich might have  0 to 2 daughther nodes

namespace TMVA {

  class BinaryTree;
  ostream& operator<< ( ostream& os, const BinaryTree& tree );

  class BinaryTree {

    friend ostream& operator<< ( ostream& os, const BinaryTree& tree );

  public:

    // or a tree with Root node "n", any daughters of this node are automatically in the tree
    BinaryTree( void );
    virtual ~BinaryTree( void );

    void SetRoot( Node* r ) { fRoot = r; }

    // Retrieves the address of the root node
    Node* GetRoot( void ) const { return fRoot; }

    // Searches for a node with the specified data
    // by calling  the private, recursive, function for searching
    Node* Search( Event* event ) const;

    // Adds an item to the tree,
    void Insert( Event* , Bool_t eventOwnership=kFALSE );

    // get number of Nodes in the Tree as counted while booking the nodes;
    Int_t GetNNodes( void );

    // cout the number of Nodes in the Tree by looping through the tree
    Int_t CountNodes( void );

    //get sum of weights of the nodes;
    Double_t GetSumOfWeights( void );

    //set the periode (number of variables)
    inline void SetPeriode( Int_t p )      { fPeriode = p; }
    // return periode (number of variables)
    inline Int_t  GetPeriode( void ) const { return fPeriode; }

    void Print(ostream & os) const;

  protected:

    Int_t      fNNodes;            //total number of nodes in the tree (counted)
    Double_t   fSumOfWeights;      //sum of the events (node) weights

  private:

    Node    *fRoot;                //the root node of the tree
    // the tree only has it's root node, the "daughters" are taken car
    // of by the "node" properties of the "root"

    // add a new  node to the tree (as daughter)
    void       Insert( Event*, Node* , Bool_t eventOwnership=kFALSE );
    // recursively search the nodes for Event
    Node* Search( Event*, Node *) const ;
    // delete a node (and the corresponding event if owned by the tree)
    void       DeleteNode( Node* );
    // ordered output of the events in the binary tree
    ostream&   PrintOrdered( ostream & os, Node* n ) const;

    Int_t fPeriode;        //periode (number of event variables)
    Int_t fCurrentDepth;   //internal variable, counting the depth of the tree during insertion

    ClassDef(BinaryTree,0); // Base class for BinarySearch and Decision Trees

  };
} // namespace TMVA

#endif

