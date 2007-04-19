// @(#)root/tmva $Id: BinarySearchTreeNode.h,v 1.5 2006/11/23 17:43:38 rdm Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: Node, NodeID                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the BinarySearch                                                 *
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

#ifndef ROOT_TMVA_BinarySearchTreeNode
#define ROOT_TMVA_BinarySearchTreeNode

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinarySearchTreeNode                                                 //
//                                                                      //
// Node for the BinarySearch  Tree                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include "Riostream.h"
#include "Rtypes.h"

#ifndef ROOT_TMVA_Node
#include "TMVA/Node.h"
#endif

namespace TMVA {

   class Event;
   class MsgLogger;

   // a class used to identify a Node; (needed for recursive reading from text file)
   // (currently it is NOT UNIQUE... but could eventually made it
   // a node in the tree structure
   class BinarySearchTreeNode : public Node  {

   public:

      // constructor of a node for the search tree
      BinarySearchTreeNode( Event* e = NULL );

      // constructor of a daughter node as a daughter of 'p'
      BinarySearchTreeNode( BinarySearchTreeNode* parent, char pos );

      // copy constructor
      BinarySearchTreeNode ( const BinarySearchTreeNode &n, 
                             BinarySearchTreeNode* parent = NULL);

      // destructor
      virtual ~BinarySearchTreeNode ();

      // test event if it decends the tree at this node to the right  
      virtual Bool_t GoesRight( const Event& ) const;
      // test event if it decends the tree at this node to the left 

      virtual Bool_t GoesLeft ( const Event& ) const;
      // test event if it is equal to the event that "makes the node" (just for the "search tree"  

      virtual Bool_t EqualsMe ( const Event& ) const;

      // set index of variable used for discrimination at this node
      inline void SetSelector( Short_t i) { fSelector = i; }
      // return index of variable used for discrimination at this node 
      inline Short_t GetSelector() const { return fSelector; }

      const std::vector<Float_t> & GetEventV() const { return fEventV; }
      Float_t                      GetWeight() const { return fWeight; }
      Bool_t                       IsSignal()  const { return fIsSignal; }

      // printout of the node
      virtual void Print( ostream& os ) const;

      // recursive printout of the node and it daughters 
      virtual void PrintRec( ostream& os ) const;

      // recursive reading of the node (essectially the whole tree) from a text file 
      virtual void ReadRec( istream& is, char &pos, 
                            UInt_t &depth, TMVA::Node* parent=NULL );

      virtual Int_t GetMemSize() const;

   private: 
      // Read the data block
      Bool_t      ReadDataRecord( istream& is );

      std::vector<Float_t> fEventV;
      Float_t     fWeight;
      Bool_t      fIsSignal;

      Short_t     fSelector;       // index of variable used in node selection (decision tree) 

      ClassDef(BinarySearchTreeNode,0) // Node for the BinarySearchTree
   };

} // namespace TMVA

#endif

