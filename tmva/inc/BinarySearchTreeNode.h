// @(#)root/tmva $Id: BinarySearchTreeNode.h,v 1.5 2006/11/16 22:51:58 helgevoss Exp $    
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

   // a class used to identify a Node; (needed for recursive reading from text file)
   // (currently it is NOT UNIQUE... but could eventually made it
   // a node in the tree structure
   class BinarySearchTreeNode : public Node  {
	    
   public:

      // constructor of a node for the search tree
      BinarySearchTreeNode( Event* e = NULL, Bool_t o=kFALSE ) : 
        TMVA::Node(), fEvent( e ), fEventOwnership ( o ),  fSelector( -1 ) {}
	    
      // constructor of a daughter node as a daughter of 'p'
      BinarySearchTreeNode( BinarySearchTreeNode* p, char pos ) : TMVA::Node(p,pos),
        fEvent( NULL ), fEventOwnership (kFALSE), fSelector( -1 ) {}

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
      inline void SetSelector( const Short_t i) { fSelector = i; }
      // return index of variable used for discrimination at this node 
      inline Short_t GetSelector() const { return fSelector; }
      // set the EVENT that forms this node (in search tree)
      inline void SetEvent( Event* e ) { fEvent = e; }

      // return the EVENT that forms this node (in search tree)
      inline Event* GetEvent() const { return fEvent; }
	    
      // printout of the node
      virtual void Print( ostream& os ) const;
	    
      // recursive printout of the node and it daughters 
      virtual void PrintRec( ostream& os ) const;

      // recursive reading of the node (essectially the whole tree) from a text file 
      virtual void ReadRec( istream& is, char &pos, 
                            UInt_t &depth, TMVA::Node* parent=NULL );
	    
      // return true/false if the EVENT* that forms the node is owned by the node or not 
      Bool_t      GetEventOwnership( void           ) { return fEventOwnership; }
      // set if the EVENT* that forms the node is owned by the node or not 
      void        SetEventOwnership( const Bool_t b ) { fEventOwnership = b; }
	    
   private: 

      Event* fEvent;               // event that forms the node (search tree)
      Bool_t      fEventOwnership; // flag if Event* is owned by the node or not
	
      Short_t     fSelector;       // index of variable used in node selection (decision tree) 
    
      ClassDef(BinarySearchTreeNode,0) // Node for the BinarySearchTree
         ;
   };
	  
} // namespace TMVA

#endif

