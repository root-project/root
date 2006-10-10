// @(#)root/tmva $Id: Node.h,v 1.5 2006/08/31 11:03:37 rdm Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: Node, NodeID                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the BinarySearch or Decision Trees                               *
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
 * $Id: Node.h,v 1.5 2006/08/31 11:03:37 rdm Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_Node
#define ROOT_TMVA_Node

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Node                                                                 //
//                                                                      //
// Node for the BinarySearch or Decision Trees                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TMVA_NodeID
#include "TMVA/NodeID.h"
#endif

namespace TMVA {

   class NodeID;
   class Node;
   class Event;
   ostream& operator << (ostream& os, const Node& node);
   ostream& operator << (ostream& os, const Node* node);

   // a class used to identify a Node; (needed for recursive reading from text file)
   // (currently it is NOT UNIQUE... but could eventually made it
   // a node in the tree structure
   class Node {

      // output operator for a node
      friend ostream& operator << (ostream& os, const Node& node);
      // output operator with a pointer to the node (which still prints the node itself)
      friend ostream& operator << (ostream& os, const Node* node);

   public:

      // constructor of a node for the search tree
      Node( Event* e =NULL, Bool_t o=kFALSE ) : fEvent( e ), fLeft( NULL ),
                                                fRight( NULL ), fParent ( NULL ), fSelector( -1 ), fEventOwnership ( o ) {}

      // constructor of a daughter node as a daughter of 'p'
      Node( Node* p ) : fEvent( NULL ), fLeft( NULL ),
                        fRight( NULL ), fParent ( p ), fSelector( -1 ), fEventOwnership (kFALSE) {}

      // destructor
      virtual ~Node ();

      // test event if it decends the tree at this node to the right
      virtual Bool_t GoesRight( const Event* ) const;
      // test event if it decends the tree at this node to the left
      virtual Bool_t GoesLeft ( const Event* ) const;
      // test event if it is equal to the event that "makes the node" (just for the "search tree"
      virtual Bool_t EqualsMe ( const Event* ) const;

      // return pointer to the left daughter node
      inline Node* GetLeft  () const { return fLeft;   }
      // return pointer to the right daughter node
      inline Node* GetRight () const { return fRight;  }
      // return pointer to the parent node
      inline Node* GetParent() const { return fParent; }

      // set pointer to the left daughter node
      inline void SetLeft  (Node* l) { fLeft   = l;}
      // set pointer to the right daughter node
      inline void SetRight (Node* r) { fRight  = r;}
      // set pointer to the parent node
      inline void SetParent(Node* p) { fParent = p;}

      // set index of variable used for discrimination at this node
      inline void SetSelector( Short_t i) { fSelector = i; }
      // set index of variable used for discrimination at this node
      inline void SetSelector( Int_t i  ) { fSelector = Short_t(i); }
      // return index of variable used for discrimination at this node
      inline Short_t GetSelector() const { return fSelector; }
      // set the EVENT that forms this node (in search tree)
      inline void SetData( Event* e ) { fEvent = e; }
      // return the EVENT that forms this node (in search tree)
      inline Event* GetData() const { return fEvent; }

      //recursively go through the part of the tree below this node and count all daughters
      Int_t  CountMeAndAllDaughters() const;
      // printout of the node
      void   Print( ostream& os ) const;

      // recursive printout of the node and it daughters
      virtual void PrintRec( ostream& os, Int_t depth=0, const std::string pos="root" ) const;
      // recursive reading of the node (essectially the whole tree) from a text file
      virtual NodeID ReadRec( ifstream& is, NodeID nodeID, Node* parent=NULL );

      // return true/false if the EVENT* that forms the node is owned by the node or not
      Bool_t      GetEventOwnership( void           ) { return fEventOwnership; }
      // set if the EVENT* that forms the node is owned by the node or not
      void        SetEventOwnership( Bool_t b ) { fEventOwnership = b; }

   private:

      Event* fEvent;   // event that forms the node (search tree)

      Node*  fLeft;    // pointers to the two "daughter" nodes
      Node*  fRight;   // pointers to the two "daughter" nodes
      Node*  fParent;  // the previous (parent) node

      Short_t     fSelector;// index of variable used in node selection (decision tree)
      Bool_t      fEventOwnership; //flag if Event* is owned by the node or not

      ClassDef(Node,0); //Node for the BinarySearch or Decision Trees
   };

} // namespace TMVA

#endif

