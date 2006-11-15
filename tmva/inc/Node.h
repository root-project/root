// @(#)root/tmva $Id: Node.h,v 1.19 2006/11/13 15:49:49 helgevoss Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: Node                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Node
#define ROOT_TMVA_Node

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Node                                                                 //
//                                                                      //
// Node base class for the BinarySearch or Decision Trees Nodes         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include "Riostream.h"
#include "Rtypes.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif



namespace TMVA {

   class Node;
   class Event;
   class BinaryTree;

   ostream& operator<<( ostream& os, const Node& node );
   ostream& operator<<( ostream& os, const Node* node );

   // a class used to identify a Node; (needed for recursive reading from text file)
   // (currently it is NOT UNIQUE... but could eventually made it
   // a node in the tree structure
   class Node {
	    
      // output operator for a node
      friend ostream& operator << (ostream& os, const Node& node);
      // output operator with a pointer to the node (which still prints the node itself)
      friend ostream& operator << (ostream& os, const Node* node);
	    
   public:

      // constructor of a node 
      Node() : fParent ( NULL ), fLeft( NULL), fRight( NULL ),  
        fPos ('u'), fDepth(0), fParentTree(NULL) { fgCount++; }
      
      // constructor of a daughter node as a daughter of 'p'
      Node( Node* p, char pos );

      // copy constructor
      Node (const Node &n );

      // destructor
      virtual ~Node ();
	      
      // test event if i{ decends the tree at this node to the right  
      virtual Bool_t GoesRight( const Event& ) const = 0;
      // test event if it decends the tree at this node to the left 

      virtual Bool_t GoesLeft ( const Event& ) const = 0;
      // test event if it is equal to the event that "makes the node" (just for the "search tree"  

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
	    
      //recursively go through the part of the tree below this node and count all daughters
      Int_t  CountMeAndAllDaughters() const;
	    
      // printout of the node
      virtual void Print( ostream& os ) const = 0;
	    
      // recursive printout of the node and it daughters 
      virtual void PrintRec ( ostream& os ) const = 0;
      // recursive reading of the node (essectially the whole tree) from a text file 
      virtual void ReadRec( istream& is, char &pos, 
                            UInt_t &depth, Node* parent=NULL ) = 0;
      
      // Set depth, layer of the where the node is within the tree, seen from the top (root)
      void SetDepth(const UInt_t d){fDepth=d;}
      
      // Return depth, layer of the where the node is within the tree, seen from the top (root)
      UInt_t GetDepth() const {return fDepth;}
      
      // set node position, i.e, the node is a left (l) or right (r) daugther
      void SetPos(const char s) {fPos=s;}
      
      // Return the node position, i.e, the node is a left (l) or right (r) daugther
      char GetPos() const {return fPos;}

      // Return the pointer to the Parent tree to which the Node belongs 
      TMVA::BinaryTree* GetParentTree() const {return fParentTree;}

      // set the pointer to the Parent Tree to which the Node belongs 
      void SetParentTree(TMVA::BinaryTree* t) {fParentTree = t;} 

      int GetCount(){return fgCount;}
   private: 

      Node*  fParent;              // the previous (parent) node
      Node*  fLeft;                // pointers to the two "daughter" nodes
      Node*  fRight;               // pointers to the two "daughter" nodes

      char         fPos;    // position, i.e. it is a left (l) or right (r) daughter 
      UInt_t       fDepth;  // depth of the node within the tree (seen from root node)

      BinaryTree*   fParentTree; //pointer to the parent tree to which the Node belongs 

      static int fgCount;   // counter of all nodes present.. for debug.. to spot memory leaks...

      ClassDef(Node,0) // Node for the BinarySearch or Decision Trees
         ;
   };
	  
} // namespace TMVA

#endif

