// @(#)root/tmva $Id: Node.cxx,v 1.13 2007/04/21 14:20:46 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Classes: Node                                                                  *
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
 * CopyRight (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Node for the BinarySearch or Decision Trees 
//                        
// for the binary search tree, it basically consists of the EVENT, and 
// pointers to the parent and daughters
//                                                                       
// in case of the Decision Tree, it specifies parent and daughters, as
// well as "which variable is used" in the selection of this node, including
// the respective cut value.
//______________________________________________________________________

#include "TMVA/Node.h"
#include "Riostream.h"
#include <stdexcept>

ClassImp(TMVA::Node)

Int_t TMVA::Node::fgCount = 0;

TMVA::Node::Node() 
   : fParent( NULL ),
     fLeft  ( NULL),
     fRight ( NULL ),
     fPos   ( 'u' ),
     fDepth ( 0 ),
     fParentTree( NULL )
{
   // default constructor
   fgCount++;
}

//_______________________________________________________________________
TMVA::Node::Node( Node* p, char pos ) 
   : fParent ( p ), 
     fLeft ( NULL ), 
     fRight( NULL ),  
     fPos  ( pos ), 
     fDepth( p->GetDepth() + 1), 
     fParentTree(p->GetParentTree()) 
{
   // constructor of a daughter node as a daughter of 'p'

   fgCount++;
   if (fPos == 'l' ) p->SetLeft(this);
   else if (fPos == 'r' ) p->SetRight(this);
}

//_______________________________________________________________________
TMVA::Node::Node (const Node &n ) 
   : fParent( NULL ), 
     fLeft  ( NULL), 
     fRight ( NULL ), 
     fPos   ( n.fPos ), 
     fDepth ( n.fDepth ), 
     fParentTree( NULL )
{
   // copy constructor, make sure you don't just copy the poiter to the node, but
   // that the parents/daugthers are initialized to 0 (and set by the copy 
   // constructors of the derived classes 
   fgCount++;
}

//_______________________________________________________________________
TMVA::Node::~Node( void )
{
   // node destructor
   fgCount--;
}

//_______________________________________________________________________
Int_t TMVA::Node::CountMeAndAllDaughters( void ) const 
{
   //recursively go through the part of the tree below this node and count all daughters
   Int_t n=1;
   if (this->GetLeft() != NULL) 
      n+= this->GetLeft()->CountMeAndAllDaughters(); 
   if (this->GetRight() != NULL) 
      n+= this->GetRight()->CountMeAndAllDaughters(); 
  
   return n;
}

//_______________________________________________________________________
Int_t TMVA::Node::GetMemSize() const 
{ 
   // calculates the memory size of the node
   Int_t size = sizeof(*this);
   if (fLeft !=0) size += fLeft->GetMemSize();
   if (fRight!=0) size += fRight->GetMemSize();
   return size;
}

// print a node
//_______________________________________________________________________
ostream& TMVA::operator<<(ostream& os, const TMVA::Node& node)
{ 
   // output operator for a node  
   node.Print(os);
   return os;                // Return the output stream.
}

//_______________________________________________________________________
ostream& TMVA::operator<<(ostream& os, const TMVA::Node* node)
{ 
   // output operator with a pointer to the node (which still prints the node itself)
   if (node!=NULL) node->Print(os);
   return os;                // Return the output stream.
}


