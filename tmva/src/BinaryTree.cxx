// @(#)root/tmva $Id: BinaryTree.cxx,v 1.24 2006/11/16 22:51:58 helgevoss Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::BinaryTree                                                      *
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
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      //
// Base class for BinarySearch and Decision Trees                       //
//______________________________________________________________________//

#include <string>
#include <stdexcept>
#include "TMVA/BinaryTree.h"
#include "Riostream.h"
#include "TMVA/Event.h"

ClassImp(TMVA::BinaryTree)
   ;

//_______________________________________________________________________
TMVA::BinaryTree::BinaryTree( void )
   : fRoot ( NULL ), 
     fNNodes ( 0 ),
     fLogger( "BinaryTree" )
{
   // constructor for a yet "empty" tree. Needs to be filled afterwards
}

//_______________________________________________________________________
TMVA::BinaryTree::~BinaryTree( void ) 
{
   //destructor (deletes the nodes and "events" if owned by the tree

   this->DeleteNode( fRoot );
}

//_______________________________________________________________________
void TMVA::BinaryTree::DeleteNode( TMVA::Node* node )
{ 
   // protected, recursive, function used by the class destructor and when Pruning
   if (node != NULL) { //If the node is not NULL...
      this->DeleteNode(node->GetLeft());  //Delete its left node.
      this->DeleteNode(node->GetRight()); //Delete its right node.
      //      cout << "bla: delete node at depth " << node->GetDepth() <<endl;
      delete node;                //Delete the node in memory....darf ich aber nicht
   }
}


//_______________________________________________________________________
TMVA::Node* TMVA::BinaryTree::GetLeftDaughter( Node *n)
{
   // get left daughter node current node "n"
   return (Node*) n->GetLeft();
}

//_______________________________________________________________________
TMVA::Node* TMVA::BinaryTree::GetRightDaughter( Node *n)
{
   // get right daughter node current node "n"
   return (Node*) n->GetRight();
}

UInt_t TMVA::BinaryTree::CountNodes(TMVA::Node *n)
{
   // return the number of nodes in the tree. (make a new count --> takes time)

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (Node*) this->GetRoot();
      if (n == NULL) return 0 ;
   } 

   UInt_t countNodes=1;

   if (this->GetLeftDaughter(n) != NULL){
      countNodes += this->CountNodes( this->GetLeftDaughter(n) );
   }
   if (this->GetRightDaughter(n) != NULL) {
      countNodes += this->CountNodes( this->GetRightDaughter(n) );
   }

   return fNNodes = countNodes;
}




//_______________________________________________________________________
void TMVA::BinaryTree::Print(ostream & os) const
{
   //recursively print the tree
   this->GetRoot()->PrintRec(os);
}

//_______________________________________________________________________
ostream& TMVA::operator<< (ostream& os, const TMVA::BinaryTree& tree)
{ 
   //print the tree recursinvely using the << operator
   tree.Print(os);
   return os; // Return the output stream.
}
