// @(#)root/tmva $Id: BinaryTree.cxx,v 1.5 2006/05/22 09:06:25 helgevoss Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * class  : TMVA::BinaryTree                                                      *
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

//_______________________________________________________________________
TMVA::BinaryTree::BinaryTree( void )
  :     fNNodes  ( 0 ), 
        fSumOfWeights( 0 ),
        fRoot ( NULL ), 
        fPeriode ( 1 ) 
{
  //constructor for a yet "empty" tree. Needs to be filled afterwards
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
   // Private, recursive, function used by the class destructor.
  if (node != NULL) { //If the node is not NULL...
    this->DeleteNode(node->GetLeft());  //Delete its left node.
    this->DeleteNode(node->GetRight()); //Delete its right node.
    delete node;                //Delete the node in memory....darf ich aber nicht
  }
}

//_______________________________________________________________________
void TMVA::BinaryTree::Insert( TMVA::Event* event, Bool_t eventOwnership ) 
{
  //insert a new "event" in the binary tree
  //   set "eventOwnershipt" to kTRUE if the event should be owned (deleted) by
  //   the tree
  fCurrentDepth=0;
  if (fRoot == NULL) {           //If the list is empty...
    fRoot = new TMVA::Node(event, eventOwnership);        //Make the new node the root.
    fNNodes++;
    fSumOfWeights+=event->GetWeight();
    fRoot->SetSelector(0);
    this->SetPeriode(event->GetEventSize());
  }
  else {
    //sanity check:
    if (event->GetEventSize() != this->GetPeriode()) {
      cout << "--- TMVA::BinaryTree: ERROR!!!  Event vector length != Periode specified in Binary Tree\n";
      cout << "---   event size: " << event->GetEventSize() << " Periode: " << this->GetPeriode() <<endl;
      cout << "---   and all this when trying filling the "<<fNNodes+1<<"th Node"<<endl;
      exit(1);
    }
    this->Insert(event, fRoot, eventOwnership); //insert a new node at the propper position  
  }
}

//_______________________________________________________________________
void TMVA::BinaryTree::Insert( TMVA::Event *event, TMVA::Node *node, Bool_t eventOwnership ) 
{
  //private internal fuction to insert a event (node) at the proper position
  fCurrentDepth++;
  if (node->GoesLeft(event)){    // If the adding item is less than the current node's data...
    if (node->GetLeft() != NULL){            // If there is a left node...
      this->Insert(event, node->GetLeft(), eventOwnership); // Add the new node to it.
    } 
    else {                             // If there is not a left node...
      TMVA::Node* current = new TMVA::Node(event, eventOwnership);        // Make the new node.
      fNNodes++;
      fSumOfWeights+=event->GetWeight();
      current->SetSelector(fCurrentDepth%event->GetEventSize());
      current->SetParent(node);          // Set the new node's previous node.
      node->SetLeft(current);            // Make it the left node of the current one.
    }  
  } 
  else if (node->GoesRight(event)) { // If the adding item is less than or equal to the current node's data...
    if (node->GetRight() != NULL) {              // If there is a right node...
      this->Insert(event, node->GetRight(), eventOwnership);    // Add the new node to it.
    } 
    else {                                 // If there is not a right node...
      TMVA::Node* current = new TMVA::Node(event, eventOwnership);            // Make the new node.
      fNNodes++;
      fSumOfWeights+=event->GetWeight();
      current->SetSelector(fCurrentDepth%event->GetEventSize());
      current->SetParent(node);              // Set the new node's previous node.
      node->SetRight(current);               // Make it the left node of the current one.
    }
  } 
  else {
    cout << "--- TMVA::BinaryTree: Error!! in Insert Event, neither left nor right :)\n";
    exit(1);
  }
}

//_______________________________________________________________________
TMVA::Node* TMVA::BinaryTree::Search( TMVA::Event* event ) const 
{ 
  //search the tree to find the node matching "event"
  return this->Search( event, fRoot );
}

//_______________________________________________________________________
TMVA::Node* TMVA::BinaryTree::Search(TMVA::Event* event, TMVA::Node* node) const 
{ 
// Private, recursive, function for searching.
  if (node != NULL) {               // If the node is not NULL...
    if (node->EqualsMe(event))      // If we have found the node...
      return node;                  // Return it
    if (node->GoesLeft(event))      // If the node's data is greater than the search item...
      return this->Search(event, node->GetLeft());  //Search the left node.
    else                          //If the node's data is less than the search item...
      return this->Search(event, node->GetRight()); //Search the right node.
  }
  else return NULL; //If the node is NULL, return NULL.
}

//_______________________________________________________________________
Double_t TMVA::BinaryTree::GetSumOfWeights( void )
{
  //return the sum of event (node) weights
  return fSumOfWeights;
}

Int_t TMVA::BinaryTree::GetNNodes( void )
{
  // return the number of nodes in the tree (as counted during tree construction)
  return fNNodes;
}

Int_t TMVA::BinaryTree::CountNodes()
{
  // return the number of nodes in the tree. (make a new count --> takes time)
  if (this->GetRoot()!=NULL) return this->GetRoot()->CountMeAndAllDaughters();
  else return 0;
}

//_______________________________________________________________________
ostream &TMVA::BinaryTree::PrintOrdered( ostream & os, TMVA::Node* n ) const
{
  //print all events in binary tree (gives ordered output)
  if (n!=NULL){
    PrintOrdered(os,n->GetLeft());
    os << *n;
    PrintOrdered(os,n->GetRight());
  }    
  return os;
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
// print the whole tree recursively in such a way that it starts with bottom/left most entry
// and then it moves up the tree always printing the left most part. Like this, 
// in a binary search tree, the output is ordered with the first number beeing the smallest.

  TMVA::Node* current = tree.GetRoot(); // Start with the parent node.
  if (current != NULL){             // If there are any nodes in the list...
    while (current->GetLeft() != NULL) // Move to the left-most node,
      current = current->GetLeft();    // so output will be ordered.
    while (current != NULL){        // While there are still more nodes in the list...
      os << *current;               // Output the current node.
      if (current->GetRight() != NULL) // If there is a right node...
        tree.PrintOrdered(os, current->GetRight()); // Print it out.
      current = current->GetParent();               // Move up one node.
    }
  }
  return os; // Return the output stream.
}







