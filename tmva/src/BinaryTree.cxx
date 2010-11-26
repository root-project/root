// @(#)root/tmva $Id$
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
 *      Joerg Stelzer   <stelzer@cern.ch>        - DESY, Germany                  *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      DESY, Germany                                                             *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// BinaryTree                                                           //
//                                                                      //
// Base class for BinarySearch and Decision Trees                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <stdexcept>

#include "TMVA/BinaryTree.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Event.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::BinaryTree)

TMVA::MsgLogger* TMVA::BinaryTree::fgLogger = 0;

//_______________________________________________________________________
TMVA::BinaryTree::BinaryTree( void )
   : fRoot  ( NULL ),
     fNNodes( 0 ),
     fDepth ( 0 )
{
   // constructor for a yet "empty" tree. Needs to be filled afterwards
   if (!fgLogger) fgLogger =  new MsgLogger("BinaryTree");
}

//_______________________________________________________________________
TMVA::BinaryTree::~BinaryTree( void )
{
   //destructor (deletes the nodes and "events" if owned by the tree
   this->DeleteNode( fRoot );
   fRoot=0;
}

//_______________________________________________________________________
void TMVA::BinaryTree::DeleteNode( TMVA::Node* node )
{
   // protected, recursive, function used by the class destructor and when Pruning
   if (node != NULL) { //If the node is not NULL...
      this->DeleteNode(node->GetLeft());  //Delete its left node.
      this->DeleteNode(node->GetRight()); //Delete its right node.

      delete node;                // Delete the node in memory
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

//_______________________________________________________________________
UInt_t TMVA::BinaryTree::CountNodes(TMVA::Node *n)
{
   // return the number of nodes in the tree. (make a new count --> takes time)

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (Node*)this->GetRoot();
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
   // recursively print the tree
   this->GetRoot()->PrintRec(os);
   os << "-1" << std::endl;
}

//_______________________________________________________________________
void* TMVA::BinaryTree::AddXMLTo(void* parent) const {
   // add attributes to XML

   void* bdt = gTools().AddChild(parent, "BinaryTree");
   gTools().AddAttr( bdt, "type" , ClassName() );
   this->GetRoot()->AddXMLTo(bdt);
   return bdt;
}

//_______________________________________________________________________
void TMVA::BinaryTree::ReadXML(void* node, UInt_t tmva_Version_Code ) {
   // read attributes from XML

   this->DeleteNode( fRoot );
   fRoot= CreateNode();

   void* trnode = gTools().GetChild(node);
   fRoot->ReadXML(trnode, tmva_Version_Code);

   this->SetTotalTreeDepth();
}


//_______________________________________________________________________
ostream& TMVA::operator<< (ostream& os, const TMVA::BinaryTree& tree)
{
   // print the tree recursinvely using the << operator
   tree.Print(os);
   return os; // Return the output stream.
}

//_______________________________________________________________________
void TMVA::BinaryTree::Read(istream & istr, UInt_t tmva_Version_Code )
{
   // Read the binary tree from an input stream.
   // The input stream format depends on the tree type,
   // it is defined be the node of the tree

   Node * currentNode = GetRoot();
   Node* parent = 0;

   if(currentNode==0) {
      currentNode=CreateNode();
      SetRoot(currentNode);
   }

   while(1) {
      if ( ! currentNode->ReadDataRecord(istr, tmva_Version_Code) ) {
         delete currentNode;
         this->SetTotalTreeDepth();
         return;
      }

      // find parent node
      while( parent!=0 && parent->GetDepth() != currentNode->GetDepth()-1) parent=parent->GetParent();
      
      if (parent!=0) { // link new node to parent
         currentNode->SetParent(parent);
         if (currentNode->GetPos()=='l') parent->SetLeft(currentNode);
         if (currentNode->GetPos()=='r') parent->SetRight(currentNode);
      }

      parent = currentNode; // latest node read might be parent of new one

      currentNode = CreateNode(); // create a new node to be read next
   }
}

//_______________________________________________________________________
istream& TMVA::operator>> (istream& istr, TMVA::BinaryTree& tree)
{ 
   // read the tree from an istream
   tree.Read(istr);
   return istr;
}
//_______________________________________________________________________
void TMVA::BinaryTree::SetTotalTreeDepth( Node *n)
{
   // descend a tree to find all its leaf nodes, fill max depth reached in the
   // tree at the same time. 

   if (n == NULL){ //default, start at the tree top, then descend recursively
      n = (Node*) this->GetRoot();
      if (n == NULL) {
         Log() << kFATAL << "SetTotalTreeDepth: started with undefined ROOT node" <<Endl;
         return ;
      }
   } 
   if (this->GetLeftDaughter(n) != NULL){
      this->SetTotalTreeDepth( this->GetLeftDaughter(n) );
   }
   if (this->GetRightDaughter(n) != NULL) {
      this->SetTotalTreeDepth( this->GetRightDaughter(n) );
   }
   if (n->GetDepth() > this->GetTotalTreeDepth()) this->SetTotalTreeDepth(n->GetDepth());

   return;
}
