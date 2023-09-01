// @(#)root/tmva $Id$
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * CopyRight (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::Node
\ingroup TMVA
Node for the BinarySearch or Decision Trees.

For the binary search tree, it basically consists of the EVENT, and
pointers to the parent and daughters

In case of the Decision Tree, it specifies parent and daughters, as
well as "which variable is used" in the selection of this node,
including the respective cut value.
*/

#include <stdexcept>
#include <iosfwd>
#include <iostream>

#include "TMVA/Node.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::Node);

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

////////////////////////////////////////////////////////////////////////////////
/// constructor of a daughter node as a daughter of 'p'

TMVA::Node::Node( Node* p, char pos )
   : fParent ( p ),
     fLeft ( NULL ),
     fRight( NULL ),
     fPos  ( pos ),
     fDepth( p->GetDepth() + 1),
     fParentTree(p->GetParentTree())
{
   fgCount++;
   if (fPos == 'l' ) p->SetLeft(this);
   else if (fPos == 'r' ) p->SetRight(this);
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor, make sure you don't just copy the pointer to the node, but
/// that the parents/daughters are initialized to 0 (and set by the copy
/// constructors of the derived classes

TMVA::Node::Node ( const Node &n )
   : fParent( NULL ),
     fLeft  ( NULL),
     fRight ( NULL ),
     fPos   ( n.fPos ),
     fDepth ( n.fDepth ),
     fParentTree( NULL )
{
   fgCount++;
}

////////////////////////////////////////////////////////////////////////////////
/// node destructor

TMVA::Node::~Node()
{
   fgCount--;
}

////////////////////////////////////////////////////////////////////////////////
/// returns the global number of instantiated nodes

int TMVA::Node::GetCount()
{
   return fgCount;
}

////////////////////////////////////////////////////////////////////////////////
///recursively go through the part of the tree below this node and count all daughters

Int_t TMVA::Node::CountMeAndAllDaughters() const
{
   Int_t n=1;
   if (this->GetLeft() != NULL)
      n+= this->GetLeft()->CountMeAndAllDaughters();
   if (this->GetRight() != NULL)
      n+= this->GetRight()->CountMeAndAllDaughters();

   return n;
}

// print a node
////////////////////////////////////////////////////////////////////////////////
/// output operator for a node

std::ostream& TMVA::operator<<( std::ostream& os, const TMVA::Node& node )
{
   node.Print(os);
   return os;                // Return the output stream.
}

////////////////////////////////////////////////////////////////////////////////
/// output operator with a pointer to the node (which still prints the node itself)

std::ostream& TMVA::operator<<( std::ostream& os, const TMVA::Node* node )
{
   if (node!=NULL) node->Print(os);
   return os;                // Return the output stream.
}

////////////////////////////////////////////////////////////////////////////////
/// add attributes to XML

void* TMVA::Node::AddXMLTo( void* parent ) const
{
   std::stringstream s("");
   AddContentToNode(s);
   void* node = gTools().AddChild(parent, "Node", s.str().c_str());
   gTools().AddAttr( node, "pos",   fPos );
   gTools().AddAttr( node, "depth", fDepth );
   this->AddAttributesToNode(node);
   if (this->GetLeft())  this->GetLeft()->AddXMLTo(node);
   if (this->GetRight()) this->GetRight()->AddXMLTo(node);
   return node;
}

////////////////////////////////////////////////////////////////////////////////
/// read attributes from XML

void TMVA::Node::ReadXML( void* node,  UInt_t tmva_Version_Code )
{
   ReadAttributes(node, tmva_Version_Code);
   const char* content = gTools().GetContent(node);
   if (content) {
      std::stringstream s(content);
      ReadContent(s);
   }
   gTools().ReadAttr( node, "pos",   fPos );
   gTools().ReadAttr( node, "depth", fDepth );

   void* ch = gTools().GetChild(node);
   while (ch) {
      Node* n = CreateNode();
      n->ReadXML(ch, tmva_Version_Code);
      if (n->GetPos()=='l')     { this->SetLeft(n);  }
      else if(n->GetPos()=='r') { this->SetRight(n); }
      else {
         std::cout << "neither left nor right" << std::endl;
      }
      ch = gTools().GetNextChild(ch);
   }
}
