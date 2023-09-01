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

/*! \class TMVA::BinarySearchTreeNode
\ingroup TMVA

Node for the BinarySearch or Decision Trees

for the binary search tree, it basically consists of the EVENT, and
pointers to the parent and daughters

in case of the Decision Tree, it specifies parent and daughters, as
well as "which variable is used" in the selection of this node, including
the respective cut value.
*/

#include <stdexcept>
#include <iomanip>
#include <assert.h>
#include <cstdlib>

#include "TString.h"

#include "TMVA/BinarySearchTreeNode.h"
#include "TMVA/Event.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Node.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::BinarySearchTreeNode);

////////////////////////////////////////////////////////////////////////////////
/// constructor of a node for the search tree

TMVA::BinarySearchTreeNode::BinarySearchTreeNode( const Event* e, UInt_t /* signalClass */ )
: TMVA::Node(),
   fEventV  ( std::vector<Float_t>() ),
   fTargets ( std::vector<Float_t>() ),
   fWeight  ( e==0?0:e->GetWeight()  ),
   fClass   ( e==0?0:e->GetClass() ), // see BinarySearchTree.h, line Mean() RMS() Min() and Max()
   fSelector( -1 )
{
   if (e!=0) {
      for (UInt_t ivar=0; ivar<e->GetNVariables(); ivar++) fEventV.push_back(e->GetValue(ivar));
      for (std::vector<Float_t>::const_iterator it = e->GetTargets().begin(); it < e->GetTargets().end(); ++it ) {
         fTargets.push_back( (*it) );
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// constructor of a daughter node as a daughter of 'p'

TMVA::BinarySearchTreeNode::BinarySearchTreeNode( BinarySearchTreeNode* parent, char pos ) :
   TMVA::Node(parent,pos),
   fEventV  ( std::vector<Float_t>() ),
   fTargets ( std::vector<Float_t>() ),
   fWeight  ( 0  ),
   fClass   ( 0 ),
   fSelector( -1 )
{
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor of a node. It will result in an explicit copy of
/// the node and recursively all it's daughters

TMVA::BinarySearchTreeNode::BinarySearchTreeNode ( const BinarySearchTreeNode &n,
                                                   BinarySearchTreeNode* parent ) :
   TMVA::Node(n),
   fEventV  ( n.fEventV   ),
   fTargets ( n.fTargets  ),
   fWeight  ( n.fWeight   ),
   fClass   ( n.fClass    ),
   fSelector( n.fSelector )
{
   this->SetParent( parent );
   if (n.GetLeft() == 0 ) this->SetLeft(NULL);
   else this->SetLeft( new BinarySearchTreeNode( *((BinarySearchTreeNode*)(n.GetLeft())),this));

   if (n.GetRight() == 0 ) this->SetRight(NULL);
   else this->SetRight( new BinarySearchTreeNode( *((BinarySearchTreeNode*)(n.GetRight())),this));

}

////////////////////////////////////////////////////////////////////////////////
/// node destructor

TMVA::BinarySearchTreeNode::~BinarySearchTreeNode()
{
}

////////////////////////////////////////////////////////////////////////////////
/// check if the event fed into the node goes/descends to the right daughter

Bool_t TMVA::BinarySearchTreeNode::GoesRight( const TMVA::Event& e ) const
{
   if (e.GetValue(fSelector) > GetEventV()[fSelector]) return true;
   else return false;
}

////////////////////////////////////////////////////////////////////////////////
/// check if the event fed into the node goes/descends to the left daughter

Bool_t TMVA::BinarySearchTreeNode::GoesLeft(const TMVA::Event& e) const
{
   if (e.GetValue(fSelector) <= GetEventV()[fSelector]) return true;
   else return false;
}

////////////////////////////////////////////////////////////////////////////////
/// check if the event fed into the node actually equals the event
/// that forms the node (in case of a search tree)

Bool_t TMVA::BinarySearchTreeNode::EqualsMe(const TMVA::Event& e) const
{
   Bool_t result = true;
   for (UInt_t i=0; i<GetEventV().size(); i++) {
      result&= (e.GetValue(i) == GetEventV()[i]);
   }
   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// print the node

void TMVA::BinarySearchTreeNode::Print( std::ostream& os ) const
{
   os << "< ***  " << std::endl << " node.Data: ";
   std::vector<Float_t>::const_iterator it=fEventV.begin();
   os << fEventV.size() << " vars: ";
   for (;it!=fEventV.end(); ++it) os << " " << std::setw(10) << *it;
   os << "  EvtWeight " << std::setw(10) << fWeight;
   os << std::setw(10) << "Class: " << GetClass() << std::endl;

   os << "Selector: " <<  this->GetSelector() <<std::endl;
   os << "My address is " << (Longptr_t)this << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: " << (Longptr_t)this->GetParent();
   if (this->GetLeft() != NULL) os << " left daughter at addr: " << (Longptr_t)this->GetLeft();
   if (this->GetRight() != NULL) os << " right daughter at addr: " << (Longptr_t)this->GetRight();

   os << " **** > "<< std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// recursively print the node and its daughters (--> print the 'tree')

void TMVA::BinarySearchTreeNode::PrintRec( std::ostream& os ) const
{
   os << this->GetDepth() << " " << this->GetPos() << " " << this->GetSelector()
      << " data: " <<  std::endl;
   std::vector<Float_t>::const_iterator it=fEventV.begin();
   os << fEventV.size() << " vars: ";
   for (;it!=fEventV.end(); ++it) os << " " << std::setw(10) << *it;
   os << "  EvtWeight " << std::setw(10) << fWeight;
   os << std::setw(10) << "Class: " << GetClass() << std::endl;

   if (this->GetLeft() != NULL)this->GetLeft()->PrintRec(os) ;
   if (this->GetRight() != NULL)this->GetRight()->PrintRec(os);
}

////////////////////////////////////////////////////////////////////////////////
/// Read the data block

Bool_t TMVA::BinarySearchTreeNode::ReadDataRecord( std::istream& is, UInt_t /* Tmva_Version_Code */  )
{
   Int_t       itmp;
   std::string tmp;
   UInt_t      depth, selIdx, nvar;
   Char_t      pos;
   TString     sigbkgd;
   Float_t     evtValFloat;

   // read depth and position
   is >> itmp;
   if ( itmp==-1 ) { return kFALSE; } // Done

   depth=(UInt_t)itmp;
   is >> pos >> selIdx;
   this->SetDepth(depth);     // depth of the tree
   this->SetPos(pos);         // either 's' (root node), 'l', or 'r'
   this->SetSelector(selIdx);

   // next line: read and build the event
   // coverity[tainted_data_argument]
   is >> nvar;
   fEventV.clear();
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      is >> evtValFloat; fEventV.push_back(evtValFloat);
   }
   is >> tmp >> fWeight;
   is >> sigbkgd;
   fClass = (sigbkgd=="S" || sigbkgd=="Signal")?0:1;

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// read attributes from XML

void TMVA::BinarySearchTreeNode::ReadAttributes(void* node, UInt_t /* tmva_Version_Code */ )
{
   gTools().ReadAttr(node, "selector", fSelector );
   gTools().ReadAttr(node, "weight",   fWeight );
   std::string sb;
   gTools().ReadAttr(node, "type",     sb);
   if (sb=="Signal" || sb=="0")
      fClass=0;
   if (sb=="1")
      fClass=1;
   //   fClass = (sb=="Signal")?0:1;
   Int_t nvars;
   gTools().ReadAttr(node, "NVars",nvars);
   fEventV.resize(nvars);
}


////////////////////////////////////////////////////////////////////////////////
/// adding attributes to tree node

void TMVA::BinarySearchTreeNode::AddAttributesToNode(void* node) const {
   gTools().AddAttr(node, "selector", fSelector );
   gTools().AddAttr(node, "weight", fWeight );
   //   gTools().AddAttr(node, "type", (IsSignal()?"Signal":"Background"));
   gTools().AddAttr(node, "type", GetClass());
   gTools().AddAttr(node, "NVars", fEventV.size());
}


////////////////////////////////////////////////////////////////////////////////
/// adding attributes to tree node

void TMVA::BinarySearchTreeNode::AddContentToNode( std::stringstream& s ) const
{
   std::ios_base::fmtflags ff = s.flags();
   s.precision( 16 );
   for (UInt_t i=0; i<fEventV.size();  i++) s << std::scientific << " " << fEventV[i];
   for (UInt_t i=0; i<fTargets.size(); i++) s << std::scientific << " " << fTargets[i];
   s.flags(ff);
}
////////////////////////////////////////////////////////////////////////////////
/// read events from node

void TMVA::BinarySearchTreeNode::ReadContent( std::stringstream& s )
{
   Float_t temp=0;
   for (UInt_t i=0; i<fEventV.size(); i++){
      s >> temp;
      fEventV[i]=temp;
   }
   while (s >> temp) fTargets.push_back(temp);
}
