// @(#)root/tmva $Id: BinarySearchTreeNode.cxx,v 1.5 2007/04/19 06:53:01 brun Exp $    
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

#include "TMVA/BinarySearchTreeNode.h"
#include "Riostream.h"
#include <stdexcept>
#include <assert.h>
#include "TMVA/Event.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::BinarySearchTreeNode)


//_______________________________________________________________________
TMVA::BinarySearchTreeNode::BinarySearchTreeNode( Event* e ) : 
   TMVA::Node(),
   fEventV(std::vector<Float_t>()),
   fWeight(e==0?0:e->GetWeight()),
   fIsSignal(e==0?kTRUE:e->IsSignal()),
   fSelector( -1 )
{
   // constructor of a node for the search tree
   if(e!=0) 
      for(UInt_t ivar=0; ivar<e->GetNVars(); ivar++)
         fEventV.push_back(e->GetVal(ivar));
}

//_______________________________________________________________________
TMVA::BinarySearchTreeNode::BinarySearchTreeNode( BinarySearchTreeNode* parent, char pos ) : 
   TMVA::Node(parent,pos),
   fEventV(std::vector<Float_t>()),
   fWeight(0),
   fIsSignal(kTRUE),
   fSelector( -1 )
{
   // constructor of a daughter node as a daughter of 'p'

}



//_______________________________________________________________________
TMVA::BinarySearchTreeNode::BinarySearchTreeNode ( const BinarySearchTreeNode &n,
                                                   BinarySearchTreeNode* parent ) : 
   TMVA::Node(n),
   fEventV(n.fEventV),
   fWeight(n.fWeight),
   fIsSignal(n.fIsSignal),
   fSelector (n.fSelector)
{
   // copy constructor of a node. It will result in an explicit copy of
   // the node an drecursively all it's daughters
   this->SetParent( parent );
   if (n.GetLeft() == 0 ) this->SetLeft(NULL);
   else this->SetLeft( new BinarySearchTreeNode( *((BinarySearchTreeNode*)(n.GetLeft())),this));
   
   if (n.GetRight() == 0 ) this->SetRight(NULL);
   else this->SetRight( new BinarySearchTreeNode( *((BinarySearchTreeNode*)(n.GetRight())),this));

}

//_______________________________________________________________________
TMVA::BinarySearchTreeNode::~BinarySearchTreeNode()
{
   // node destructor
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::GoesRight( const TMVA::Event& e ) const 
{
   // check if the event fed into the node goes/decends to the right daughter
   if (e.GetVal(fSelector) > GetEventV()[fSelector]) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::GoesLeft(const TMVA::Event& e) const
{
   // check if the event fed into the node goes/decends to the left daughter
   if (e.GetVal(fSelector) <= GetEventV()[fSelector]) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::EqualsMe(const TMVA::Event& e) const
{
   // check if the event fed into the node actually equals the event
   // that forms the node (in case of a search tree)
   Bool_t result = true;
   for (UInt_t i=0; i<GetEventV().size(); i++) {
      result&= (e.GetVal(i) == GetEventV()[i]);
   }
   return result;
}

//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::Print( ostream& os ) const
{
   // print the node
   os << "< ***  " <<endl << " node.Data: ";
   std::vector<Float_t>::const_iterator it=fEventV.begin();
   os << fEventV.size() << " vars: ";
   for(;it!=fEventV.end(); it++) os << " " << std::setw(10) << *it;
   os << "  EvtWeight " << std::setw(10) << fWeight;
   os << std::setw(10) << (fIsSignal?" Signal":" Background") << endl;

   os << "Selector: " <<  this->GetSelector() <<endl;
   os << "My address is " << long(this) << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: " << long(this->GetParent()) ;
   if (this->GetLeft() != NULL) os << " left daughter at addr: " << long(this->GetLeft());
   if (this->GetRight() != NULL) os << " right daughter at addr: " << long(this->GetRight()) ;
   
   os << " **** > "<< endl;
}

//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::PrintRec( ostream& os ) const
{
   // recursively print the node and its daughters (--> print the 'tree')
   os << this->GetDepth() << " " << this->GetPos() << " " << this->GetSelector()
      << " data: " <<  endl;
   std::vector<Float_t>::const_iterator it=fEventV.begin();
   os << fEventV.size() << " vars: ";
   for(;it!=fEventV.end(); it++) os << " " << std::setw(10) << *it;
   os << "  EvtWeight " << std::setw(10) << fWeight;
   os << std::setw(10) << (fIsSignal?" Signal":" Background") << endl;

   if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os) ;
   if(this->GetRight() != NULL)this->GetRight()->PrintRec(os);
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::ReadDataRecord( istream& is ) 
{
   // Read the data block

   std::string tmp;
   UInt_t      depth;
   char        pos;
   TString     sigbkgd;
   Float_t     evtValFloat;
   //   Int_t       evtValInt;
   //   Float_t     evtWeight;
   UInt_t      nvar;
   Int_t       itmp;
   UInt_t      selIdx;

   // read depth and position
   is >> itmp;
   if( itmp==-1 ) { delete this; return kFALSE; }
   depth = itmp;
   is >> pos >> selIdx >> tmp;
   this->SetDepth(itmp);   // depth of the tree
   this->SetPos(pos);      // either 's' (root node), 'l', or 'r'
   this->SetSelector(selIdx);

   // read and build the event
   is >> nvar >> tmp;
   fEventV.clear();
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      is >> evtValFloat;
      fEventV.push_back(evtValFloat);
//       if(GetEvent()->IsFloat(ivar)) {
//          is >> evtValFloat;
//          GetEvent()->SetValFloatNoCheck(ivar, evtValFloat);
//          fEventV.push_back(evtValFloat);
//       } else {
//          is >> evtValInt;
//          GetEvent()->SetValIntNoCheck(ivar, evtValInt);
//          fEventV.push_back((Float_t)evtValInt);
//       }
   }
   is >> tmp >> fWeight;
   //   GetEvent()->SetWeight(fWeight);
   is >> sigbkgd;
   //   GetEvent()->SetType((sigbkgd=="Signal")?Types::kSignal:Types::kBackground);
   fIsSignal = (sigbkgd=="Signal");
   // -------------------------

   return kTRUE;
}

//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::ReadRec( istream& is,  char &pos, UInt_t &depth,
                                          TMVA::Node* parent )
{
   // recursively read the node and its daughters (--> print the 'tree')

   if( ! ReadDataRecord(is) ) return;

   depth = GetDepth();
   pos   = GetPos();

   // find parent node
   while( parent!=0 && parent->GetDepth() != GetDepth()-1) parent=parent->GetParent();

   if(parent!=0) {
      SetParent(parent);
      if(GetPos()=='l') parent->SetLeft(this);
      if(GetPos()=='r') parent->SetRight(this);
   }
   
   char childPos;
   UInt_t childDepth;
   TMVA::Node * newNode = new TMVA::BinarySearchTreeNode(  );
   newNode->ReadRec(is, childPos, childDepth, this);
}



//_______________________________________________________________________
Int_t TMVA::BinarySearchTreeNode::GetMemSize() const { 
   // Calculate the size of the node in memory
   Int_t size = sizeof(*this);
   if(GetLeft()!=0) size += GetLeft()->GetMemSize();
   if(GetRight()!=0) size += GetRight()->GetMemSize();
   return size;
}
