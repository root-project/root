// @(#)root/tmva $Id: BinarySearchTreeNode.cxx,v 1.4 2006/11/13 15:49:49 helgevoss Exp $    
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * CopyRight (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
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
#include "TMVA/Event.h"

ClassImp(TMVA::BinarySearchTreeNode)
   ;

//_______________________________________________________________________
TMVA::BinarySearchTreeNode::BinarySearchTreeNode ( const BinarySearchTreeNode  &n,
                                                   BinarySearchTreeNode* parent) : 
   TMVA::Node(n),
   fEvent ( n.fEvent==NULL ? NULL : new Event(*(n.fEvent)) ),
   fEventOwnership (kTRUE),
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
TMVA::BinarySearchTreeNode::~BinarySearchTreeNode ( void )
{
   // node destructor
   if (this->GetEventOwnership()) delete fEvent;
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::GoesRight(const TMVA::Event& e) const {
   // check if the event fed into the node goes/decends to the right daughter
   if (e.GetVal(fSelector) > fEvent->GetVal(fSelector)) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::GoesLeft(const TMVA::Event& e) const
{
   // check if the event fed into the node goes/decends to the left daughter
   if (e.GetVal(fSelector) <= fEvent->GetVal(fSelector)) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::BinarySearchTreeNode::EqualsMe(const TMVA::Event& e) const
{
   // check if the event fed into the node actually equals the event
   // that forms the node (in case of a search tree)
   Bool_t result = true;
   for (UInt_t i=0; i<fEvent->GetNVars(); i++) {
      result&= (e.GetVal(i) == fEvent->GetVal(i));
   }
   return result;
}


// print a node
//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::Print(ostream& os) const
{
   //print the node
   os << "< ***  " <<endl << " node.Data: " << this->GetEvent() 
      << "at address " <<long(this->GetEvent()) <<endl;
   os << "Selector: " <<  this->GetSelector() <<endl;
   os << "My address is " << long(this) << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: " << long(this->GetParent()) ;
   if (this->GetLeft() != NULL) os << " left daughter at addr: " << long(this->GetLeft());
   if (this->GetRight() != NULL) os << " right daughter at addr: " << long(this->GetRight()) ;
   
   os << " **** > "<< endl;
}

//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::PrintRec(ostream& os ) const
{
   //recursively print the node and its daughters (--> print the 'tree')
   os << this->GetDepth() << " " << this->GetPos() 
      << " node.Data: " <<  endl << this->GetEvent() ;
   os << this->GetDepth() << " " << this->GetPos() 
      << " Selector: " <<  this->GetSelector() <<endl;
   if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os) ;
   if(this->GetRight() != NULL)this->GetRight()->PrintRec(os);
}

//_______________________________________________________________________
void TMVA::BinarySearchTreeNode::ReadRec(istream& is,  char &pos, UInt_t &depth,
                                         TMVA::Node* parent )
{
   //recursively read the node and its daughters (--> print the 'tree')
   std::string tmp;
   Int_t itmp;

   if (parent==NULL) {
      is >> itmp >> pos ;
      this->SetDepth(itmp);
      this->SetPos(pos);
   } else {
      this->SetDepth(depth);
      this->SetPos(pos);
   }

   is >> tmp;

   
   std::cout  << kFATAL << " Cannot read revents yet as the constuctor would need "
              << " know about the VariableInfo" << std::endl; exit(1);

   //TMVA::Event* e = new TMVA::Event();
   this->SetEventOwnership(kTRUE);
   // read the event
   Double_t dtmp;
   Int_t nvar;
   is >> tmp >> tmp >> nvar >> tmp >> tmp >> tmp >> dtmp;
   // e->SetWeight(dtmp);
   for (int i=0; i<nvar; i++){
      is >> dtmp; 
      //      e->SetVal(i,dtmp);
   }
   //   this->SetEvent(e);

   is >> itmp >> pos >> tmp >> tmp;
   this->SetSelector((UInt_t)atoi(tmp.c_str()));

   is >> depth >> pos;

   if (depth == this->GetDepth()+1){
      if (pos=='l') {
         this->SetLeft(new TMVA::BinarySearchTreeNode() );
         this->GetLeft()->SetParent(this);
         this->GetLeft()->ReadRec(is,pos,depth,this);
      }
   }
   if (depth == this->GetDepth()+1){
      if (pos=='r') {
         this->SetRight(new TMVA::BinarySearchTreeNode());
         this->GetRight()->SetParent(this);
         this->GetRight()->ReadRec(is,pos,depth,this);
      }
   }
}



