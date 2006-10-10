// @(#)root/tmva $Id: Node.cxx,v 1.3 2006/05/23 19:35:06 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Classes: TMVA::Node, TMVA::NodeID                                              *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
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

#include <string>
#include "TMVA/Node.h"
#include "Riostream.h"
#include <stdexcept>
#include "TMVA/Event.h"

ClassImp(TMVA::Node)

//_______________________________________________________________________
TMVA::Node::~Node( void )
{
   // node destructor
   if (this->GetEventOwnership()) delete fEvent;
}

//_______________________________________________________________________
Bool_t TMVA::Node::GoesRight(const TMVA::Event * e) const {
   // check if the event fed into the node goes/decends to the right daughter
   if (e->GetData(fSelector) > fEvent->GetData(fSelector)) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::Node::GoesLeft(const TMVA::Event * e) const
{
   // check if the event fed into the node goes/decends to the left daughter
   if (e->GetData(fSelector) <= fEvent->GetData(fSelector)) return true;
   else return false;
}

//_______________________________________________________________________
Bool_t TMVA::Node::EqualsMe(const TMVA::Event * e) const
{
   // check if the event fed into the node actually equals the event
   // that forms the node (in case of a search tree)
   Bool_t result = true;
   for (Int_t i=0; i<fEvent->GetEventSize(); i++) {
      result &= (e->GetData(i) == fEvent->GetData(i));
   }
   return result;
}

//_______________________________________________________________________
Int_t TMVA::Node::CountMeAndAllDaughters( void ) const 
{
   //recursively go through the part of the tree below this node and count all daughters
   Int_t n=1;
   if(this->GetLeft() != NULL) 
      n+= this->GetLeft()->CountMeAndAllDaughters(); 
   if(this->GetRight() != NULL) 
      n+= this->GetRight()->CountMeAndAllDaughters(); 
  
   return n;
}

// print a node
//_______________________________________________________________________
void TMVA::Node::Print(ostream& os) const
{
   //print the node
   os << "node.Data: " <<  endl << this->GetData() <<endl;
   os << "Selector: " <<  this->GetSelector() <<endl;
   os << "  address: "<< this 
      << "  Parent: " << this->GetParent() 
      << "  Left: " <<  this->GetLeft() 
      << "  Right: " << this->GetRight()
      << endl;
}

//_______________________________________________________________________
void TMVA::Node::PrintRec(ostream& os, const Int_t Depth, const std::string pos ) const
{
   //recursively print the node and its daughters (--> print the 'tree')
   os << Depth << " " << pos << " node.Data: " <<  endl << this->GetData() 
      << endl;
   os << Depth << " " << pos << " Selector: " <<  this->GetSelector() <<endl;
   if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os,Depth+1,"Left") ;
   if(this->GetRight() != NULL)this->GetRight()->PrintRec(os,Depth+1,"Right");
}

//_______________________________________________________________________
TMVA::NodeID TMVA::Node::ReadRec(ifstream& is, TMVA::NodeID nodeID, TMVA::Node* Parent )
{
   //recursively read the node and its daughters (--> print the 'tree')
   std::string tmp;
   Int_t itmp;
   std::string pos;
   TMVA::NodeID nextNodeID;
   if (Parent==NULL) {
      is >> itmp >> pos >> tmp;
      nodeID.SetDepth(itmp);
      nodeID.SetPos(pos);
   }
   else is >> tmp;

   TMVA::Event* e=new TMVA::Event();
   this->SetEventOwnership(kTRUE);
   e->Read(is);
   this->SetData(e);
   is >> itmp >> pos >> tmp >> tmp;
   this->SetSelector(atoi(tmp.c_str()));
   is >> itmp >> pos;
   nextNodeID.SetDepth(itmp);
   nextNodeID.SetPos(pos);
 
   if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
      if (nextNodeID.GetPos()=="Left") {
         this->SetLeft(new TMVA::Node());
         this->GetLeft()->SetParent(this);
         nextNodeID = this->GetLeft()->ReadRec(is,nextNodeID,this);
      }
   }
   if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
      if (nextNodeID.GetPos()=="Right") {
         this->SetRight(new TMVA::Node());
         this->GetRight()->SetParent(this);
         nextNodeID = this->GetRight()->ReadRec(is,nextNodeID,this);
      }
   }
   return nextNodeID;
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



