// @(#)root/tmva $Id: TMVA_Node.cpp,v 1.9 2006/05/03 19:45:38 helgevoss Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Classes: TMVA_Node, TMVA_NodeID                                                *
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
//_______________________________________________________________________

#include <string>
#include "TMVA_Node.h"
#include "Riostream.h"
#include <stdexcept>
#include "TMVA_Event.h"

#define DEBUG_TMVA_BinaryTree kFALSE

ClassImp(TMVA_Node)

//_______________________________________________________________________
TMVA_Node::~TMVA_Node( void )
{
  if (this->GetEventOwnership()) delete fEvent;
}

//_______________________________________________________________________
Bool_t TMVA_Node::GoesRight(const TMVA_Event * e) const{
  if (e->GetData(fSelector) > fEvent->GetData(fSelector)) return true;
  else return false;
}

//_______________________________________________________________________
Bool_t TMVA_Node::GoesLeft(const TMVA_Event * e) const
{
  if (e->GetData(fSelector) <= fEvent->GetData(fSelector)) return true;
  else return false;
}

//_______________________________________________________________________
Bool_t TMVA_Node::EqualsMe(const TMVA_Event * e) const
{
  Bool_t result = true;
  for (Int_t i=0; i<fEvent->GetEventSize(); i++) {
    result *= (e->GetData(i) == fEvent->GetData(i));
  }
  return result;
}

//_______________________________________________________________________
Int_t TMVA_Node::CountMeAndAllDaughters( void ) const 
{
  Int_t n=1;
  if(this->GetLeft() != NULL) 
    n+= this->GetLeft()->CountMeAndAllDaughters(); 
  if(this->GetRight() != NULL) 
    n+= this->GetRight()->CountMeAndAllDaughters(); 
  
  return n;
}

// print a node
//_______________________________________________________________________
void TMVA_Node::Print(ostream& os) const
{
  os << "node.Data: " <<  endl << this->GetData() <<endl;
  os << "Selector: " <<  this->GetSelector() <<endl;
  os << "  address: "<< this 
     << "  Parent: " << this->GetParent() 
     << "  Left: " <<  this->GetLeft() 
     << "  Right: " << this->GetRight()
     << endl;
}

//_______________________________________________________________________
void TMVA_Node::PrintRec(ostream& os, const Int_t Depth, const std::string pos ) const
{
  os << Depth << " " << pos << " node.Data: " <<  endl << this->GetData() 
     << endl;
  os << Depth << " " << pos << " Selector: " <<  this->GetSelector() <<endl;
  if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os,Depth+1,"Left") ;
  if(this->GetRight() != NULL)this->GetRight()->PrintRec(os,Depth+1,"Right");
}

//_______________________________________________________________________
TMVA_NodeID TMVA_Node::ReadRec(ifstream& is, TMVA_NodeID nodeID, TMVA_Node* Parent )
{
  std::string tmp;
  Int_t itmp;
  std::string pos;
  TMVA_NodeID nextNodeID;
  if (Parent==NULL) {
    is >> itmp >> pos >> tmp;
    nodeID.SetDepth(itmp);
    nodeID.SetPos(pos);
  }
  else is >> tmp;

  TMVA_Event* e=new TMVA_Event();
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
      this->SetLeft(new TMVA_Node());
      this->GetLeft()->SetParent(this);
      nextNodeID = this->GetLeft()->ReadRec(is,nextNodeID,this);
    }
  }
  if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
    if (nextNodeID.GetPos()=="Right") {
      this->SetRight(new TMVA_Node());
      this->GetRight()->SetParent(this);
      nextNodeID = this->GetRight()->ReadRec(is,nextNodeID,this);
    }
  }
  return nextNodeID;
}

// print a node
//_______________________________________________________________________
ostream& operator<<(ostream& os, const TMVA_Node& node)
{ 
  node.Print(os);
  return os;                // Return the output stream.
}

//_______________________________________________________________________
ostream& operator<<(ostream& os, const TMVA_Node* node)
{ 
  if (node!=NULL) node->Print(os);
  return os;                // Return the output stream.
}



