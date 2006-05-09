// @(#)root/tmva $Id: TMVA_DecisionTreeNode.cxx,v 1.2 2006/05/08 17:56:50 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_DecisionTreeNode                                                 *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of a Decision Tree Node                                    *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_DecisionTreeNode.cxx,v 1.2 2006/05/08 17:56:50 brun Exp $    
 **********************************************************************************/
   
//_______________________________________________________________________
//                                                                      
// Node for the Decision Tree                                           
//                                                                      
//_______________________________________________________________________

#include <iostream>
#include <algorithm>

#include "TMVA_DecisionTreeNode.h"
#include "TMVA_BinarySearchTree.h"
#include "TMVA_Tools.h"

using std::string;

ClassImp(TMVA_DecisionTreeNode)

//_______________________________________________________________________
TMVA_DecisionTreeNode::TMVA_DecisionTreeNode(TMVA_Event* e)
  : TMVA_Node(e) 
{
  fCutMin=0;
  fCutMax=0;
  fCutType= kTRUE;
  
  fSoverSB=-1;
  fSeparationIndex=-1;
  fSeparationGain=-1;
  fNEvents = -1;
  fNodeType=-99;

}

//_______________________________________________________________________
TMVA_DecisionTreeNode::TMVA_DecisionTreeNode(TMVA_Node* p)
  : TMVA_Node(p) 
{
  fCutMin=0;
  fCutMax=0;
  fCutType= kTRUE;
  
  fSoverSB=-1;
  fSeparationIndex=-1;
  fSeparationGain=-1;
  fNEvents = -1;
  fNodeType=-99;
}

//_______________________________________________________________________
Bool_t TMVA_DecisionTreeNode::GoesRight(const TMVA_Event * e) const
{
  Bool_t result;
  
  result =  (e->GetData(this->GetSelector()) > this->GetCutMin() && 
             e->GetData(this->GetSelector()) <= this->GetCutMax() ); 
  
  if (fCutType == kTRUE) return result; //the cuts are selecting Signal ;
  else return !result;

}

//_______________________________________________________________________
Bool_t TMVA_DecisionTreeNode::GoesLeft(const TMVA_Event * e) const
{
  if (!this->GoesRight(e)) return kTRUE;
  else return kFALSE;
}

//_______________________________________________________________________
void TMVA_DecisionTreeNode::PrintRec(ostream& os, const Int_t Depth, const string pos ) const
{
  os << Depth << " " << pos << " ivar: " <<  this->GetSelector()
     << " mn: " << this->GetCutMin() 
     << " mx: " << this->GetCutMax() 
     << " cType: " << this->GetCutType() 
     << " pur: " << this->GetSoverSB()
     << " sepI: " << this->GetSeparationIndex()
     << " sepG: " << this->GetSeparationGain()
     << " nEv: " << this->GetNEvents()
     << " nType: " << this->GetNodeType()<<endl;
  
  if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os,Depth+1,"l") ;
  if(this->GetRight() != NULL)this->GetRight()->PrintRec(os,Depth+1,"r");
}

//_______________________________________________________________________
TMVA_NodeID TMVA_DecisionTreeNode::ReadRec(ifstream& is, TMVA_NodeID nodeID, TMVA_Node* Parent )
{
  string tmp;
  Double_t dtmp1, dtmp2, dtmp3, dtmp4, dtmp5, dtmp6, dtmp7;
  Int_t itmp, itmp1, itmp2;
  string pos;
  TMVA_NodeID nextNodeID;
  if (Parent==NULL) {
    is >> itmp >> pos ;
    nodeID.SetDepth(itmp);
    nodeID.SetPos(pos);
  }
  
  is >> tmp >> itmp1 >> tmp >> dtmp1 >> tmp >> dtmp2 >> tmp >> dtmp3
     >> tmp >> dtmp4 >> tmp >> dtmp5 >> tmp >> dtmp6 >> tmp >> dtmp7 
     >> tmp >> itmp2;
  this->SetSelector(itmp1);
  this->SetCutMin(dtmp1);
  this->SetCutMax(dtmp2);
  this->SetCutType(dtmp3);
  this->SetSoverSB(dtmp4);
  this->SetSeparationIndex(dtmp5);
  this->SetSeparationGain(dtmp6);
  this->SetNEvents(dtmp7);
  this->SetNodeType(itmp2);

  is >> itmp >> pos ;
  nextNodeID.SetDepth(itmp);
  nextNodeID.SetPos(pos);
 
  if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
    if (nextNodeID.GetPos()=="l") {
      this->SetLeft(new TMVA_DecisionTreeNode());
      this->GetLeft()->SetParent(this);
      nextNodeID = this->GetLeft()->ReadRec(is,nextNodeID,this);
    }
  }
  if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
    if (nextNodeID.GetPos()=="r") {
      this->SetRight(new TMVA_DecisionTreeNode());
      this->GetRight()->SetParent(this);
      nextNodeID = this->GetRight()->ReadRec(is,nextNodeID,this);
    }
  }
  return nextNodeID;
}


