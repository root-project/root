// @(#)root/tmva $Id: DecisionTreeNode.cxx,v 1.7 2006/05/23 09:53:10 stelzer Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DecisionTreeNode                                                *
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
 **********************************************************************************/
   
//_______________________________________________________________________
//                                                                      
// Node for the Decision Tree                                           
//
// The node specifies ONE variable out of the given set of selection variable
// that is used to split the sample which "arrives" at the node, into a left
// (background-enhanced) and a right (signal-enhanced) sample.                                                                       
//_______________________________________________________________________

#include <iostream>
#include <algorithm>

#include "TMVA/DecisionTreeNode.h"
#include "TMVA/BinarySearchTree.h"
#include "TMVA/Tools.h"

using std::string;

ClassImp(TMVA::DecisionTreeNode)

//_______________________________________________________________________
   TMVA::DecisionTreeNode::DecisionTreeNode(TMVA::Event* e)
      : TMVA::Node(e) 
{
   // constructor of an essentially "empty" node floating in space
   fCutValue=0;
   fCutType= kTRUE;
  
   fSoverSB=-1;
   fSeparationIndex=-1;
   fSeparationGain=-1;
   fNEvents = -1;
   fNodeType=-99;
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(TMVA::Node* p)
   : TMVA::Node(p) 
{
   // constructor of a daughter node as a daughter of 'p'
   fCutValue=0;
   fCutType= kTRUE;
  
   fSoverSB=-1;
   fSeparationIndex=-1;
   fSeparationGain=-1;
   fNEvents = -1;
   fNodeType=-99;
}

//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesRight(const TMVA::Event * e) const
{
   // test event if it decends the tree at this node to the right  
   Bool_t result;
  
   result =  (e->GetData(this->GetSelector()) > this->GetCutValue() );
  
   if (fCutType == kTRUE) return result; //the cuts are selecting Signal ;
   else return !result;

}

//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesLeft(const TMVA::Event * e) const
{
   // test event if it decends the tree at this node to the left 
   if (!this->GoesRight(e)) return kTRUE;
   else return kFALSE;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRec(ostream& os, const Int_t Depth, const string pos ) const
{
   //recursively print the node and its daughters (--> print the 'tree')

   os << Depth << " " << pos << " ivar: " <<  this->GetSelector()
      << " cut: " << this->GetCutValue() 
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
TMVA::NodeID TMVA::DecisionTreeNode::ReadRec(ifstream& is, TMVA::NodeID nodeID, TMVA::Node* Parent )
{
   //recursively read the node and its daughters (--> read the 'tree')
   string tmp;
   Double_t dtmp1, dtmp2, dtmp3, dtmp4, dtmp5, dtmp6;
   Int_t itmp, itmp1, itmp2;
   string pos;
   TMVA::NodeID nextNodeID;
   if (Parent==NULL) {
      is >> itmp >> pos ;
      nodeID.SetDepth(itmp);
      nodeID.SetPos(pos);
   }
  
   is >> tmp >> itmp1 >> tmp >> dtmp1 >> tmp >> dtmp2 >> tmp >> dtmp3
      >> tmp >> dtmp4 >> tmp >> dtmp5 >> tmp >> dtmp6 
      >> tmp >> itmp2;
   this->SetSelector(itmp1);
   this->SetCutValue(dtmp1);
   this->SetCutType(dtmp2);
   this->SetSoverSB(dtmp3);
   this->SetSeparationIndex(dtmp4);
   this->SetSeparationGain(dtmp5);
   this->SetNEvents(dtmp6);
   this->SetNodeType(itmp2);

  is >> itmp >> pos ;
  nextNodeID.SetDepth(itmp);
  nextNodeID.SetPos(pos);
 
  if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
    if (nextNodeID.GetPos()=="l") {
      this->SetLeft(new TMVA::DecisionTreeNode());
      this->GetLeft()->SetParent(this);
      nextNodeID = this->GetLeft()->ReadRec(is,nextNodeID,this);
    }
  }
  if (nextNodeID.GetDepth() == nodeID.GetDepth()+1){
    if (nextNodeID.GetPos()=="r") {
      this->SetRight(new TMVA::DecisionTreeNode());
      this->GetRight()->SetParent(this);
      nextNodeID = this->GetRight()->ReadRec(is,nextNodeID,this);
    }
  }
  return nextNodeID;
}


