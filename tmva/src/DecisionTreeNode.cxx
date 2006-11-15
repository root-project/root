// @(#)root/tmva $Id: DecisionTreeNode.cxx,v 1.23 2006/11/14 15:03:46 stelzer Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::DecisionTreeNode                                                *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
   
//_______________________________________________________________________
//                                                                      
// Node for the Decision Tree                                           
//
// The node specifies ONE variable out of the given set of selection variable
// that is used to split the sample which "arrives" at the node, into a left
// (background-enhanced) and a right (signal-enhanced) sample.
//_______________________________________________________________________

#include <algorithm>
#include "Riostream.h"

#include "TMVA/DecisionTreeNode.h"
//#include "TMVA/BinarySearchTree.h"
#include "TMVA/Tools.h"
#include "TMVA/Event.h"

using std::string;

ClassImp(TMVA::DecisionTreeNode)
   ;
   
//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode()
   : TMVA::Node(),
     fCutValue(0),
     fCutType ( kTRUE ),
     fSelector ( -1 ),  
     fNSigEvents ( 0 ),
     fNBkgEvents ( 0 ),
     fNEvents ( -1 ),
     fSeparationIndex (-1 ),
     fSeparationGain ( -1 ),
     fNodeType (-99 ),
     fSequence ( 0 ) 
{
   // constructor of an essentially "empty" node floating in space
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(TMVA::Node* p, char pos)
   : TMVA::Node(p, pos), 
     fCutValue(0),
     fCutType ( kTRUE ),
     fSelector ( -1 ),  
     fNSigEvents ( 0 ),
     fNBkgEvents ( 0 ),
     fNEvents ( -1 ),
     fSeparationIndex (-1 ),
     fSeparationGain ( -1 ),
     fNodeType (-99 ),
     fSequence ( 0 ) 
{
   // constructor of a daughter node as a daughter of 'p'

   // get the sequence, depending on if it is a left or a right daughter
   if (pos == 'r' ){
      ULong_t tmp =1; for (UInt_t i=1; i<this->GetDepth(); i++) {tmp *= 2; }  //  (2^depth) 
      fSequence =  ((DecisionTreeNode*)p)->GetSequence() + tmp;
   } else {
      fSequence =  ((DecisionTreeNode*)p)->GetSequence();
   }      

}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(const TMVA::DecisionTreeNode &n,
                                         DecisionTreeNode* parent)
   : TMVA::Node(n),
     fCutValue( n.fCutValue ),
     fCutType ( n.fCutType ),
     fSelector ( n.fSelector ),  
     fNSigEvents ( n.fNSigEvents ),
     fNBkgEvents ( n.fNBkgEvents ),
     fNEvents ( n.fNEvents ),
     fSeparationIndex ( n.fSeparationIndex ),
     fSeparationGain ( n.fSeparationGain ),
     fNodeType ( n.fNodeType ),
     fSequence ( n.fSequence )  
{
   // copy constructor of a node. It will result in an explicit copy of
   // the node and recursively all it's daughters
   this->SetParent( parent );
   if (n.GetLeft() == 0 ) this->SetLeft(NULL);
   else this->SetLeft( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetLeft())),this));
   
   if (n.GetRight() == 0 ) this->SetRight(NULL);
   else this->SetRight( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetRight())),this));
}


//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesRight(const TMVA::Event & e) const
{
   // test event if it decends the tree at this node to the right  
   Bool_t result;
  
   result =  (e.GetVal(this->GetSelector()) > this->GetCutValue() );
  
   if (fCutType == kTRUE) return result; //the cuts are selecting Signal ;
   else return !result;

}

//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesLeft(const TMVA::Event & e) const
{
   // test event if it decends the tree at this node to the left 
   if (!this->GoesRight(e)) return kTRUE;
   else return kFALSE;
}


//_______________________________________________________________________
Double_t TMVA::DecisionTreeNode::GetSoverSB( void ) const  
{
   // return the S/(S+B) for the node
 return this->GetNSigEvents() / ( this->GetNSigEvents() + this->GetNBkgEvents()); 
}

//_______________________________________________________________________
Double_t TMVA::DecisionTreeNode::GetPurity( void ) const  
{
   // return the purity of the node. that means  S/(S+B) for signal nodes 
   // and B/(S+B) for nodes classified as background
   Double_t p = this->GetNSigEvents() / ( this->GetNSigEvents() + this->GetNBkgEvents());  

   p = p>0.5 ? p : 1-p ; 

   return p;
   
   
}





// print a node
//_______________________________________________________________________
void TMVA::DecisionTreeNode::Print(ostream& os) const
{
   //print the node
   os << "< ***  " <<endl; 
   os << " d: " << this->GetDepth()
      << " seq: " << this->GetSequence()
      << " ivar: " <<  this->GetSelector()
      << " cut: " << this->GetCutValue() 
      << " cType: " << this->GetCutType() 
      << " s: " << this->GetNSigEvents()
      << " b: " << this->GetNBkgEvents()
      << " nEv: " << this->GetNEvents()
      << " sepI: " << this->GetSeparationIndex()
      << " sepG: " << this->GetSeparationGain()
      << " nType: " << this->GetNodeType()
      <<endl;
   
   os << "My address is " << long(this) << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: " << long(this->GetParent()) ;
   if (this->GetLeft() != NULL) os << " left daughter at addr: " << long(this->GetLeft());
   if (this->GetRight() != NULL) os << " right daughter at addr: " << long(this->GetRight()) ;
   
   os << " **** > "<< endl;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRec(ostream& os) const
{
   //recursively print the node and its daughters (--> print the 'tree')

   os << this->GetDepth() 
      << " " << this->GetPos() 
      << " seq: " << this->GetSequence()
      << " ivar: " <<  this->GetSelector()
      << " cut: " << this->GetCutValue() 
      << " cType: " << this->GetCutType() 
      << " s: " << this->GetNSigEvents()
      << " b: " << this->GetNBkgEvents()
      << " nEv: " << this->GetNEvents()
      << " sepI: " << this->GetSeparationIndex()
      << " sepG: " << this->GetSeparationGain()
      << " nType: " << this->GetNodeType()
      <<endl;
  
   if(this->GetLeft() != NULL)this->GetLeft()->PrintRec(os) ;
   if(this->GetRight() != NULL)this->GetRight()->PrintRec(os);
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::ReadRec(istream& is,  char &pos, UInt_t &depth,
                                         TMVA::Node* parent )
{
   //recursively read the node and its daughters (--> read the 'tree')
   string tmp;
   Double_t dtmp1, dtmp2, dtmp3, dtmp4, dtmp5, dtmp6, dtmp7;
   UInt_t itmp;
   Int_t itmp1, itmp2;
   ULong_t lseq;

   if (parent==NULL) {
      is >> itmp >> pos ;
      //      this->SetDepth(itmp);
      this->SetPos(pos);
   } else {
      //      this->SetDepth(depth);
      this->SetPos(pos);
   }
  
   is >> tmp >> lseq
      >> tmp >> itmp1 >> tmp >> dtmp1 >> tmp >> dtmp2 >> tmp >> dtmp3
      >> tmp >> dtmp4 >> tmp >> dtmp5 >> tmp >> dtmp6 
      >> tmp >> dtmp7 >> tmp >> itmp2 ;

      
   this->SetSelector((UInt_t)itmp1);
   this->SetCutValue(dtmp1);
   this->SetCutType(dtmp2);
   this->SetNSigEvents(dtmp3);
   this->SetNBkgEvents(dtmp4);
   this->SetNEvents(dtmp5);
   this->SetSeparationIndex(dtmp6);
   this->SetSeparationGain(dtmp7);
   this->SetNodeType(itmp2);
   this->SetSequence(lseq);

   UInt_t nextDepth;
   char nextPos;
   is >> itmp1 >> nextPos ;
   nextDepth = UInt_t(itmp1);
      
   if ( UInt_t(nextDepth) == this->GetDepth()+1){
      if (nextPos=='l') {
         this->SetLeft(new TMVA::DecisionTreeNode(this,'l'));
         this->GetLeft()->ReadRec(is,nextPos,nextDepth,this);
      }
   }
   if ( UInt_t(nextDepth) == this->GetDepth()+1){
      if (nextPos=='r') {
         this->SetRight(new TMVA::DecisionTreeNode(this,'r'));
         this->GetRight()->ReadRec(is,nextPos,nextDepth,this);
      }
   }
   pos = nextPos;
   depth= nextDepth;
}


//_______________________________________________________________________
void TMVA::DecisionTreeNode::ClearNodeAndAllDaughters()
{
   // clear the nodes (their S/N, Nevents etc), just keep the structure of the tree
   fNSigEvents=0;
   fNBkgEvents=0;
   fNEvents = 0;
   fSeparationIndex=-1;
   fSeparationGain=-1;


   if(this->GetLeft() != NULL)
      ((DecisionTreeNode*)(this->GetLeft()))->ClearNodeAndAllDaughters();
   if(this->GetRight() != NULL)
      ((DecisionTreeNode*)(this->GetRight()))->ClearNodeAndAllDaughters();
}
