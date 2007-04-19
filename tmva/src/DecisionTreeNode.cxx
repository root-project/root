// @(#)root/tmva $Id: DecisionTreeNode.cxx,v 1.11 2006/11/20 15:35:28 brun Exp $    
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
TMVA::MsgLogger* TMVA::DecisionTreeNode::fLogger = 0;
   
//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode()
   : TMVA::Node(),
     fCutValue(0),
     fCutType ( kTRUE ),
     fSelector ( -1 ),  
     fNSigEvents ( 0 ),
     fNBkgEvents ( 0 ),
     fNEvents ( -1 ),
     fNSigEvents_unweighted ( 0 ),
     fNBkgEvents_unweighted ( 0 ),
     fNEvents_unweighted ( 0 ),
     fSeparationIndex (-1 ),
     fSeparationGain ( -1 ),
     fNodeType (-99 ),
     fSequence ( 0 ) 
{
   // constructor of an essentially "empty" node floating in space
   if (!fLogger) fLogger = new TMVA::MsgLogger( "DecisionTreeNode" );
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(TMVA::Node* p, char pos)
   : TMVA::Node(p, pos), 
     fCutValue( 0 ),
     fCutType ( kTRUE ),
     fSelector( -1 ),  
     fNSigEvents ( 0 ),
     fNBkgEvents ( 0 ),
     fNEvents ( -1 ),
     fNSigEvents_unweighted ( 0 ),
     fNBkgEvents_unweighted ( 0 ),
     fNEvents_unweighted ( 0 ),
     fSeparationIndex( -1 ),
     fSeparationGain ( -1 ),
     fNodeType( -99 ),
     fSequence( 0 ) 
{
   // constructor of a daughter node as a daughter of 'p'
   if (!fLogger) fLogger = new TMVA::MsgLogger( "DecisionTreeNode" );

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
     fSelector( n.fSelector ),  
     fNSigEvents ( n.fNSigEvents ),
     fNBkgEvents ( n.fNBkgEvents ),
     fNEvents ( n.fNEvents ),
     fNSigEvents_unweighted ( n.fNSigEvents_unweighted ),
     fNBkgEvents_unweighted ( n.fNBkgEvents_unweighted ),
     fNEvents_unweighted ( n.fNEvents_unweighted ),
     fSeparationIndex( n.fSeparationIndex ),
     fSeparationGain ( n.fSeparationGain ),
     fNodeType( n.fNodeType ),
     fSequence( n.fSequence )  
{
   // copy constructor of a node. It will result in an explicit copy of
   // the node and recursively all it's daughters
   if (!fLogger) fLogger = new TMVA::MsgLogger( "DecisionTreeNode" );

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
Double_t TMVA::DecisionTreeNode::GetPurity( void ) const  
{
   // return the S/(S+B) (purity) for the node
   // REM: even if nodes with purity 0.01 are very PURE background nodes, they still
   //      get a small value of the purity.
   if ( ( this->GetNSigEvents() + this->GetNBkgEvents() ) > 0 ) {
      return this->GetNSigEvents() / ( this->GetNSigEvents() + this->GetNBkgEvents()); 
   }
   else {
      *fLogger << kINFO << "Zero events in purity calcuation , retrun purity=0.5" << Endl;
      return 0.5;
   }
}

// print a node
//_______________________________________________________________________
void TMVA::DecisionTreeNode::Print(ostream& os) const
{
   //print the node
   os << "< ***  "  << endl; 
   os << " d: "     << this->GetDepth()
      << " seq: "   << this->GetSequence()
      << " ivar: "  << this->GetSelector()
      << " cut: "   << this->GetCutValue() 
      << " cType: " << this->GetCutType() 
      << " s: "     << this->GetNSigEvents()
      << " b: "     << this->GetNBkgEvents()
      << " nEv: "   << this->GetNEvents()
      << " suw: "     << this->GetNSigEvents_unweighted()
      << " buw: "     << this->GetNBkgEvents_unweighted()
      << " nEvuw: "   << this->GetNEvents_unweighted()
      << " sepI: "  << this->GetSeparationIndex()
      << " sepG: "  << this->GetSeparationGain()
      << " nType: " << this->GetNodeType()
      <<endl;
   
   os << "My address is " << long(this) << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: "         << long(this->GetParent()) ;
   if (this->GetLeft()   != NULL) os << " left daughter at addr: "  << long(this->GetLeft());
   if (this->GetRight()  != NULL) os << " right daughter at addr: " << long(this->GetRight()) ;
   
   os << " **** > "<< endl;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRec(ostream& os) const
{
   //recursively print the node and its daughters (--> print the 'tree')

   os << this->GetDepth() 
      << " "         << this->GetPos() 
      << " seq: "    << this->GetSequence()
      << " ivar: "   << this->GetSelector()
      << " cut: "    << this->GetCutValue() 
      << " cType: "  << this->GetCutType() 
      << " s: "      << this->GetNSigEvents()
      << " b: "      << this->GetNBkgEvents()
      << " nEv: "    << this->GetNEvents()
      << " suw: "     << this->GetNSigEvents_unweighted()
      << " buw: "     << this->GetNBkgEvents_unweighted()
      << " nEvuw: "   << this->GetNEvents_unweighted()
      << " sepI: "   << this->GetSeparationIndex()
      << " sepG: "   << this->GetSeparationGain()
      << " nType: "  << this->GetNodeType()
      <<endl;
  
   if(this->GetLeft()  != NULL) this->GetLeft() ->PrintRec(os);
   if(this->GetRight() != NULL) this->GetRight()->PrintRec(os);
}

//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::ReadDataRecord( istream& is ) 
{
   // Read the data block

   string tmp;
   Double_t dtmp1, dtmp2, dtmp3, dtmp4, dtmp5, dtmp6, dtmp7, dtmp8, dtmp9, dtmp10;
   Int_t depth, itmp1, itmp2;
   ULong_t lseq;
   char pos;

   // the format is
   // 2 r seq: 2 ivar: 0 cut: -1.5324 cType: 0 s: 353 b: 1053 nEv: 1406 suw: 353 buw: 1053 nEvuw: 1406 sepI: 0.188032 sepG: 8.18513 nType: 0

   is >> depth;                                         // 2
   if( depth==-1 ) { delete this; return kFALSE; }
   is >> pos ;                                          // r
   this->SetDepth(depth);
   this->SetPos(pos);

   is >> tmp >> lseq                                    // seq: 2
      >> tmp >> itmp1                                   // ivar: 0
      >> tmp >> dtmp1                                   // cut: -1.5324       
      >> tmp >> dtmp2                                   // cType: 0           
      >> tmp >> dtmp3                                   // s: 353             
      >> tmp >> dtmp4                                   // b: 1053            
      >> tmp >> dtmp5                                   // nEv: 1406          
      >> tmp >> dtmp6                                   // suw: 353             
      >> tmp >> dtmp7                                   // buw: 1053            
      >> tmp >> dtmp8                                   // nEvuw: 1406          
      >> tmp >> dtmp9                                   // sepI: 0.188032     
      >> tmp >> dtmp10                                   // sepG: 8.18513      
      >> tmp >> itmp2 ;                                 // nType: 0           

   
   this->SetSelector((UInt_t)itmp1);
   this->SetCutValue(dtmp1);
   this->SetCutType(dtmp2);
   this->SetNSigEvents(dtmp3);
   this->SetNBkgEvents(dtmp4);
   this->SetNEvents(dtmp5);
   this->SetNSigEvents_unweighted(dtmp6);
   this->SetNBkgEvents_unweighted(dtmp7);
   this->SetNEvents_unweighted(dtmp8);
   this->SetSeparationIndex(dtmp9);
   this->SetSeparationGain(dtmp10);
   this->SetNodeType(itmp2);
   this->SetSequence(lseq);

   return kTRUE;
}



//_______________________________________________________________________
void TMVA::DecisionTreeNode::ReadRec( istream& is,  char &pos, UInt_t &depth,
                                      TMVA::Node* parent )
{
   //recursively read the node and its daughters (--> read the 'tree')
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
   TMVA::Node * newNode = new TMVA::DecisionTreeNode();
   newNode->ReadRec(is, childPos, childDepth, this);
}



//_______________________________________________________________________
void TMVA::DecisionTreeNode::ClearNodeAndAllDaughters()
{
   // clear the nodes (their S/N, Nevents etc), just keep the structure of the tree
   fNSigEvents=0;
   fNBkgEvents=0;
   fNEvents = 0;
   fNSigEvents_unweighted=0;
   fNBkgEvents_unweighted=0;
   fNEvents_unweighted = 0;
   fSeparationIndex=-1;
   fSeparationGain=-1;


   if(this->GetLeft() != NULL)
      ((DecisionTreeNode*)(this->GetLeft()))->ClearNodeAndAllDaughters();
   if(this->GetRight() != NULL)
      ((DecisionTreeNode*)(this->GetRight()))->ClearNodeAndAllDaughters();
}
