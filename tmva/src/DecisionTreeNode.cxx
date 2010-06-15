// @(#)root/tmva $Id$    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne 

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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Eckhard von Toerne <evt@physik.uni-bonn.de>  - U. of Bonn, Germany        *
 *                                                                                *
 * CopyRight (c) 2009:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
*       U. of Bonn, Germany                                                       *
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
#include <exception>
#include <iomanip>

#include "TMVA/MsgLogger.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/Tools.h"
#include "TMVA/Event.h"

using std::string;

ClassImp(TMVA::DecisionTreeNode)

TMVA::MsgLogger* TMVA::DecisionTreeNode::fgLogger = 0;
bool     TMVA::DecisionTreeNode::fgIsTraining = false;

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode()
   : TMVA::Node(),
     fCutValue(0),
     fCutType ( kTRUE ),
     fSelector ( -1 ),       
     fResponse(-99 ),
     fNodeType (-99 ),
     fSequence ( 0 ),
     fIsTerminalNode( kFALSE )
{
   // constructor of an essentially "empty" node floating in space
   if (!fgLogger) fgLogger = new TMVA::MsgLogger( "DecisionTreeNode" );

   if (fgIsTraining){
      fTrainInfo = new DTNodeTrainingInfo();
      //std::cout << "Node constructor with TrainingINFO"<<std::endl;
   }
   else {
      //std::cout << "**Node constructor WITHOUT TrainingINFO"<<std::endl;
      fTrainInfo = 0;
   }
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(TMVA::Node* p, char pos)
   : TMVA::Node(p, pos), 
     fCutValue( 0 ),
     fCutType ( kTRUE ),
     fSelector( -1 ),  
     fResponse(-99 ),
     fNodeType( -99 ),
     fSequence( 0 ),
     fIsTerminalNode( kFALSE )
{
   // constructor of a daughter node as a daughter of 'p'
   if (!fgLogger) fgLogger = new TMVA::MsgLogger( "DecisionTreeNode" );

   // get the sequence, depending on if it is a left or a right daughter
   if (pos == 'r' ){
      ULong_t tmp =1; for (UInt_t i=1; i<this->GetDepth(); i++) {tmp *= 2; }  //  (2^depth) 
      fSequence =  ((DecisionTreeNode*)p)->GetSequence() + tmp;
   } else {
      fSequence =  ((DecisionTreeNode*)p)->GetSequence();
   }      

   if (fgIsTraining){
      fTrainInfo = new DTNodeTrainingInfo();
      //std::cout << "Node constructor with TrainingINFO"<<std::endl;
   }
   else {
      //std::cout << "**Node constructor WITHOUT TrainingINFO"<<std::endl;
      fTrainInfo = 0;
   }
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode(const TMVA::DecisionTreeNode &n,
                                         DecisionTreeNode* parent)
   : TMVA::Node(n),
     fCutValue( n.fCutValue ),
     fCutType ( n.fCutType ),
     fSelector( n.fSelector ),  
     fResponse( n.fResponse ),
     fNodeType( n.fNodeType ),
     fSequence( n.fSequence ),
     fIsTerminalNode( n.fIsTerminalNode )  
{
   // copy constructor of a node. It will result in an explicit copy of
   // the node and recursively all it's daughters
   if (!fgLogger) fgLogger = new TMVA::MsgLogger( "DecisionTreeNode" );

   this->SetParent( parent );
   if (n.GetLeft() == 0 ) this->SetLeft(NULL);
   else this->SetLeft( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetLeft())),this));
   
   if (n.GetRight() == 0 ) this->SetRight(NULL);
   else this->SetRight( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetRight())),this));
   
   if (fgIsTraining){
      fTrainInfo = new DTNodeTrainingInfo(*(n.fTrainInfo));
      //std::cout << "Node constructor with TrainingINFO"<<std::endl;
   }
   else {
      //std::cout << "**Node constructor WITHOUT TrainingINFO"<<std::endl;
      fTrainInfo = 0;
   }
}


//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesRight(const TMVA::Event & e) const
{
   // test event if it decends the tree at this node to the right  
   Bool_t result;
  
   result =  (e.GetValue(this->GetSelector()) > this->GetCutValue() );
  
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
Float_t TMVA::DecisionTreeNode::GetPurity( void ) const  
{
   // return the S/(S+B) (purity) for the node
   // REM: even if nodes with purity 0.01 are very PURE background nodes, they still
   //      get a small value of the purity.
   if ( ( this->GetNSigEvents() + this->GetNBkgEvents() ) > 0 ) {
      return this->GetNSigEvents() / ( this->GetNSigEvents() + this->GetNBkgEvents()); 
   }
   else {
      *fgLogger << kINFO << "Zero events in purity calcuation , return purity=0.5" << Endl;
      this->Print(*fgLogger);
      return 0.5;
   }
}

// print a node
//_______________________________________________________________________
void TMVA::DecisionTreeNode::Print(ostream& os) const
{
   //print the node
   os << "< ***  "  << std::endl; 
   os << " d: "     << this->GetDepth()
      << " seq: "   << this->GetSequence()
      << " ivar: "  << this->GetSelector()
      << " cut: "   << this->GetCutValue() 
      << " cType: " << this->GetCutType() 
      << " s: "     << this->GetNSigEvents()
      << " b: "     << this->GetNBkgEvents()
      << " nEv: "   << this->GetNEvents()
      << " suw: "   << this->GetNSigEvents_unweighted()
      << " buw: "   << this->GetNBkgEvents_unweighted()
      << " nEvuw: " << this->GetNEvents_unweighted()
      << " sepI: "  << this->GetSeparationIndex()
      << " sepG: "  << this->GetSeparationGain()
      << " nType: " << this->GetNodeType()
      << std::endl;
   
   os << "My address is " << long(this) << ", ";
   if (this->GetParent() != NULL) os << " parent at addr: "         << long(this->GetParent()) ;
   if (this->GetLeft()   != NULL) os << " left daughter at addr: "  << long(this->GetLeft());
   if (this->GetRight()  != NULL) os << " right daughter at addr: " << long(this->GetRight()) ;
   
   os << " **** > " << std::endl;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRec(ostream& os) const
{
   //recursively print the node and its daughters (--> print the 'tree')

   os << this->GetDepth() 
      << std::setprecision(6)
      << " "         << this->GetPos() 
      << " seq: "    << this->GetSequence()
      << " ivar: "   << this->GetSelector()
      << " cut: "    << this->GetCutValue() 
      << " cType: "  << this->GetCutType() 
      << " s: "      << this->GetNSigEvents()
      << " b: "      << this->GetNBkgEvents()
      << " nEv: "    << this->GetNEvents()
      << " suw: "    << this->GetNSigEvents_unweighted()
      << " buw: "    << this->GetNBkgEvents_unweighted()
      << " nEvuw: "  << this->GetNEvents_unweighted()
      << " sepI: "   << this->GetSeparationIndex()
      << " sepG: "   << this->GetSeparationGain()
      << " res: "    << this->GetResponse()
      << " rms: "    << this->GetRMS()
      << " nType: "  << this->GetNodeType();
   if (this->GetCC() > 10000000000000.) os << " CC: " << 100000. << std::endl;
   else os << " CC: "  << this->GetCC() << std::endl;
  
   if (this->GetLeft()  != NULL) this->GetLeft() ->PrintRec(os);
   if (this->GetRight() != NULL) this->GetRight()->PrintRec(os);
}

//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::ReadDataRecord( istream& is, UInt_t tmva_Version_Code ) 
{
   // Read the data block

   string tmp;
   
   Float_t cutVal, cutType, nsig, nbkg, nEv, nsig_unweighted, nbkg_unweighted, nEv_unweighted;
   Float_t separationIndex, separationGain, response(-99), cc(0);
   Int_t   depth, ivar, nodeType;
   ULong_t lseq;
   char pos;

   is >> depth;                                         // 2
   if ( depth==-1 ) { return kFALSE; }
   //   if ( depth==-1 ) { delete this; return kFALSE; }
   is >> pos ;                                          // r
   this->SetDepth(depth);
   this->SetPos(pos);

   if (tmva_Version_Code < TMVA_VERSION(4,0,0)) {
      is >> tmp >> lseq 
         >> tmp >> ivar 
         >> tmp >> cutVal  
         >> tmp >> cutType 
         >> tmp >> nsig    
         >> tmp >> nbkg    
         >> tmp >> nEv     
         >> tmp >> nsig_unweighted 
         >> tmp >> nbkg_unweighted   
         >> tmp >> nEv_unweighted    
         >> tmp >> separationIndex   
         >> tmp >> separationGain    
         >> tmp >> nodeType;         
   } else { 
      is >> tmp >> lseq 
         >> tmp >> ivar 
         >> tmp >> cutVal  
         >> tmp >> cutType 
         >> tmp >> nsig    
         >> tmp >> nbkg    
         >> tmp >> nEv     
         >> tmp >> nsig_unweighted 
         >> tmp >> nbkg_unweighted   
         >> tmp >> nEv_unweighted    
         >> tmp >> separationIndex   
         >> tmp >> separationGain    
         >> tmp >> response
         >> tmp >> nodeType           
         >> tmp >> cc;
   }

   this->SetSelector((UInt_t)ivar);
   this->SetCutValue(cutVal);
   this->SetCutType(cutType);
   this->SetNSigEvents(nsig);
   this->SetNBkgEvents(nbkg);
   this->SetNEvents(nEv);
   this->SetNSigEvents_unweighted(nsig_unweighted);
   this->SetNBkgEvents_unweighted(nbkg_unweighted);
   this->SetNEvents_unweighted(nEv_unweighted);
   this->SetSeparationIndex(separationIndex);
   this->SetSeparationGain(separationGain);
   this->SetNodeType(nodeType);
   
   this->SetResponse(response);
   this->SetSequence(lseq);
   this->SetCC(cc);

   return kTRUE;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::ClearNodeAndAllDaughters()
{
   // clear the nodes (their S/N, Nevents etc), just keep the structure of the tree
   SetNSigEvents(0);
   SetNBkgEvents(0);
   SetNEvents(0);
   SetNSigEvents_unweighted(0);
   SetNBkgEvents_unweighted(0);
   SetNEvents_unweighted(0);
   SetSeparationIndex(-1);
   SetSeparationGain(-1);

   if (this->GetLeft()  != NULL) ((DecisionTreeNode*)(this->GetLeft()))->ClearNodeAndAllDaughters();
   if (this->GetRight() != NULL) ((DecisionTreeNode*)(this->GetRight()))->ClearNodeAndAllDaughters();
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::ResetValidationData( ) {
   // temporary stored node values (number of events, etc.) that originate
   // not from the training but from the validation data (used in pruning)
   SetNBValidation( 0.0 );
   SetNSValidation( 0.0 );
   SetSumTarget( 0 );
   SetSumTarget2( 0 );

   if(GetLeftDaughter() != NULL && GetRightDaughter() != NULL) {
      GetLeftDaughter()->ResetValidationData();
      GetRightDaughter()->ResetValidationData();
   }
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintPrune( ostream& os ) const {
   // printout of the node (can be read in with ReadDataRecord)

   os << "----------------------" << std::endl 
      << "|~T_t| " << GetNTerminal() << std::endl 
      << "R(t): " << GetNodeR() << std::endl 
      << "R(T_t): " << GetSubTreeR() << std::endl
      << "g(t): " << GetAlpha() << std::endl
      << "G(t): "  << GetAlphaMinSubtree() << std::endl;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRecPrune( ostream& os ) const {
   // recursive printout of the node and its daughters 

   this->PrintPrune(os);
   if(this->GetLeft() != NULL && this->GetRight() != NULL) {
      ((DecisionTreeNode*)this->GetLeft())->PrintRecPrune(os);
      ((DecisionTreeNode*)this->GetRight())->PrintRecPrune(os);
   }
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::SetCC(Double_t cc) 
{
   if (fTrainInfo) fTrainInfo->fCC = cc; 
   else *fgLogger << kFATAL << "call to SetCC without trainingInfo" << Endl;
}

//_______________________________________________________________________
Float_t TMVA::DecisionTreeNode::GetSampleMin(UInt_t ivar) const {
   // return the minimum of variable ivar from the training sample 
   // that pass/end up in this node 
   if (fTrainInfo && ivar < fTrainInfo->fSampleMin.size()) return fTrainInfo->fSampleMin[ivar];
   else *fgLogger << kFATAL << "You asked for Min of the event sample in node for variable " 
                 << ivar << " that is out of range" << Endl;
   return -9999;
}

//_______________________________________________________________________
Float_t TMVA::DecisionTreeNode::GetSampleMax(UInt_t ivar) const {
   // return the maximum of variable ivar from the training sample 
   // that pass/end up in this node 
   if (fTrainInfo && ivar < fTrainInfo->fSampleMin.size()) return fTrainInfo->fSampleMax[ivar];
   else *fgLogger << kFATAL << "You asked for Max of the event sample in node for variable " 
                 << ivar << " that is out of range" << Endl;
   return 9999;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::SetSampleMin(UInt_t ivar, Float_t xmin){
   // set the minimum of variable ivar from the training sample 
   // that pass/end up in this node 
   if ( fTrainInfo && ivar >= fTrainInfo->fSampleMin.size()) fTrainInfo->fSampleMin.resize(ivar+1);
   fTrainInfo->fSampleMin[ivar]=xmin;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::SetSampleMax(UInt_t ivar, Float_t xmax){
   // set the maximum of variable ivar from the training sample 
   // that pass/end up in this node 
   if ( fTrainInfo && ivar >= fTrainInfo->fSampleMax.size()) fTrainInfo->fSampleMax.resize(ivar+1);
   fTrainInfo->fSampleMax[ivar]=xmax;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::ReadAttributes(void* node, UInt_t /* tmva_Version_Code */  ) 
{   
   Float_t tempNSigEvents,tempNBkgEvents,tempNEvents,tempNSigEvents_unweighted,  tempNBkgEvents_unweighted,tempNEvents_unweighted, tempSeparationIndex, tempSeparationGain;  
   Double_t tempCC;

   // read attribute from xml
   gTools().ReadAttr(node, "Seq",   fSequence               );
   gTools().ReadAttr(node, "IVar",  fSelector               );
   gTools().ReadAttr(node, "Cut",   fCutValue               );
   gTools().ReadAttr(node, "cType", fCutType                );
   gTools().ReadAttr(node, "nS",    tempNSigEvents             );
   gTools().ReadAttr(node, "nB",    tempNBkgEvents             );
   gTools().ReadAttr(node, "nEv",   tempNEvents                );
   gTools().ReadAttr(node, "nSuw",  tempNSigEvents_unweighted  );
   gTools().ReadAttr(node, "nBuw",  tempNBkgEvents_unweighted  );
   gTools().ReadAttr(node, "nEvuw", tempNEvents_unweighted     );
   gTools().ReadAttr(node, "sepI",  tempSeparationIndex        );
   gTools().ReadAttr(node, "sepG",  tempSeparationGain         );
   gTools().ReadAttr(node, "res",   fResponse               );
   gTools().ReadAttr(node, "rms",   fRMS                    );
   gTools().ReadAttr(node, "nType", fNodeType               );
   gTools().ReadAttr(node, "CC",    tempCC                  );
   if (fTrainInfo){
      SetNSigEvents(tempNSigEvents);
      SetNBkgEvents(tempNBkgEvents);
      SetNEvents(tempNEvents);
      SetNSigEvents_unweighted(tempNSigEvents_unweighted);
      SetNBkgEvents_unweighted(tempNBkgEvents_unweighted);
      SetNEvents_unweighted(tempNEvents_unweighted);
      SetSeparationIndex(tempSeparationIndex);
      SetSeparationGain(tempSeparationGain);
      SetCC(tempCC);  
   }
}


//_______________________________________________________________________
void TMVA::DecisionTreeNode::AddAttributesToNode(void* node) const
{
   // add attribute to xml
   gTools().AddAttr(node, "Seq",   GetSequence());
   gTools().AddAttr(node, "IVar",  GetSelector());
   gTools().AddAttr(node, "Cut",   GetCutValue());
   gTools().AddAttr(node, "cType", GetCutType());
   gTools().AddAttr(node, "nS",    GetNSigEvents());
   gTools().AddAttr(node, "nB",    GetNBkgEvents());
   gTools().AddAttr(node, "nEv",   GetNEvents());
   gTools().AddAttr(node, "nSuw",  GetNSigEvents_unweighted());
   gTools().AddAttr(node, "nBuw",  GetNBkgEvents_unweighted());
   gTools().AddAttr(node, "nEvuw", GetNEvents_unweighted());
   gTools().AddAttr(node, "sepI",  GetSeparationIndex());
   gTools().AddAttr(node, "sepG",  GetSeparationGain());
   gTools().AddAttr(node, "res",   GetResponse());
   gTools().AddAttr(node, "rms",   GetRMS());
   gTools().AddAttr(node, "nType", GetNodeType());
   gTools().AddAttr(node, "CC",    (GetCC() > 10000000000000.)?100000.:GetCC());
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::AddContentToNode( std::stringstream& /*s*/ ) const
{   
   // adding attributes to tree node  (well, was used in BinarySearchTree,
   // and somehow I guess someone programmed it such that we need this in
   // this tree too, although we don't..)
}

//_______________________________________________________________________ 
void TMVA::DecisionTreeNode::ReadContent( std::stringstream& /*s*/ )
{
   // reading attributes from tree node  (well, was used in BinarySearchTree,
   // and somehow I guess someone programmed it such that we need this in
   // this tree too, although we don't..)
}
