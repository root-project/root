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
 *      U. of Bonn, Germany                                                       *
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
#include <limits>

#include "TMVA/MsgLogger.h"
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/Tools.h"
#include "TMVA/Event.h"

using std::string;

ClassImp(TMVA::DecisionTreeNode)

bool     TMVA::DecisionTreeNode::fgIsTraining = false;
UInt_t   TMVA::DecisionTreeNode::fgTmva_Version_Code = 0;
//_______________________________________________________________________
TMVA::DecisionTreeNode::DecisionTreeNode()
   : TMVA::Node(),
     fCutValue(0),
     fCutType ( kTRUE ),
     fSelector ( -1 ),
     fResponse(-99 ),
     fRMS(0),
     fNodeType (-99 ),
     fPurity (-99),
     fIsTerminalNode( kFALSE )
{
   // constructor of an essentially "empty" node floating in space
   if (DecisionTreeNode::fgIsTraining){
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
     fRMS(0),
     fNodeType( -99 ),
     fPurity (-99),
     fIsTerminalNode( kFALSE )
{
   // constructor of a daughter node as a daughter of 'p'
   if (DecisionTreeNode::fgIsTraining){
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
     fRMS     ( n.fRMS),
     fNodeType( n.fNodeType ),
     fPurity  ( n.fPurity),
     fIsTerminalNode( n.fIsTerminalNode )
{
   // copy constructor of a node. It will result in an explicit copy of
   // the node and recursively all it's daughters
   this->SetParent( parent );
   if (n.GetLeft() == 0 ) this->SetLeft(NULL);
   else this->SetLeft( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetLeft())),this));

   if (n.GetRight() == 0 ) this->SetRight(NULL);
   else this->SetRight( new DecisionTreeNode( *((DecisionTreeNode*)(n.GetRight())),this));

   if (DecisionTreeNode::fgIsTraining){
      fTrainInfo = new DTNodeTrainingInfo(*(n.fTrainInfo));
      //std::cout << "Node constructor with TrainingINFO"<<std::endl;
   }
   else {
      //std::cout << "**Node constructor WITHOUT TrainingINFO"<<std::endl;
      fTrainInfo = 0;
   }
}

//_______________________________________________________________________
TMVA::DecisionTreeNode::~DecisionTreeNode(){
   // destructor
   delete fTrainInfo;
}


//_______________________________________________________________________
Bool_t TMVA::DecisionTreeNode::GoesRight(const TMVA::Event & e) const
{
   // test event if it decends the tree at this node to the right
   Bool_t result;
   // first check if the fisher criterium is used or ordinary cuts:
   if (GetNFisherCoeff() == 0){

      result = (e.GetValue(this->GetSelector()) >= this->GetCutValue() );

   }else{

      Double_t fisher = this->GetFisherCoeff(fFisherCoeff.size()-1); // the offset
      for (UInt_t ivar=0; ivar<fFisherCoeff.size()-1; ivar++)
         fisher += this->GetFisherCoeff(ivar)*(e.GetValue(ivar));

      result = fisher > this->GetCutValue();
   }

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
void TMVA::DecisionTreeNode::SetPurity( void )
{
   // return the S/(S+B) (purity) for the node
   // REM: even if nodes with purity 0.01 are very PURE background nodes, they still
   //      get a small value of the purity.

   if ( ( this->GetNSigEvents() + this->GetNBkgEvents() ) > 0 ) {
      fPurity = this->GetNSigEvents() / ( this->GetNSigEvents() + this->GetNBkgEvents());
   }
   else {
      Log() << kINFO << "Zero events in purity calcuation , return purity=0.5" << Endl;
      this->Print(Log());
      fPurity = 0.5;
   }
   return;
}

// print a node
//_______________________________________________________________________
void TMVA::DecisionTreeNode::Print(std::ostream& os) const
{
   //print the node
   os << "< ***  "  << std::endl;
   os << " d: "     << this->GetDepth()
      << std::setprecision(6)
      << "NCoef: "  << this->GetNFisherCoeff();
   for (Int_t i=0; i< (Int_t) this->GetNFisherCoeff(); i++) { os << "fC"<<i<<": " << this->GetFisherCoeff(i);}
   os << " ivar: "  << this->GetSelector()
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
void TMVA::DecisionTreeNode::PrintRec(std::ostream& os) const
{
   //recursively print the node and its daughters (--> print the 'tree')

   os << this->GetDepth()
      << std::setprecision(6)
      << " "         << this->GetPos()
      << "NCoef: "   << this->GetNFisherCoeff();
   for (Int_t i=0; i< (Int_t) this->GetNFisherCoeff(); i++) {os << "fC"<<i<<": " << this->GetFisherCoeff(i);}
   os << " ivar: "   << this->GetSelector()
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
Bool_t TMVA::DecisionTreeNode::ReadDataRecord( std::istream& is, UInt_t tmva_Version_Code )
{
   // Read the data block
   fgTmva_Version_Code=tmva_Version_Code;
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
   this->SetNodeType(nodeType);
   if (fTrainInfo){
      this->SetNSigEvents(nsig);
      this->SetNBkgEvents(nbkg);
      this->SetNEvents(nEv);
      this->SetNSigEvents_unweighted(nsig_unweighted);
      this->SetNBkgEvents_unweighted(nbkg_unweighted);
      this->SetNEvents_unweighted(nEv_unweighted);
      this->SetSeparationIndex(separationIndex);
      this->SetSeparationGain(separationGain);
      this->SetPurity();
      //      this->SetResponse(response); old .txt weightfiles don't know regression yet
      this->SetCC(cc);
   }

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
   SetPurity();

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

   if(GetLeft() != NULL && GetRight() != NULL) {
      GetLeft()->ResetValidationData();
      GetRight()->ResetValidationData();
   }
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintPrune( std::ostream& os ) const {
   // printout of the node (can be read in with ReadDataRecord)

   os << "----------------------" << std::endl
      << "|~T_t| " << GetNTerminal() << std::endl
      << "R(t): " << GetNodeR() << std::endl
      << "R(T_t): " << GetSubTreeR() << std::endl
      << "g(t): " << GetAlpha() << std::endl
      << "G(t): "  << GetAlphaMinSubtree() << std::endl;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::PrintRecPrune( std::ostream& os ) const {
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
   else Log() << kFATAL << "call to SetCC without trainingInfo" << Endl;
}

//_______________________________________________________________________
Float_t TMVA::DecisionTreeNode::GetSampleMin(UInt_t ivar) const {
   // return the minimum of variable ivar from the training sample
   // that pass/end up in this node
   if (fTrainInfo && ivar < fTrainInfo->fSampleMin.size()) return fTrainInfo->fSampleMin[ivar];
   else Log() << kFATAL << "You asked for Min of the event sample in node for variable "
              << ivar << " that is out of range" << Endl;
   return -9999;
}

//_______________________________________________________________________
Float_t TMVA::DecisionTreeNode::GetSampleMax(UInt_t ivar) const {
   // return the maximum of variable ivar from the training sample
   // that pass/end up in this node
   if (fTrainInfo && ivar < fTrainInfo->fSampleMin.size()) return fTrainInfo->fSampleMax[ivar];
   else Log() << kFATAL << "You asked for Max of the event sample in node for variable "
              << ivar << " that is out of range" << Endl;
   return 9999;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::SetSampleMin(UInt_t ivar, Float_t xmin){
   // set the minimum of variable ivar from the training sample
   // that pass/end up in this node
   if ( fTrainInfo) {
      if ( ivar >= fTrainInfo->fSampleMin.size()) fTrainInfo->fSampleMin.resize(ivar+1);
      fTrainInfo->fSampleMin[ivar]=xmin;
   }
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::SetSampleMax(UInt_t ivar, Float_t xmax){
   // set the maximum of variable ivar from the training sample
   // that pass/end up in this node
   if( ! fTrainInfo ) return;
   if ( ivar >= fTrainInfo->fSampleMax.size() )
      fTrainInfo->fSampleMax.resize(ivar+1);
   fTrainInfo->fSampleMax[ivar]=xmax;
}

//_______________________________________________________________________
void TMVA::DecisionTreeNode::ReadAttributes(void* node, UInt_t /* tmva_Version_Code */  )
{
   Float_t tempNSigEvents,tempNBkgEvents;

   Int_t nCoef;
   if (gTools().HasAttr(node, "NCoef")){
      gTools().ReadAttr(node, "NCoef",  nCoef                  );
      this->SetNFisherCoeff(nCoef);
      Double_t tmp;
      for (Int_t i=0; i< (Int_t) this->GetNFisherCoeff(); i++) {
         gTools().ReadAttr(node, Form("fC%d",i),  tmp          );
         this->SetFisherCoeff(i,tmp);
      }
   }else{
      this->SetNFisherCoeff(0);
   }
   gTools().ReadAttr(node, "IVar",  fSelector               );
   gTools().ReadAttr(node, "Cut",   fCutValue               );
   gTools().ReadAttr(node, "cType", fCutType                );
   if (gTools().HasAttr(node,"res")) gTools().ReadAttr(node, "res",   fResponse);
   if (gTools().HasAttr(node,"rms")) gTools().ReadAttr(node, "rms",   fRMS);
   //   else {
   if( gTools().HasAttr(node, "purity") ) {
      gTools().ReadAttr(node, "purity",fPurity );
   } else {
      gTools().ReadAttr(node, "nS",    tempNSigEvents             );
      gTools().ReadAttr(node, "nB",    tempNBkgEvents             );
      fPurity = tempNSigEvents / (tempNSigEvents + tempNBkgEvents);
   }
   //   }
   gTools().ReadAttr(node, "nType", fNodeType               );
}


//_______________________________________________________________________
void TMVA::DecisionTreeNode::AddAttributesToNode(void* node) const
{
   // add attribute to xml
   gTools().AddAttr(node, "NCoef", GetNFisherCoeff());
   for (Int_t i=0; i< (Int_t) this->GetNFisherCoeff(); i++)
      gTools().AddAttr(node, Form("fC%d",i),  this->GetFisherCoeff(i));

   gTools().AddAttr(node, "IVar",  GetSelector());
   gTools().AddAttr(node, "Cut",   GetCutValue());
   gTools().AddAttr(node, "cType", GetCutType());

   //UInt_t analysisType = (dynamic_cast<const TMVA::DecisionTree*>(GetParentTree()) )->GetAnalysisType();
   //   if ( analysisType == TMVA::Types:: kRegression) {
   gTools().AddAttr(node, "res",   GetResponse());
   gTools().AddAttr(node, "rms",   GetRMS());
   //} else if ( analysisType == TMVA::Types::kClassification) {
   gTools().AddAttr(node, "purity",GetPurity());
   //}
   gTools().AddAttr(node, "nType", GetNodeType());
}

//_______________________________________________________________________
void  TMVA::DecisionTreeNode::SetFisherCoeff(Int_t ivar, Double_t coeff)
{
   // set fisher coefficients
   if ((Int_t) fFisherCoeff.size()<ivar+1) fFisherCoeff.resize(ivar+1) ;
   fFisherCoeff[ivar]=coeff;
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
//_______________________________________________________________________
TMVA::MsgLogger& TMVA::DecisionTreeNode::Log() {
  TTHREAD_TLS_DECL_ARG(MsgLogger,logger,"DecisionTreeNode");    // static because there is a huge number of nodes...
  return logger;
}
