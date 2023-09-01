/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : CCTreeWrapper                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: a light wrapper of a decision tree, used to perform cost          *
 *              complexity pruning "in-place" Cost Complexity Pruning             *
 *                                                                                *
 * Author: Doug Schouten (dschoute@sfu.ca)                                        *
 *                                                                                *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Texas at Austin, USA                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::CCTreeWrapper
\ingroup TMVA

*/

#include "TMVA/CCTreeWrapper.h"
#include "TMVA/DecisionTree.h"

#include <iostream>
#include <limits>

using namespace TMVA;

////////////////////////////////////////////////////////////////////////////////
///constructor of the CCTreeNode

TMVA::CCTreeWrapper::CCTreeNode::CCTreeNode( DecisionTreeNode* n ) :
   Node(),
   fNLeafDaughters(0),
   fNodeResubstitutionEstimate(-1.0),
   fResubstitutionEstimate(-1.0),
   fAlphaC(-1.0),
   fMinAlphaC(-1.0),
   fDTNode(n)
{
   if ( n != NULL && n->GetRight() != NULL && n->GetLeft() != NULL ) {
      SetRight( new CCTreeNode( ((DecisionTreeNode*) n->GetRight()) ) );
      GetRight()->SetParent(this);
      SetLeft( new CCTreeNode( ((DecisionTreeNode*) n->GetLeft()) ) );
      GetLeft()->SetParent(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// destructor of a CCTreeNode

TMVA::CCTreeWrapper::CCTreeNode::~CCTreeNode() {
   if(GetLeft() != NULL) delete GetLeftDaughter();
   if(GetRight() != NULL) delete GetRightDaughter();
}

////////////////////////////////////////////////////////////////////////////////
/// initialize a node from a data record

Bool_t TMVA::CCTreeWrapper::CCTreeNode::ReadDataRecord( std::istream& in, UInt_t /* tmva_Version_Code */ ) {
   std::string header, title;
   in >> header;
   in >> title; in >> fNLeafDaughters;
   in >> title; in >> fNodeResubstitutionEstimate;
   in >> title; in >> fResubstitutionEstimate;
   in >> title; in >> fAlphaC;
   in >> title; in >> fMinAlphaC;
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// printout of the node (can be read in with ReadDataRecord)

void TMVA::CCTreeWrapper::CCTreeNode::Print( std::ostream& os ) const {
   os << "----------------------" << std::endl
      << "|~T_t| " << fNLeafDaughters << std::endl
      << "R(t): " << fNodeResubstitutionEstimate << std::endl
      << "R(T_t): " << fResubstitutionEstimate << std::endl
      << "g(t): " << fAlphaC << std::endl
      << "G(t): " << fMinAlphaC << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// recursive printout of the node and its daughters

void TMVA::CCTreeWrapper::CCTreeNode::PrintRec( std::ostream& os ) const {
   this->Print(os);
   if(this->GetLeft() != NULL && this->GetRight() != NULL) {
      this->GetLeft()->PrintRec(os);
      this->GetRight()->PrintRec(os);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::CCTreeWrapper::CCTreeWrapper( DecisionTree* T, SeparationBase* qualityIndex ) :
   fRoot(NULL)
{
   fDTParent = T;
   fRoot = new CCTreeNode( dynamic_cast<DecisionTreeNode*>(T->GetRoot()) );
   fQualityIndex = qualityIndex;
   InitTree(fRoot);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::CCTreeWrapper::~CCTreeWrapper( ) {
   delete fRoot;
}

////////////////////////////////////////////////////////////////////////////////
/// initialize the node t and all its descendants

void TMVA::CCTreeWrapper::InitTree( CCTreeNode* t )
{
   Double_t s = t->GetDTNode()->GetNSigEvents();
   Double_t b = t->GetDTNode()->GetNBkgEvents();
   //   Double_t s = t->GetDTNode()->GetNSigEvents_unweighted();
   //   Double_t b = t->GetDTNode()->GetNBkgEvents_unweighted();
   // set R(t) = Gini(t) or MisclassificationError(t), etc.
   t->SetNodeResubstitutionEstimate((s+b)*fQualityIndex->GetSeparationIndex(s,b));

   if(t->GetLeft() != NULL && t->GetRight() != NULL) { // n is an interior (non-leaf) node
      // traverse the tree
      InitTree(t->GetLeftDaughter());
      InitTree(t->GetRightDaughter());
      // set |~T_t|
      t->SetNLeafDaughters(t->GetLeftDaughter()->GetNLeafDaughters() +
                           t->GetRightDaughter()->GetNLeafDaughters());
      // set R(T) = sum[t' in ~T]{ R(t) }
      t->SetResubstitutionEstimate(t->GetLeftDaughter()->GetResubstitutionEstimate() +
                                   t->GetRightDaughter()->GetResubstitutionEstimate());
      // set g(t)
      t->SetAlphaC((t->GetNodeResubstitutionEstimate() - t->GetResubstitutionEstimate()) /
                   (t->GetNLeafDaughters() - 1));
      // G(t) = min( g(t), G(l(t)), G(r(t)) )
      t->SetMinAlphaC(std::min(t->GetAlphaC(), std::min(t->GetLeftDaughter()->GetMinAlphaC(),
                                                        t->GetRightDaughter()->GetMinAlphaC())));
   }
   else { // n is a terminal node
      t->SetNLeafDaughters(1);
      t->SetResubstitutionEstimate((s+b)*fQualityIndex->GetSeparationIndex(s,b));
      t->SetAlphaC(std::numeric_limits<double>::infinity( ));
      t->SetMinAlphaC(std::numeric_limits<double>::infinity( ));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// remove the branch rooted at node t

void TMVA::CCTreeWrapper::PruneNode( CCTreeNode* t )
{
   if( t->GetLeft() != NULL &&
       t->GetRight() != NULL ) {
      CCTreeNode* l = t->GetLeftDaughter();
      CCTreeNode* r = t->GetRightDaughter();
      t->SetNLeafDaughters( 1 );
      t->SetResubstitutionEstimate( t->GetNodeResubstitutionEstimate() );
      t->SetAlphaC( std::numeric_limits<double>::infinity( ) );
      t->SetMinAlphaC( std::numeric_limits<double>::infinity( ) );
      delete l;
      delete r;
      t->SetLeft(NULL);
      t->SetRight(NULL);
   }else{
      std::cout << " ERROR in CCTreeWrapper::PruneNode: you try to prune a leaf node.. that does not make sense " << std::endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// return the misclassification rate of a pruned tree for a validation event sample
/// using an EventList

Double_t TMVA::CCTreeWrapper::TestTreeQuality( const EventList* validationSample )
{
   Double_t ncorrect=0, nfalse=0;
   for (UInt_t ievt=0; ievt < validationSample->size(); ievt++) {
      Bool_t isSignalType = (CheckEvent(*(*validationSample)[ievt]) > fDTParent->GetNodePurityLimit() ) ? 1 : 0;

      if (isSignalType == ((*validationSample)[ievt]->GetClass() == 0)) {
         ncorrect += (*validationSample)[ievt]->GetWeight();
      }
      else{
         nfalse += (*validationSample)[ievt]->GetWeight();
      }
   }
   return  ncorrect / (ncorrect + nfalse);
}

////////////////////////////////////////////////////////////////////////////////
/// return the misclassification rate of a pruned tree for a validation event sample
/// using the DataSet

Double_t TMVA::CCTreeWrapper::TestTreeQuality( const DataSet* validationSample )
{
   validationSample->SetCurrentType(Types::kValidation);
   // test the tree quality.. in terms of Misclassification
   Double_t ncorrect=0, nfalse=0;
   for (Long64_t ievt=0; ievt<validationSample->GetNEvents(); ievt++){
      const Event *ev = validationSample->GetEvent(ievt);

      Bool_t isSignalType = (CheckEvent(*ev) > fDTParent->GetNodePurityLimit() ) ? 1 : 0;

      if (isSignalType == (ev->GetClass() == 0)) {
         ncorrect += ev->GetWeight();
      }
      else{
         nfalse += ev->GetWeight();
      }
   }
   return  ncorrect / (ncorrect + nfalse);
}

////////////////////////////////////////////////////////////////////////////////
/// return the decision tree output for an event

Double_t TMVA::CCTreeWrapper::CheckEvent( const TMVA::Event & e, Bool_t useYesNoLeaf )
{
   const DecisionTreeNode* current = fRoot->GetDTNode();
   CCTreeNode* t = fRoot;

   while(//current->GetNodeType() == 0 &&
         t->GetLeft() != NULL &&
         t->GetRight() != NULL){ // at an interior (non-leaf) node
      if (current->GoesRight(e)) {
         //current = (DecisionTreeNode*)current->GetRight();
         t = t->GetRightDaughter();
         current = t->GetDTNode();
      }
      else {
         //current = (DecisionTreeNode*)current->GetLeft();
         t = t->GetLeftDaughter();
         current = t->GetDTNode();
      }
   }

   if (useYesNoLeaf) return (current->GetPurity() > fDTParent->GetNodePurityLimit() ? 1.0 : -1.0);
   else return current->GetPurity();
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::CCTreeWrapper::CCTreeNode::AddAttributesToNode( void* /*node*/ ) const
{}

////////////////////////////////////////////////////////////////////////////////

void TMVA::CCTreeWrapper::CCTreeNode::AddContentToNode( std::stringstream& /*s*/ ) const
{}

////////////////////////////////////////////////////////////////////////////////

void TMVA::CCTreeWrapper::CCTreeNode::ReadAttributes( void* /*node*/, UInt_t /* tmva_Version_Code */  )
{}

////////////////////////////////////////////////////////////////////////////////

void TMVA::CCTreeWrapper::CCTreeNode::ReadContent( std::stringstream& /*s*/ )
{}
