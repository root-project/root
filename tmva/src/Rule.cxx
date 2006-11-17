// @(#)root/tmva $Id: Rule.cxx,v 1.32 2006/11/16 22:51:59 helgevoss Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Rule                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class describung a 'rule'                                               * 
 *      Each internal node of a tree defines a rule from all the parental nodes.  *
 *      A rule consists of atleast 2 nodes.                                       *
 *      Input: a decision tree (in the constructor)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//________________________________________________________________________________
//
// Implementation of a rule
//
// A rule is simply a branch or a part of a branch in a tree.
// It fullfills the following:
// * First node is the root node of the originating tree
// * Consists of a minimum of 2 nodes
// * A rule returns for a given event:
//    0 : if the event fails at any node
//    1 : otherwise
// * If the rule contains <2 nodes, it returns 0 SHOULD NOT HAPPEN!
//
// The coefficient is found by either brute force or some sort of
// intelligent fitting. See the RuleEnsemble class for more info.
//________________________________________________________________________________

#include "Riostream.h"

#include "TMVA/Event.h"
#include "TMVA/Rule.h"
#include "TMVA/RuleFit.h"
#include "TMVA/RuleEnsemble.h"
#include "TMVA/MethodRuleFit.h"

//_______________________________________________________________________
TMVA::Rule::Rule( RuleEnsemble *re,
                  const std::vector< const Node * >& nodes,
                  const std::vector< Int_t >       & cutdirs )
   : fNorm          ( 1.0 )
   , fSupport       ( 0.0 )
   , fSigma         ( 0.0 )
   , fCoefficient   ( 0.0 )
   , fImportance    ( 0.0 )
   , fImportanceRef ( 1.0 )
   , fRuleEnsemble  ( re )
   , fLogger( "Rule" )
{
   // the main constructor for a Rule

   //
   // input:
   //   nodes  - a vector of TMVA::Node; from these all possible rules will be created
   //
   //
   SetNodes( nodes );
   SetCutDirs( cutdirs );
}

//_______________________________________________________________________
void TMVA::Rule::SetNodes( const std::vector< const TMVA::Node * >& nodes )
{
   // Sets the node list to be used for the rule generation.
   // Checks that the list is OK:
   // * it must contain the root node
   // * it must have atleast 2 nodes
   //
   fNodes.clear();
   //
   // Return if node list is empty
   //
   if (nodes.size()<2) return;
   //
   // Set the nodes used for the rule
   //
   const TMVA::Node *firstNode = *(nodes.begin());
   //
   // First node must be root. Just a consistency check...
   //
   if (firstNode==0) fLogger << kFATAL << "<SetNodes> First node is NULL!" << Endl;
   else {
      if (firstNode->GetParent()!=0) {
         fLogger << kFATAL << "<SetNodes> invalid Rule - first node in list is not root;"
                 << " rule created but will not have any associated nodes." << Endl;
         return;
      }
   }
   fNodes   = nodes;
   fSSB     = dynamic_cast<const TMVA::DecisionTreeNode*>(fNodes.back())->GetSoverSB();
   fSSBNeve = dynamic_cast<const TMVA::DecisionTreeNode*>(fNodes.back())->GetNEvents();
}

//_______________________________________________________________________
const Int_t TMVA::Rule::GetNumVarsUsed() const
{
   //
   // Get the number of different variables used for this Rule
   //
   std::vector<Int_t> counter;
   Int_t maxind=0;
   Int_t sel;
   UInt_t nnodes = fNodes.size()-1;
   //
   // There exists probably a more elegant way using stl...
   //
   // Establish max variable index
   //
   for ( UInt_t i=0; i<nnodes; i++) {
      sel = dynamic_cast<const TMVA::DecisionTreeNode*>(fNodes[i])->GetSelector();
      if (sel>maxind) maxind=sel;
   }
   counter.resize(maxind+1);
   //
   // Count each index
   //
   for ( UInt_t i=0; i<nnodes; i++) {
      sel = dynamic_cast<const TMVA::DecisionTreeNode*>(fNodes[i])->GetSelector();
      counter[sel]++;
   }
   //
   // Count the number of occuring variables
   //
   Int_t nvars=0;
   for ( Int_t i=0; i<maxind+1; i++) {
      if (counter[i]>0) nvars++;
   }
   return nvars;
}

//_______________________________________________________________________
const Bool_t TMVA::Rule::ContainsVariable(Int_t iv) const
{
   // check if variable in node
   Bool_t found    = kFALSE;
   Bool_t doneLoop = kFALSE;
   UInt_t nnodes = fNodes.size();
   UInt_t i=0;
   while (!doneLoop) {
      if (dynamic_cast<const TMVA::DecisionTreeNode*>(fNodes[i])->GetSelector() == iv){ 
         found=kTRUE;
      }
      i++;
      doneLoop = (found || (i==nnodes));
   }
   return found;
}

//_______________________________________________________________________
Int_t TMVA::Rule::EqualSingleNode( const Rule& other ) const
{
   // REMOVE
   // Returns:
   //   +1 : equal, one node, cutvalue(this) > cutvalue(other)
   //   -1 : equal, one node, cutvalue(this) < cutvalue(other)
   //    0 : they do not match
   //
   if (fNodes.size()!=2) return 0;
   Bool_t almostEqual = Equal( other, kFALSE, 0 );
   if (!almostEqual) return 0;
   //
   const TMVA::DecisionTreeNode *otherNode = static_cast< const TMVA::DecisionTreeNode *>(other.GetNode(0));
   const TMVA::DecisionTreeNode *node      = static_cast< const TMVA::DecisionTreeNode *>(fNodes[0]);
   return ( node->GetCutValue() > otherNode->GetCutValue() ? +1:-1 );
}


//_______________________________________________________________________
Int_t TMVA::Rule::Equivalent( const Rule& other ) const
{
   // not used
   if (&other); // dummy

   fLogger << kFATAL << "Should not yet be called : Rule::Equivalent()" << Endl;

   return 0;
}

//_______________________________________________________________________
Bool_t TMVA::Rule::Equal( const Rule& other, Bool_t useCutValue, Double_t maxdist ) const
//
// Compare two rules.
// useCutValue: true -> calculate a distance between the two rules based on the cut values
//                      if the rule cuts are not equal, the distance is < 0 (-1.0)
//                      return true if d<maxdist
//              false-> ignore maxdist, return true if rules are equal, ignoring cut values
// maxdist:     max distance allowed; if < 0 => set useCutValue=false;
//
{
   Bool_t rval=kFALSE;
   if (maxdist<0) useCutValue=kFALSE;
   Double_t d = RuleDist( other, useCutValue );
   // cut value used - return true if 0<=d<maxdist
   if (useCutValue) rval = ( (!(d<0)) && (d<maxdist) );
   else rval = (!(d<0));
   // cut value not used, return true if <> -1
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::Rule::RuleDist( const Rule& other, Bool_t useCutValue ) const
//
// Returns:
// -1.0 : rules are NOT equal, i.e, variables and/or cut directions are wrong
//   >=0: rules are equal apart from the cutvalue, returns d = sqrt(sum(c1-c2)^2)
// If not useCutValue, the distance is exactly zero if they are equal
//
{
   // first get effective rules
   std::vector<Int_t> ruleThis;
   std::vector<Int_t> ruleOther;
   GetEffectiveRule(ruleThis);
   other.GetEffectiveRule(ruleOther);
   if (ruleThis.size()!=ruleOther.size()) return -1.0; // check number of nodes
   //
   const UInt_t nnodes = ruleThis.size();
   const Double_t norm = 1.0; // TODO: the norm should be just unity - makes sense!?
                              // 1.0/Double_t(nnodes-1); // just the number of equal nodes
   UInt_t in = 0;
   Bool_t equal = true;
   const TMVA::DecisionTreeNode *node, *otherNode;
   Double_t a,b,s,rms;
   Double_t sumdc2=0;
   Double_t sumw=0;
   Int_t indThis, indOther;
   // compare nodes
   // A 'distance' is assigned if the two rules has exactly the same set of cuts but with
   // different cut values.
   // The distance is given in number of sigmas
   //
   while ((equal) && (in<nnodes-1)) {
      indThis  = ruleThis[in];
      indOther = ruleOther[in];
      if (fCutDirs[indThis]!=other.GetCutDir(indOther)) equal=false; // check cut type
      if (equal) {
         otherNode = static_cast< const TMVA::DecisionTreeNode *>(other.GetNode(indOther));
         node = static_cast< const TMVA::DecisionTreeNode *>(fNodes[indThis]);
         if (node->GetSelector()!= otherNode->GetSelector() ) equal=false; // check all variable names
         if (equal && useCutValue) {
            a = node->GetCutValue();
            b = otherNode->GetCutValue();
            // messy - but ok...
            rms = fRuleEnsemble->GetRuleFit()->GetMethodRuleFit()->Data().GetRMS(node->GetSelector());
            s = ( rms>0 ? (a-b)/rms : 0 );
            sumdc2 += s*s;
            sumw   += 1.0/(rms*rms); // TODO: probably not needed
         }
      }
      in++;
   }
   if (!useCutValue) sumdc2 = (equal ? 0.0:-1.0); // ignore cut values
   else              sumdc2 = (equal ? sqrt(sumdc2)*norm : -1.0);

   return sumdc2;
}

//_______________________________________________________________________
void TMVA::Rule::GetEffectiveRule( std::vector<Int_t>& nodeind ) const
{
   //
   // Returns a vector of node indecis which correspond to the effective rule.
   // E.g, the rule:
   //  v1<0.1
   //  v1<0.05
   //  v4>0.12
   //
   // is effectively the same as:
   //  v1<0.05
   //  v4>0.12
   //
   nodeind.clear();
   UInt_t nnodes = fNodes.size();
   if (nnodes==2) { // just one cut, return all nodes
      for (UInt_t i=0; i<nnodes; i++) {
         nodeind.push_back(i);
      }
      return;
   }
   //
   // Loop over all nodes - 1 (last node is not a cut)
   //
   Bool_t equal;
   UInt_t in=0;
   while (in<nnodes-1) {
      equal = kFALSE;
      while ((CmpNodeCut(in,in+1)!=0) && (in<nnodes-2)) {
         in++;
         equal = kTRUE;
      }
      nodeind.push_back(in);
      in++;
   }
   nodeind.push_back(nnodes-1); // last should always be the end-node
}

//_______________________________________________________________________
Int_t TMVA::Rule::CmpNodeCut( Int_t node1, Int_t node2 ) const
{
   // returns:
   // -1 : nodes are equal, cutvalue1<cutvalue2
   //  0 : nodes are not equal; different var or cutdir
   // +1 : nodes are equal, cutvalue1>cutvalue2
   //
   Int_t nnodes = Int_t(fNodes.size());
   // Check nodes
   if ((node1<0) || (node2<0) || (node1>nnodes-2) || (node2>nnodes-2)) return 0;

   // Check cut direction
   if (fCutDirs[node1]!=fCutDirs[node2]) return 0;

   // Check cut variable
   const TMVA::DecisionTreeNode *dnode1 = static_cast< const TMVA::DecisionTreeNode *>(fNodes[node1]);
   const TMVA::DecisionTreeNode *dnode2 = static_cast< const TMVA::DecisionTreeNode *>(fNodes[node2]);
   if (dnode1->GetSelector()!=dnode2->GetSelector()) return 0;

   // Check cut value
   return (dnode1->GetCutValue()>dnode2->GetCutValue() ? +1 : -1);
}

//_______________________________________________________________________
Bool_t TMVA::Rule::IsSimpleRule() const
{
   // REMOVE
   // A simple rule is defined by:
   // * contains one variable
   // * all cuts are in the same direction
   //
   if (fNodes.size()==2) { // size=2 -> only one cut -> always simple
      return kTRUE;
   }
   //
   if (GetNumVarsUsed()!=1) {
      return kFALSE;
   }
   UInt_t ic = 0;
   UInt_t nc = fCutDirs.size()-1;
   Bool_t done = kFALSE;
   Int_t co;
   Int_t coPrev=0;
   Bool_t allop=kFALSE;
   while (!done) {
      co = fCutDirs[ic];
      if (ic==0) coPrev = co;
      allop = (coPrev==co);
      ic++;
      done = ((ic==nc) || (!allop));
      coPrev = co;
   }
   return allop;
}

//_______________________________________________________________________
Bool_t TMVA::Rule::operator==( const TMVA::Rule& other ) const
{
   // comparison operator ==

   return this->Equal( other, kTRUE, 1e-3 );
}

//_______________________________________________________________________
Bool_t TMVA::Rule::operator<( const TMVA::Rule& other ) const
{
   // comparison operator <
   return (fImportance < other.GetImportance());
}

//_______________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::Rule& rule )
{
   // ostream operator
   rule.Print( os );
   return os;
}

//_______________________________________________________________________
const TString & TMVA::Rule::GetVarName( Int_t i ) const
{
   // returns the name of a rule

   return fRuleEnsemble->GetMethodRuleFit()->GetInputExp(i);
}

//_______________________________________________________________________
void TMVA::Rule::Copy( const Rule& other )
{
   // copy function
   if(this != &other) {
      SetRuleEnsemble( other.GetRuleEnsemble() );
      SetNodes( other.GetNodes() );
      fSSB     = other.GetSSB();
      fSSBNeve = other.GetSSBNeve();
      SetCutDirs( other.GetCutDirs() );
      SetCoefficient(other.GetCoefficient());
      SetSupport( other.GetSupport() );
      SetSigma( other.GetSigma() );
      SetNorm( other.GetNorm() );
      CalcImportance();
      SetImportanceRef( other.GetImportanceRef() );
   }
}

//_______________________________________________________________________
void TMVA::Rule::Print( ostream& os ) const
{
   // print function
   Int_t ind;
   Int_t sel,ntype,nnodes;
   Double_t data, ssbval;
   const TMVA::DecisionTreeNode *node, *nextNode;
   std::vector<Int_t> nodes;
   GetEffectiveRule( nodes );
   //
   nnodes = nodes.size();
   os << "    Importance  = " << Form("%1.4f", fImportance/fImportanceRef) << endl;
   os << "    Coefficient = " << Form("%1.4f", fCoefficient) << endl;
   os << "    Support     = " << Form("%1.4f", fSupport)  << endl;
   os << "    S/(S+B)     = " << Form("%1.4f", fSSB)  << endl;  

   for ( Int_t i=0; i<nnodes-1; i++) {
      ind = nodes[i];
      node = dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[ind]);
      nextNode = (i+1<nnodes ? dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[ind+1]):0);
      data = 0;
      os << "    ";
      if (node!=0) {
         sel  = node->GetSelector();
         data = node->GetCutValue();
         ntype = GetCutDir(ind);
         ssbval = nextNode->GetSoverSB();
         os << Form("* Cut %2d",i+1)
            << " : " << GetVarName(sel)
            << (ntype==1 ? " > ":" < ")
            << Form("%10.3g",data)
            << "   S/(S+B) = " << Form("%1.3f",ssbval) << endl;
      } 
      else fLogger << kFATAL << "BUG WARNING - NOT A DecisionTreeNode!" << Endl;
   }
   if (nnodes<2) os << "     *** WARNING - <EMPTY RULE> ***" << endl;
}

//_______________________________________________________________________
void TMVA::Rule::PrintRaw( ostream& os ) const
{
   // extensive print function
   const TMVA::DecisionTreeNode *node;
   std::vector<Int_t> nodes;
   GetEffectiveRule( nodes );
   UInt_t nnodes = nodes.size();
   os << "Parameters: "
      << fImportance << " "
      << fImportanceRef << " "
      << fCoefficient << " "
      << fSupport << " "
      << fSigma << " "
      << fNorm << " "
      << fSSB << " "
      << fSSBNeve << " "
      << endl;
   os << "N(nodes): " << nnodes << endl; // mark end of nodes
   for ( UInt_t i=0; i<nnodes; i++) {
      node = dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[nodes[i]]);
      os << "Node " << i << " : " << flush;
      os << node->GetSelector()
         << " " << node->GetCutValue()
         << " " << GetCutDir(nodes[i])
         << " " << node->GetCutType() 
         //         << " " << node->GetSoverSB()
         << " " << node->GetNSigEvents()
         << " " << node->GetNBkgEvents()
         << " " << node->GetSeparationIndex()
         << " " << node->GetSeparationGain()
         << " " << node->GetNEvents()
         << " " << node->GetNodeType() << endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::ReadRaw( istream& istr )
{
   // read function (format is the same as written by PrintRaw)

   TString dummy;
   TMVA::DecisionTreeNode *node;
   std::vector<Int_t> nodes;
   UInt_t nnodes = nodes.size();
   istr >> dummy
        >> fImportance
        >> fImportanceRef
        >> fCoefficient
        >> fSupport
        >> fSigma
        >> fNorm
        >> fSSB
        >> fSSBNeve;
   istr >> dummy >> nnodes;
   Double_t cutval, cuttype, s, b, sepind,sepgain,neve;
   Int_t    idum, cutdir, nodetype;
   UInt_t   selvar;
   fNodes.clear();
   fCutDirs.clear();
   //
   for ( UInt_t i=0; i<nnodes; i++) {
      node = new TMVA::DecisionTreeNode();
      istr >> dummy >> idum; // get 'Node' and index
      istr >> dummy;         // get ':'
      istr >> selvar >> cutval >> cutdir >> cuttype >> s >> b >> sepind >> sepgain >> neve >> nodetype;
      node->SetSelector(selvar);
      node->SetCutValue(cutval);
      fCutDirs.push_back(cutdir);
      node->SetCutType(cuttype) ;
      //      node->SetSoverSB(sosb);
      node->SetNSigEvents(s);
      node->SetNBkgEvents(s);
      node->SetSeparationIndex(sepind);
      node->SetSeparationGain(sepgain);
      node->SetNEvents(neve);
      node->SetNodeType(nodetype);
      fNodes.push_back(node);
   }
}

