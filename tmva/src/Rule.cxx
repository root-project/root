// @(#)root/tmva $Id: Rule.cxx,v 1.19 2006/10/04 22:29:27 andreas.hoecker Exp $
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
 *      MPI-KP Heidelberg, Germany                                                * 
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
//ClassImp(TMVA::Rule)

//_______________________________________________________________________
TMVA::Rule::Rule( RuleEnsemble *re,
                  const std::vector< const Node * > & nodes,
                  const std::vector< Int_t >        & cutdirs,
                  const std::vector<TString> *inpvars )
{
   // the main constructor for a Rule
   //
   // input:
   //   nodes  - a vector of TMVA::Node; from these all possible rules will be created
   //
   //
   fRuleEnsemble  = re;
   SetNodes( nodes );
   SetCutDirs( cutdirs );
   fInputVars     = inpvars;
   fNorm          = 1.0;
   fCoefficient   = 0.0;
   fSupport       = 0.0;
   fSigma         = 0.0;
   fImportance    = 0.0;
   fImportanceRef = 1.0;
}

//_______________________________________________________________________
void TMVA::Rule::SetNodes( const std::vector< const TMVA::Node * > & nodes )
{
   //
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
   //  std::cout << "Rule::SetNodes() Finding first node " << std::endl;
   const TMVA::Node *firstNode = *(nodes.begin());
   //
   // First node must be root. Just a consistency check...
   //
   if (firstNode==0) {
      std::cout << "ERROR in Rule::SetNodes() First node is NULL!" << std::endl;
      return;
   }
   if (firstNode!=0) {
      if (firstNode->GetParent()!=0) {
         std::cout << "ERROR in Rule::SetNodes(): Invalid Rule - first node in list is not root!" << std::endl;
         std::cout << "                           Rule created but will not have any associated nodes." << std::endl;
         return;
      }
   }
   fNodes = nodes;
}

//_______________________________________________________________________
void TMVA::Rule::SetCutDirs( const std::vector< Int_t > & cutdirs )
{
   fCutDirs = cutdirs;
}

//_______________________________________________________________________
const Int_t TMVA::Rule::GetNumVars() const
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
      sel = fNodes[i]->GetSelector();
      if (sel>maxind) maxind=sel;
   }
   counter.resize(maxind+1);
   //
   // Count each index
   //
   for ( UInt_t i=0; i<nnodes; i++) {
      sel = fNodes[i]->GetSelector();
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
   Bool_t found    = kFALSE;
   Bool_t doneLoop = kFALSE;
   UInt_t nnodes = fNodes.size();
   UInt_t i=0;
   while (!doneLoop) {
      if (fNodes[i]->GetSelector() == iv) found=kTRUE;
      i++;
      doneLoop = (found || (i==nnodes));
   }
   return found;
}


//_______________________________________________________________________
Double_t TMVA::Rule::EvalEvent( const TMVA::Event& e ) const
{
   //
   // Will go through list of nodes to see if event is accepted by rule.
   // Return 1 if yes and 0 if not.
   // Evaluates eq. (7) in RuleFit paper.
   //
   if (fNodes.size()<2) {
      std::cout << "--- Rule : ERROR - rule has less than 2 nodes! BUG!!!" << std::endl;
      return 0.0; // It's an empty rule SHOULD NOT HAPPEN!
   }
   //
   // Loop over all nodes until the end or the event fails
   //
   Double_t rval = 1.0;
   UInt_t nnodes = fNodes.size()-1;
   UInt_t n=0;
   Bool_t stepOK=kTRUE;

   while (stepOK && (n<nnodes)) {
      stepOK = ( ((fCutDirs[n]== +1) && fNodes[n]->GoesRight(e)) ||
                 ((fCutDirs[n]== -1) && fNodes[n]->GoesLeft(e)) );
      if (!stepOK) rval = 0.0;
      n++;
   }
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::Rule::EvalEvent( const TMVA::Event& e, Bool_t norm ) const
{
   //
   // Evaluates the event for the given rule.
   // It will return the value scaled with the coefficient and
   // if requested, also normalized.
   //
   if (fCoefficient==0) return 0.0;
   Double_t rval = EvalEvent(e);
   rval *= (norm ? fNorm : 1.0 );
   return rval*fCoefficient;
}

//_______________________________________________________________________
Double_t TMVA::Rule::EvalEventSB( const TMVA::Event& e ) const
{
   //
   // Evaluates the event.
   // Returns S/(S+B) if the event passes, else 0.
   //
   Double_t r = EvalEvent(e);
   Double_t sb=0.0;
   if (fNodes.size()>0) {
      sb = GetRuleSSB();
   }
   return r*sb;
}

//_______________________________________________________________________
Int_t TMVA::Rule::EqualSingleNode( const Rule & other ) const
//REMOVE
// Returns:
//   +1 : equal, one node, cutvalue(this) > cutvalue(other)
//   -1 : equal, one node, cutvalue(this) < cutvalue(other)
//    0 : they do not match
//
{
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
   if (&other); // dummy

   std::cout << "Should not be called : Rule::Equivalent()" << std::endl;
   exit(-1);
   return 0;
}

//_______________________________________________________________________
bool TMVA::Rule::Equal( const Rule & other, Bool_t useCutValue, Double_t maxdist ) const
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
Double_t TMVA::Rule::RuleDist( const Rule & other, Bool_t useCutValue ) const
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
         //         std::cout << " Cutdirs" << std::endl;
         otherNode = static_cast< const TMVA::DecisionTreeNode *>(other.GetNode(indOther));
         node = static_cast< const TMVA::DecisionTreeNode *>(fNodes[indThis]);
         if (node->GetSelector()!= otherNode->GetSelector() ) equal=false; // check all variable names
         if (equal && useCutValue) {
            //            std::cout << "  Selector" << std::endl;
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
bool TMVA::Rule::IsSignalRule() const
{
   Double_t ssb = GetRuleSSB();
   return (ssb>0.5);
}

//_______________________________________________________________________
void TMVA::Rule::GetEffectiveRule( std::vector<Int_t> & nodeind ) const
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
{
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
bool TMVA::Rule::IsSimpleRule() const
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
   if (GetNumVars()!=1) {
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
   bool TMVA::Rule::operator==( const TMVA::Rule & other ) const
{
   return this->Equal( other, kTRUE, 1e-3 );
}

//_______________________________________________________________________
bool TMVA::Rule::operator<( const TMVA::Rule & other ) const
{
   return (fNodes.size() < other.GetNodes().size());
}
//_______________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::Rule & rule )
{
   rule.Print( os );
   return os;
}

//_______________________________________________________________________
const TString * TMVA::Rule::GetVarName( Int_t i ) const
{
   if (fInputVars) {
      Int_t nvars = Int_t(fInputVars->size());
      if ( (i<nvars) && (i>-1) ) return &((*fInputVars)[i]);
   }
   return 0;
}

//_______________________________________________________________________
void TMVA::Rule::Copy( const Rule & other )
{
   fInputVars = other.GetInputVars();
   //
   SetRuleEnsemble( other.GetRuleEnsemble() );
   SetNodes( other.GetNodes() );
   SetCutDirs( other.GetCutDirs() );
   SetCoefficient(other.GetCoefficient());
   SetSupport( other.GetSupport() );
   SetSigma( other.GetSigma() );
   SetNorm( other.GetNorm() );
   CalcImportance();
   SetImportanceRef( other.GetImportanceRef() );
}

//_______________________________________________________________________
void TMVA::Rule::Print( ostream & os ) const
{
   Int_t ind;
   Int_t sel,ntype,nnodes;
   Double_t data, ssbval;
   const TString *varname;
   const TMVA::DecisionTreeNode *node, *nextNode;
   std::vector<Int_t> nodes;
   GetEffectiveRule( nodes );
   //
   nnodes = nodes.size();
   os << "    Importance  = " << Form("%1.4f", fImportance/fImportanceRef) << std::endl;
   os << "    Coefficient = " << Form("%1.4f", fCoefficient) << std::endl;
   os << "    Support     = " << Form("%1.4f", fSupport)  << std::endl;
   //
   for ( Int_t i=0; i<nnodes-1; i++) {
      ind = nodes[i];
      node = dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[ind]);
      nextNode = (i+1<nnodes ? dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[ind+1]):0);
      data = 0;
      os << "    ";
      if (node!=0) {
         sel  = node->GetSelector();
         varname = GetVarName(sel);
         data = node->GetCutValue();
         ntype = GetCutDir(ind);
         ssbval = nextNode->GetSoverSB();
         os << Form("* Cut %2d",i+1)
            << " : " << *varname //(varname ? varname:Form("%4d",sel))
            << (ntype==1 ? " > ":" < ")
            << Form("%10.3g",data)
            << "   S/(S+B) = " << Form("%1.3f",ssbval) << std::endl;
      } else {
         os << "BUG WARNING - NOT A DecisionTreeNode!" << std::endl;
         exit(1);
      }
   }
   if (nnodes<2) {
      os << "     *** WARNING - <EMPTY RULE> ***" << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::PrintRaw( ostream & os ) const
{
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
      << endl;
   os << "N(nodes): " << nnodes << endl; // mark end of nodes
   for ( UInt_t i=0; i<nnodes; i++) {
      node = dynamic_cast< const TMVA::DecisionTreeNode * >(fNodes[nodes[i]]);
      os << "Node " << i << " : " << flush;
      os << node->GetSelector()
         << " " << node->GetCutValue()
         << " " << GetCutDir(nodes[i])
         << " " << node->GetCutType() 
         << " " << node->GetSoverSB()
         << " " << node->GetSeparationIndex()
         << " " << node->GetSeparationGain()
         << " " << node->GetNEvents()
         << " " << node->GetNodeType() << endl;
   }
}

//_______________________________________________________________________
void TMVA::Rule::ReadRaw( istream & istr )
{
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
        >> fNorm;
   istr >> dummy >> nnodes;
   Double_t cutval, cuttype, sosb,sepind,sepgain,neve;
   Int_t    idum, cutdir, nodetype;
   UInt_t   selvar;
   fNodes.clear();
   fCutDirs.clear();
   //
   for ( UInt_t i=0; i<nnodes; i++) {
      node = new TMVA::DecisionTreeNode();
      istr >> dummy >> idum; // get 'Node' and index
      istr >> dummy;         // get ':'
      istr >> selvar >> cutval >> cutdir >> cuttype >> sosb >> sepind >> sepgain >> neve >> nodetype;
      node->SetSelector(selvar);
      node->SetCutValue(cutval);
      fCutDirs.push_back(cutdir);
      node->SetCutType(cuttype) ;
      node->SetSoverSB(sosb);
      node->SetSeparationIndex(sepind);
      node->SetSeparationGain(sepgain);
      node->SetNEvents(neve);
      node->SetNodeType(nodetype);
      fNodes.push_back(node);
   }
}

//_______________________________________________________________________
bool TMVA::RuleCmp::operator()( const Rule * first, const Rule * second ) const
{
   std::cout << "operator()(*,*)" << std::endl;

   if (first->GetNodes().size()!=second->GetNodes().size()) return false; // check number of nodes
   std::cout << "N(nodes)" << std::endl;
   UInt_t nnodes = first->GetNodes().size();
   UInt_t in = 0;
   Bool_t equal = true;
   Bool_t eqsel=false;
   const TMVA::DecisionTreeNode *node, *secondNode;
   Double_t a,b,s,p;
   while ((equal) && (in<nnodes-1)) {
      if (first->GetCutDir(in)!=second->GetCutDir(in)) equal=false; // check cut type
      if (equal) {
         std::cout << in << "/" << nnodes-1 <<" Cutdirs" << std::endl;
         secondNode = static_cast< const TMVA::DecisionTreeNode *>(second->GetNode(in));
         node = static_cast< const TMVA::DecisionTreeNode *>(first->GetNode(in));
         if (node->GetSelector()!= secondNode->GetSelector() ) equal=false; // check all variable names
         if (equal) {
            std::cout << "    Selector" << std::endl;
            eqsel = true;
            a = node->GetCutValue();
            b = secondNode->GetCutValue();
            s = a+b;
            if (fabs(s)>0) {
               p = fabs((a-b)/s);
               if (p>0.001) equal=false; // check cut values
            }
         }
      }
      in++;
   }
   if (eqsel) {
      std::cout << "RULE 1" << std::endl;
      std::cout << *first;
      std::cout << "RULE 2" << std::endl;
      std::cout << *second;
   }
   if (equal) std::cout << "     EQUAL!" << std::endl;
   return equal;
}
