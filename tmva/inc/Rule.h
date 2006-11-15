// @(#)root/tmva $Id: Rule.h,v 1.22 2006/10/23 01:51:39 stelzer Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Rule                                                                  *
 *                                                                                *
 * Description:                                                                   *
 *      A class describung a 'rule'                                               * 
 *      Each internal node of a tree defines a rule from all the parental nodes.  *
 *      A rule consists of atleast 2 nodes.                                       *
 *      Input: a decision tree (in the constructor)                               *
 *             its coefficient                                                    *
 *                                                                                *
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
#ifndef ROOT_TMVA_Rule
#define ROOT_TMVA_Rule

#include <cmath>

#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class RuleEnsemble;

   class Rule;

   ostream& operator<<( ostream& os, const Rule & rule );

   class Rule {

      // ouput operator for a Rule
      friend ostream& operator<< ( ostream& os, const Rule & rule );

   public:

      // main constructor
      Rule( RuleEnsemble *re,
            const std::vector< const Node * > & nodes,
            const std::vector< Int_t >        & cutdires );

      // copy constructor
      Rule( const Rule & other ) { Copy( other ); }

      // empty constructor
      Rule() {}

      virtual ~Rule() {}

      // set RuleEnsemble ptr
      void SetRuleEnsemble( const RuleEnsemble *re ) { fRuleEnsemble = re; }

      // set Rule norm
      void SetNorm(Double_t norm)       { fNorm = (norm>0 ? 1.0/norm:1.0); }

      // set coefficient
      void SetCoefficient(Double_t v)   { fCoefficient=v; }

      // set support
      void SetSupport(Double_t v)       { fSupport=v; }

      // set sigma
      void SetSigma(Double_t v)         { fSigma=v; }

      // set s/(s+b)
      void SetSSB(Double_t v)           { fSSB=v; }

      // set N(eve) accepted by rule
      void SetSSBNeve(Double_t v)       { fSSBNeve=v; }

      // set reference importance
      void SetImportanceRef(Double_t v) { fImportanceRef=(v>0 ? v:1.0); }

      // calculate importance
      void CalcImportance()             { fImportance = fabs(fCoefficient)*fSigma; }

      // get the relative importance
      const Double_t GetRelImportance()  const { return fImportance/fImportanceRef; }

      // evaluate the Rule for the given Event using the coefficient
      inline Double_t EvalEvent( const Event& e, Bool_t norm ) const;

      // evaluate the Rule for the given Event, not using normalization or the coefficent
      inline Double_t EvalEvent( const Event& e ) const;

      // evaluate the Rule and return S/(S+B) of the end node
      // if >0 => signal and <0 bkg
      inline Double_t EvalEventSB( const Event& e ) const;

      // REMOVE
      // returns +-1 if the rules are equal single rules appart from the cutvalue
      Int_t EqualSingleNode( const Rule & other ) const;

      // test if two rules are equal
      Bool_t Equal( const Rule & other, Bool_t useCutValue, Double_t maxdist ) const;

      // test if two rules are equivalent REMOVE
      Int_t  Equivalent( const Rule & other ) const;

      // test if a rule is 'simple', ie only one var with one cut direction
      Bool_t IsSimpleRule() const;

      // returns true if the trained S/(S+B) of the last node is > 0.5
      Double_t GetSSB()       const { return fSSB; }
      Double_t GetSSBNeve()   const { return fSSBNeve; }
      Bool_t   IsSignalRule() const { return (fSSB>0.5); }

      // copy operator
      void operator=( const Rule & other )  { Copy( other ); }

      // identical operator
      Bool_t operator==( const Rule & other ) const;

      Bool_t operator<( const Rule & other ) const;

      // get number of variables used in Rule
      const Int_t GetNumVarsUsed() const;

      // check if variable is used by the rule
      const Bool_t ContainsVariable(Int_t iv) const;

      // get the effective rule
      void GetEffectiveRule( std::vector<Int_t> & nodeind ) const;

      // accessors
      const RuleEnsemble                * GetRuleEnsemble()  const { return fRuleEnsemble; }
      const Double_t                      GetCoefficient()   const { return fCoefficient; }
      const Double_t                      GetSupport()       const { return fSupport; }
      const Double_t                      GetSigma()         const { return fSigma; }
      const Double_t                      GetNorm()          const { return fNorm; }
      const Double_t                      GetImportance()    const { return fImportance; }
      const Double_t                      GetImportanceRef() const { return fImportanceRef; }
      const std::vector< const Node * > & GetNodes()         const { return fNodes; }
      const std::vector< Int_t >        & GetCutDirs()       const { return fCutDirs; }

      const Node *GetNode( Int_t i )     const { return fNodes[i]; }
      Int_t       GetCutDir( Int_t i )  const { return fCutDirs[i]; }

      // print just the raw info, used for weight file generation
      void PrintRaw( ostream& os ) const;

      void ReadRaw( istream& os );

   private:

      // print info about the Rule
      void Print( ostream& os ) const;

      // copy from another rule
      void Copy( const Rule & other );

      // set the vector of nodes
      void SetNodes( const std::vector< const Node * > & nodes );
      void SetCutDirs( const std::vector< Int_t > & cutdir ) { fCutDirs = cutdir; }

      // compare two nodes
      Int_t CmpNodeCut( Int_t node1, Int_t node2 ) const;

      // get distance between two equal (ie apart from the cut values) rules
      Double_t RuleDist( const Rule & other, Bool_t useCutValue ) const;

      // get the name of variable with index i
      const TString & GetVarName( Int_t i) const;

      std::vector< const Node * > fNodes;         // vector of all nodes in the Rule
      std::vector< Int_t>         fCutDirs;       // cut type +1 - right, -1 - left, 0 - undefined; PHASE OUT
      Double_t                    fNorm;          // normalization - usually 1.0/t(k)
      Double_t                    fSupport;       // s(k)
      Double_t                    fSigma;         // t(k) = sqrt(s*(1-s))
      Double_t                    fCoefficient;   // rule coeff. a(k)
      Double_t                    fImportance;    // importance of rule
      Double_t                    fImportanceRef; // importance ref
      const RuleEnsemble         *fRuleEnsemble;  // pointer to parent RuleEnsemble
      Double_t                    fSSB;           // S/(S+B) for rule
      Double_t                    fSSBNeve;       // N(events) reaching the last node in reevaluation

      mutable MsgLogger           fLogger;        // message logger

   };

} // end of TMVA namespace

//_______________________________________________________________________
inline Double_t TMVA::Rule::EvalEvent( const TMVA::Event& e, Bool_t norm ) const
{
   //
   // Evaluates the event for the given rule.
   // It will return the value scaled with the coefficient and
   // if requested, also normalized.
   //
   if (fCoefficient==0) return 0.0;
   Double_t rval = TMVA::Rule::EvalEvent(e);
   rval *= (norm ? fNorm : 1.0 );
   return rval*fCoefficient;
}

//_______________________________________________________________________
inline Double_t TMVA::Rule::EvalEvent( const TMVA::Event& e ) const
{
   // Will go through list of nodes to see if event is accepted by rule.
   // Return 1 if yes and 0 if not.
   // Evaluates eq. (7) in RuleFit paper.
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
inline Double_t TMVA::Rule::EvalEventSB( const TMVA::Event& e ) const
{
   // Evaluates the event.
   // Returns S/(S+B) if the event passes, else 0.
   //
   Double_t r = TMVA::Rule::EvalEvent(e);
   return (r>0 ? r*GetSSB():0);
}

#endif
