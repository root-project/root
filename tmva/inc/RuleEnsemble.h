// @(#)root/tmva $Id: RuleEnsemble.h,v 1.17 2006/10/03 17:49:10 tegen Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RuleEnsemble                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class generating an ensemble of rules                                   *
 *      Input:  a forest of decision trees                                        *
 *      Output: an ensemble of rules                                              *
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

#ifndef ROOT_TMVA_RuleEnsemble
#define ROOT_TMVA_RuleEnsemble

#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif

#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

#ifndef ROOT_TMVA_Rule
#include "TMVA/Rule.h"
#endif

class TH1F;

namespace TMVA {
   class RuleFit;
   class RuleEnsemble {

      // output operator for a RuleEnsemble
      friend ostream& operator<< ( ostream& os, const RuleEnsemble & rules );
      
   public:
      enum LearningModel { kFull, kRules, kLinear };
      // main constructor
      RuleEnsemble( RuleFit *rf );

      // copy constructor
      RuleEnsemble( const RuleEnsemble & other ) { Copy( other ); }

      // empty constructor
      RuleEnsemble() {}

      virtual ~RuleEnsemble();

      // initialize
      void Initialize( RuleFit *rf );

      // makes the model - calls MakeRules() and MakeLinearTerms()
      void MakeModel();

      // generates the rules from a given forest of decision trees
      void MakeRules( const std::vector< const DecisionTree *> & forest );

      // make the linear terms
      void MakeLinearTerms();

      // select linear model
      void SetModelLinear() { fLearningModel = kLinear; }
      // select rule model
      void SetModelRules()  { fLearningModel = kRules; }
      // select full (linear+rules) model
      void SetModelFull()   { fLearningModel = kFull; }

      void SetDoLinearTerms( Bool_t f )      { fDoLinear = f; }

      // set coefficients
      void  SetCoefficients( const std::vector< Double_t > & v );
      void  SetCoefficient( UInt_t i, Double_t v )                  { if (i<fRules.size()) fRules[i]->SetCoefficient(v); }
      //
      void  SetOffset(Double_t v=0.0)                               { fOffset=v; }
      void  AddOffset(Double_t v)                                   { fOffset+=v; }
      void  SetLinCoefficients( const std::vector< Double_t > & v ) { fLinCoefficients = v; }
      void  SetLinCoefficient( UInt_t i, Double_t v )               { fLinCoefficients[i] = v; }

      // clear coefficients
      void  ClearCoefficients( Double_t val=0 )    { for (UInt_t i=0; i<fRules.size(); i++) fRules[i]->SetCoefficient(val); }
      void  ClearLinCoefficients( Double_t val=0 ) { for (UInt_t i=0; i<fLinCoefficients.size(); i++) fLinCoefficients[i]=val; }
      void  ClearLinNorm( Double_t val=1.0 )       { for (UInt_t i=0; i<fLinNorm.size(); i++) fLinNorm[i]=val; }

      // set maximum allowed distance between equal rules
      void SetMaxRuleDist(Double_t maxdist)    { fMaxRuleDist = maxdist; }

      // set minimum rule importance - used by CleanupRules()
      void SetImportanceCut(Double_t minimp=0) { fImportanceCut=minimp; }

      // Calculate the number of possible rules from a given tree
      Int_t CalcNRules( const TMVA::DecisionTree *dtree );
      // Recursivly search for end-nodes; used by CalcNRules()
      void  FindNEndNodes( const TMVA::Node *node, Int_t & nendnodes );

      // evaluates the event using the ensemble of rules
      Double_t EvalEvent( const Event& e ) const;

      // evaluate the linear term
      Double_t EvalLinEvent( UInt_t vind, const Event& e, Bool_t norm ) const;
      Double_t EvalLinEvent( const Event& e ) const;

      // calculate p(y=1|x) for a given event using the linear terms
      Double_t PdfLinVar( UInt_t var, const TMVA::Event& e, Double_t & fs, Double_t & fb ) const;
      Double_t PdfLinVar( const TMVA::Event& e, Int_t & nsig, Int_t & ntot ) const;

      // calculate p(y=1|x) for a given event using the rules
      Double_t PdfRule( const TMVA::Rule *rule, const TMVA::Event& e ) const;
      Double_t PdfRule( const TMVA::Event& e, Int_t & nsig, Int_t & ntot ) const;

      // calculate F* = 2*p(y=1|x) - 1
      Double_t FStar( const TMVA::Event& e ) const;

      // calculates the support for all rules given the set of events
      void CalcRuleSupport();

      // calculates rule importance
      void CalcImportance();

      // calculates rule importance
      Double_t CalcRuleImportance();

      // calculates linear importance
      Double_t CalcLinImportance();

      // calculates variable importance
      void CalcVarImportance();

      // remove rules of low importance
      void CleanupRules();

      // remove linear terms of low importance
      void CleanupLinear();

      // remove rules with just one variable and one cut direction
      void RemoveSimpleRules();

      // remove similar rules
      void RemoveSimilarRules();

      // get rule statistics
      void RuleStatistics();

      // copy operator
      void operator=( const RuleEnsemble & other ) { Copy( other ); }

      // calculate sum of the squared coefficents
      Double_t CoefficientRadius();

      // fill the vector with the coefficients
      void GetCoefficients( std::vector< Double_t > & v );

      // accessors
      const RuleFit                         *GetRuleFit()         const { return fRuleFit; }
      const std::vector<const Event *>      *GetTrainingEvents()  const;
      const std::vector< Int_t >            *GetSubsampleEvents() const;
      void                                   GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const;
      //
      const UInt_t                    GetNSubsamples() const;
      const Event *                   GetTrainingEvent(UInt_t i) const;
      const Event *                   GetTrainingEvent(UInt_t i, UInt_t isub)  const;
      //
      const Bool_t                    DoLinear()        const {
         return (fLearningModel==kFull) ||
            (fLearningModel == kLinear); }
      const Bool_t                    DoRules()        const {
         return (fLearningModel==kFull) ||
            (fLearningModel == kRules); }
      const Bool_t                    DoFull()        const { return (fLearningModel==kFull); }
      const LearningModel             GetLearningModel() const { return fLearningModel; }
      const Double_t                  GetImportanceCut() const { return fImportanceCut; }
      const Double_t                  GetOffset()            const { return fOffset; }
      const UInt_t                    GetNRules()            const { return fRules.size(); }
      const std::vector< Rule * >   & GetRulesConst()        const { return fRules; }
      std::vector< Rule * >   & GetRules()                   { return fRules; }
      const std::vector< Int_t  >   & GetRulesNCuts()        const { return fRulesNCuts; }
      const std::vector< Double_t > & GetLinCoefficients()   const { return fLinCoefficients; }
      const std::vector< Double_t > & GetLinNorm()           const { return fLinNorm; }
      const std::vector< Double_t > & GetLinImportance()  const { return fLinImportance; }
      const std::vector< Double_t > & GetVarImportance()     const { return fVarImportance; }

      const Rule    *GetRulesConst(int i)        const { return fRules[i]; }
      Rule    *GetRules(int i)                   { return fRules[i]; }
      const Int_t    GetRulesNCuts(int i)        const { return fRulesNCuts[i]; }
      const Double_t GetMaxRuleDist()            const { return fMaxRuleDist; }
      const Double_t GetLinCoefficients(int i)   const { return fLinCoefficients[i]; }
      const Double_t GetLinNorm(int i)           const { return fLinNorm[i]; }
      const Double_t GetLinImportance(int i)  const { return fLinImportance[i]; }
      const Double_t GetVarImportance(int i)     const { return fVarImportance[i]; }
      const Double_t GetRulePTag(int i)   const { return fRulePTag[i]; }
      const Double_t GetRulePSS(int i)    const { return fRulePSS[i]; }
      const Double_t GetRulePSB(int i)    const { return fRulePSB[i]; }
      const Double_t GetRulePBS(int i)    const { return fRulePBS[i]; }
      const Double_t GetRulePBB(int i)    const { return fRulePBB[i]; }

      const Double_t GetAverageSupport()    const { return fAverageSupport; }
      const Double_t GetAverageRuleSigma()  const { return fAverageRuleSigma; }

      // print the model in a cryptic way
      void  PrintRaw( ostream & os ) const;

      // read the model from input stream
      void  ReadRaw( istream & istr );

   private:
      // print the ensemble
      void  Print( ostream & os ) const;

      // copy method
      void  Copy( RuleEnsemble const & other );

      // set the corrected number of cuts per rule (eg [v<0.6] & [v<0.5] is ONE cut )
      void SetRulesNCuts();

      // set all coeffs to default values
      void  ResetCoefficients();

      // make rules form one decision tree
      void  MakeRulesFromTree( const DecisionTree *dtree );

      // add a rule with tghe given end-node
      void  AddRule( const Node *node );

      // make a rule
      Rule *MakeTheRule( const Node *node );

      Bool_t                        fDoLinear;          // if true, include linear terms in the model REMOVE
      LearningModel                 fLearningModel;     // can be full (rules+linear), rules, linear
      Double_t                      fImportanceCut;     // minimum importance accepted
      Double_t                      fOffset;            // offset in discriminator function
      std::vector< Rule * >         fRules;             // vector of rules
      std::vector< Int_t >          fRulesNCuts;        // corrected number of cuts (<= N(nodes)-1)
      std::vector< Bool_t >         fLinTermOK;         // flags linear terms with sufficient strong importance
      std::vector< Double_t >       fLinDP;             // delta+ in eq 24, ref 2
      std::vector< Double_t >       fLinDM;             // delta-
      std::vector< Double_t >       fLinCoefficients;   // linear coefficients, one per variable
      std::vector< Double_t >       fLinNorm;           // norm of ditto, see after eq 26 in ref 2
      std::vector< TH1F * >         fLinPDFB;           // pdfs for each variable, background
      std::vector< TH1F * >         fLinPDFS;           // pdfs for each variable, signal
      std::vector< Double_t >       fLinImportance;     // linear term importance
      std::vector< Double_t >       fVarImportance;     // one importance per input variable
      Double_t                      fImportanceRef;     // reference importance (max)
      Double_t                      fAverageSupport;    // average support (over all rules)
      Double_t                      fAverageRuleSigma;  // average rule sigma
      //
      std::vector< Double_t >       fRulePSS;   // p(tag as S|S) - tagged as S if rule is SIG and the event is accepted
      std::vector< Double_t >       fRulePSB;   // p(tag as S|B)
      std::vector< Double_t >       fRulePBS;
      std::vector< Double_t >       fRulePBB;
      std::vector< Double_t >       fRulePTag;  // p(tag)
      Double_t                      fRuleFSig;  // N(sig)/N(sig)+N(bkg)

      Double_t                      fMaxRuleDist;
      //
      const RuleFit                *fRuleFit;

      //    ClassDef(RuleEnsemble,0)
   };
};

#endif
