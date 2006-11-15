// @(#)root/tmva $Id: RuleFit.h,v 1.24 2006/10/17 07:44:57 tegen Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RuleFit                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class implementing various fits of rule ensembles                       *
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

#ifndef ROOT_TMVA_RuleFit
#define ROOT_TMVA_RuleFit

#ifndef ROOT_TMVA_DecisionTree
#include "TMVA/DecisionTree.h"
#endif
#ifndef ROOT_TMVA_RuleEnsemble
#include "TMVA/RuleEnsemble.h"
#endif
#ifndef ROOT_TMVA_RuleFitParams
#include "TMVA/RuleFitParams.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class MethodRuleFit;

   class RuleFit {

   public:

      // main constructor
      RuleFit( const TMVA::MethodRuleFit *rfbase,
               const std::vector<TMVA::DecisionTree *> & forest,
               const std::vector<Event *> & trainingEvents,
               Double_t samplefrac );

      // empty constructor
      RuleFit( void );

      virtual ~RuleFit( void );

      void Initialise(  const TMVA::MethodRuleFit *rfbase,
                        const std::vector<TMVA::DecisionTree *> & forest,
                        const std::vector<Event *> & trainingEvents,
                        Double_t samplefrac );

      void SetTrainingEvents( const std::vector<Event *> & el, Double_t sampfrac );

      // calculate and print some statistics on the given forest
      void ForestStatistics();

      // calculate the discriminating variable for the given event
      Double_t EvalEvent( const Event& e );

      // do the fitting of the coefficients
      void     FitCoefficients();

      // calculate variable and rule importance from a set of events
      void     CalcImportance();

      // set usage of linear term
      void     SetModelLinear()                      { fRuleEnsemble.SetModelLinear(); }
      // set usage of rules
      void     SetModelRules()                       { fRuleEnsemble.SetModelRules(); }
      // set usage of linear term
      void     SetModelFull()                        { fRuleEnsemble.SetModelFull(); }
      // set minimum importance allowed
      void     SetImportanceCut( Double_t minimp=0 ) { fRuleEnsemble.SetImportanceCut(minimp); }
      // set max rule distance - see RuleEnsemble
      void     SetMaxRuleDist( Double_t maxd )       { fRuleEnsemble.SetMaxRuleDist(maxd); }
      // set path related parameters
      void     SetGDTau( Double_t t=0.0 )       { fRuleFitParams.SetGDTau(t); }
      void     SetGDPathStep( Double_t s=0.01 ) { fRuleFitParams.SetGDPathStep(s); }
      void     SetGDNPathSteps( Int_t n=100 )   { fRuleFitParams.SetGDNPathSteps(n); }
      // accessors
      const UInt_t  GetNSubsamples() const { return (fSubsampleEvents.size()>1 ? fSubsampleEvents.size()-1:0); }
      const Event*  GetTrainingEvent(UInt_t i)  const { return fTrainingEvents[i]; }
      const Event*  GetTrainingEvent(UInt_t i, UInt_t isub)  const { return &(fTrainingEvents[fSubsampleEvents[isub]])[i]; }

      const std::vector< const TMVA::Event * > & GetTrainingEvents()  const { return fTrainingEvents; }
      const std::vector< Int_t >               & GetSubsampleEvents() const { return fSubsampleEvents; }
      void                                       GetSubsampleEvents(Int_t sub, UInt_t & ibeg, UInt_t & iend) const;
      //
      const std::vector< const TMVA::DecisionTree *> & GetForest()     const { return fForest; }
      const RuleEnsemble                       & GetRuleEnsemble()     const { return fRuleEnsemble; }
            RuleEnsemble                       * GetRuleEnsemblePtr()        { return &fRuleEnsemble; }
      const RuleFitParams                      & GetRuleFitParams()    const { return fRuleFitParams; }
            RuleFitParams                      * GetRuleFitParamsPtr()       { return &fRuleFitParams; }
      const MethodRuleFit                      * GetMethodRuleFit()    const { return fMethodRuleFit; }

   private:

      // copy constructor
      RuleFit( const RuleFit & other );

      // copy method
      void Copy( const RuleFit & other );

      std::vector<const TMVA::Event *>    fTrainingEvents;  // all training events
      std::vector< Int_t >                fSubsampleEvents; // iterators marking the beginning of each cross validation sample
      std::vector< const TMVA::DecisionTree *>  fForest;    // the input forest of decision trees
      RuleEnsemble                        fRuleEnsemble;    // the ensemble of rules
      RuleFitParams                       fRuleFitParams;   // fit rule parameters
      const MethodRuleFit                *fMethodRuleFit;   // pointer the method which initialised this RuleFit instance

      mutable MsgLogger                   fLogger;          // message logger

      ClassDef(RuleFit,0)  // the actual calculations to Friedman's RuleFit method
         ;
   };
}

#endif
