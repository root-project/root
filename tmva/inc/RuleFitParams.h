// @(#)root/tmva $Id: RuleFitParams.h,v 1.1 2006/10/09 15:55:02 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RuleFitParams                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class doing the actual fitting of a linear model using rules as         *
 *      base functions.                                                           *
 *      Reference paper: 1.Gradient Directed Regularization                       *
 *                         Friedman, Popescu, 2004                                *
 *                       2.Predictive Learning with Rule Ensembles                *
 *                         Friedman, Popescu, 2005                                *
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

#ifndef ROOT_TMVA_RuleFitParams
#define ROOT_TMVA_RuleFitParams

#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

class TTree;
namespace TMVA {
   class RuleFit;
   class RuleEnsemble;
   
   class RuleFitParams {
   public:
      RuleFitParams();
      virtual ~RuleFitParams();

      void Init();

      inline void SetRuleFit( RuleFit *rf )  { fRuleFit = rf; Init(); }
      inline void SetGDNPathSteps( Int_t np ) { fGDNPathSteps = np; };
      inline void SetGDPathStep( Double_t s ) { fGDPathStep = s; };
      inline void SetGDTau( Double_t t ) { fGDTau = t; };
      inline void SetGDErrNsigma( Double_t s ) { fGDErrNsigma = s; };

      // return type such that +1 = signal and -1 = background
      inline Int_t Type( const Event * e ) const { return (e->IsSignal() ? 1:-1); }
      //
      const std::vector<const Event *>   *GetTrainingEvents()  const;
      const std::vector< Int_t >         *GetSubsampleEvents() const;
      void                                GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const;
      //
      inline const UInt_t                 GetNSubsamples() const;
      inline const Event *                GetTrainingEvent(UInt_t i) const;
      inline const Event *                GetTrainingEvent(UInt_t i, UInt_t isub)  const;

      // Linear model; eq 2 in paper with x->f(x) where f(x) is a rule
      Double_t LinearModel( const Event& e ) const;

      // Loss function; Huber loss eq 33
      virtual Double_t LossFunction( const Event& e ) const;

      // Empirical risk, including regularization
      Double_t Risk() const;
    
      // Penalty function; Lasso function (eq 8)
      Double_t Penalty() const;

      // make path for binary classification (squared-error ramp, sect 6 in ref 1)
      void MakeGDPath();

   protected:
      // typedef of an Event const iterator
      typedef std::vector<Event *>::const_iterator  EventItr;

      // init ntuple
      void InitNtuple();

      // fill ntuple with coefficient info
      void FillCoefficients();

      // set the regularization - lasso method (NOT A PRIORITY)
      inline void SetRegularization( Double_t r ) { fRegularization = r; }

      // estimate of binary error rate
      Double_t ErrorRateBin(Int_t set, Double_t & df);

      // estimate of scale average error rate
      Double_t ErrorRateReg(Int_t set);

      // make gradient vector (eq 44 in ref 1)
      void MakeGradientVector();

      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      void UpdateCoefficients();

      // calculate average of responses (initial estimate of a0)
      Double_t CalcAverageResponse();

      // calculate offset (a0 with nonzero coeffs)
      Double_t CalcOffset(Int_t set);

      RuleFit      * fRuleFit;      // rule fit
      RuleEnsemble * fRuleEnsemble; // rule ensemble

      Double_t fRegularization; // regularization

      std::vector<Double_t> fGradVec;        // gradient vector - dimension = number of rules in ensemble
      std::vector<Double_t> fGradVecLin;     // gradient vector - dimension = number of variables
      std::vector<Double_t> fGradVecMin;     // gradient vector, min
      Double_t              fGradOfs;        // gradient for offset
      //
      //      std::vector<Double_t> fGradStep;       // gradient vector step, eq 25, ref 1
      Double_t              fGDTau;    // threshold parameter (tau in eq 26, ref 1)
      Double_t              fGDPathStep;   // step size along path (delta nu in eq 22, ref 1)
      Int_t                 fGDNPathSteps;
      Double_t              fGDErrNsigma;  // threshold difference from minimum (n sigmas)
      //
      std::vector<Double_t> fFstar;
      Double_t              fFstarMedian;
      Int_t                 fPerfUsedSet; // used data set for Performance() - if -1, undefined, select 0
      Bool_t                fFstarValid;
      //
      TTree                *fGDNtuple;
      Double_t              fNTRisk;
      Double_t              fNTErrorRate;
      Double_t              fNTNuval;
      Double_t              fNTCoefRad;
      Double_t              fNTOffset;
      Double_t             *fNTCoeff;
      Double_t             *fNTLinCoeff;
   };
   //
   class AbsValue {
   public:
      bool operator()(Double_t first, Double_t second) const
      {
         return TMath::Abs(first) < TMath::Abs(second);
      }
   };
};

#endif
