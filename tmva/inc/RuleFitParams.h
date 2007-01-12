// @(#)root/tmva $Id: RuleFitParams.h,v 1.8 2006/11/23 17:43:39 rdm Exp $
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
 *      MPI-K Heidelberg, Germany                                                 * 
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
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
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

      void SetRuleFit( RuleFit *rf )    { fRuleFit = rf; Init(); }
      //
      void SetGDNPathSteps( Int_t np )  { fGDNPathSteps = np; }
      void SetGDPathStep( Double_t s )  { fGDPathStep = s; }
      void SetGDTau( Double_t t )       { fGDTau = t; }
      void SetGDErrNsigma( Double_t s ) { fGDErrNsigma = s; }

      // return type such that +1 = signal and -1 = background
      Int_t Type( const Event * e ) const { return (e->IsSignal() ? 1:-1); }
      //
      const std::vector<const Event *> *GetTrainingEvents()  const { return &fTrainingEvents; }
      const std::vector< Int_t >       *GetSubsampleEvents() const;
      void                              GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const;
      UInt_t                            GetPathIdx1() const { return fPathIdx1; }
      UInt_t                            GetPathIdx2() const { return fPathIdx2; }
      UInt_t                            GetPerfIdx1() const { return fPerfIdx1; }
      UInt_t                            GetPerfIdx2() const { return fPerfIdx2; }
      //
      UInt_t                       GetNSubsamples() const;
      const Event *                GetTrainingEvent(UInt_t i) const { return fTrainingEvents[i]; }
      const Event *                GetTrainingEvent(UInt_t i, UInt_t isub)  const;

      // Loss function; Huber loss eq 33
      Double_t LossFunction( const Event& e ) const;

      // Empirical risk, including regularization
      Double_t Risk(UInt_t ibeg, UInt_t iend) const;
    
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

      // estimate the optimum scoring function
      void CalcFStar(UInt_t ibeg, UInt_t iend);

      // estimate of binary error rate
      Double_t ErrorRateBin(UInt_t ibeg, UInt_t iend);

      // estimate of scale average error rate
      Double_t ErrorRateReg(UInt_t ibeg, UInt_t iend);

      // estimate of the distance between current and optimum risk
      Double_t ErrorRateRisk(UInt_t ibeg, UInt_t iend);

      // estimate 1-area under ROC
      Double_t ErrorRateRoc(UInt_t ibeg, UInt_t iend);

      // make gradient vector (eq 44 in ref 1)
      void MakeGradientVector(UInt_t ibeg, UInt_t iend);

      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      void UpdateCoefficients();

      // calculate average of responses (initial estimate of a0)
      Double_t CalcAverageResponse(UInt_t ibeg, UInt_t iend);

      RuleFit      * fRuleFit;      // rule fit
      RuleEnsemble * fRuleEnsemble; // rule ensemble
      std::vector<const Event *>  fTrainingEvents; // ptr to training events
      // Event indecis for path/validation - TODO: should let the user decide
      // Now it is just a simple one-fold cross validation.
      UInt_t                fPathIdx1;       // first event index for path search
      UInt_t                fPathIdx2;       // last event index for path search
      UInt_t                fPerfIdx1;       // first event index for performance evaluation
      UInt_t                fPerfIdx2;       // last event index for performance evaluation

      std::vector<Double_t> fGradVec;        // gradient vector - dimension = number of rules in ensemble
      std::vector<Double_t> fGradVecLin;     // gradient vector - dimension = number of variables
      std::vector<Double_t> fGradVecMin;     // gradient vector, min
      Double_t              fGradOfs;        // gradient for offset
      //
      Double_t              fGDTau;          // threshold parameter (tau in eq 26, ref 1)
      Double_t              fGDPathStep;     // step size along path (delta nu in eq 22, ref 1)
      Int_t                 fGDNPathSteps;   // number of path steps
      Double_t              fGDErrNsigma;    // threshold difference from minimum (n sigmas)
      //
      std::vector<Double_t> fFstar;          // vector of F*() - filled in CalcFStar()
      Double_t              fFstarMedian;    // median value of F*() using 
      //
      TTree                *fGDNtuple;       // Gradient path ntuple, contains params for each step along the path
      Double_t              fNTRisk;         // GD path: risk
      Double_t              fNTErrorRate;    // GD path: error rate (or performance)
      Double_t              fNTNuval;        // GD path: value of nu
      Double_t              fNTCoefRad;      // GD path: 'radius' of all rulecoeffs
      Double_t              fNTOffset;       // GD path: model offset
      Double_t             *fNTCoeff;        // GD path: rule coefficients
      Double_t             *fNTLinCoeff;     // GD path: linear coefficients

      Double_t              fsigave;         // Sigma of current signal score function F(sig)
      Double_t              fsigrms;         // Rms of F(sig)
      Double_t              fbkgave;         // Average of F(bkg)
      Double_t              fbkgrms;         // Rms of F(bkg)

   private:

      mutable MsgLogger     fLogger;         // message logger
   };

   // --------------------------------------------------------

   class AbsValue {

   public:

      bool operator()( Double_t first, Double_t second ) const { return TMath::Abs(first) < TMath::Abs(second); }
   };
}


#endif
