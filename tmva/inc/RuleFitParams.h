// @(#)root/tmva $Id: RuleFitParams.h,v 1.10 2007/02/02 19:16:05 brun Exp $
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

#if ROOT_VERSION_CODE >= 364802
#ifndef ROOT_TMathBase
#include "TMathBase.h"
#endif
#else
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#endif

#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
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

      // set message type
      void SetMsgType( EMsgType t ) { fLogger.SetMinType(t); }

      // set RuleFit ptr
      void SetRuleFit( RuleFit *rf )    { fRuleFit = rf; Init(); }
      //
      // GD path: set N(path steps)
      void SetGDNPathSteps( Int_t np )  { fGDNPathSteps = np; }

      // GD path: set path step size
      void SetGDPathStep( Double_t s )  { fGDPathStep = s; }

      // GD path: set tau search range
      void SetGDTau( Double_t t0, Double_t t1 )
      {
         fGDTauMin = (t0>1.0 ? 1.0:(t0<0.0 ? 0.0:t0));
         fGDTauMax = (t1>1.0 ? 1.0:(t1<0.0 ? 0.0:t1));
         if (fGDTauMax<fGDTauMin) fGDTauMax = fGDTauMin;
      }

      // GD path: set number of steps in tau search range
      void SetGDTauScan( UInt_t n )        { fGDTauScan = n; }

      // GD path: set tau
      void SetGDTau( Double_t t ) { fGDTau = t; }


      void SetGDErrScale( Double_t s ) { fGDErrScale = s; }
      void SetGDNTau( UInt_t n )        { fGDNTau = n; fGDTauVec.resize(n); }
      void SetGDTauVec( std::vector<Double_t> & tau ) { fGDNTau = tau.size(); fGDTauMin = tau[0]; fGDTauMax = tau[fGDNTau-1]; fGDTauVec = tau; }

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

      // initialize GD path
      void InitGD();

      // find best tau and return the number of scan steps used
      Int_t FindGDTau();

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
      Double_t ErrorRateRocRaw( std::vector<Double_t> & sFsig, std::vector<Double_t> & sFbkg );
      Double_t ErrorRateRoc(UInt_t ibeg, UInt_t iend);
      void     ErrorRateRocTst(UInt_t ibeg, UInt_t iend);

      // make gradient vector (eq 44 in ref 1)
      void MakeGradientVector(UInt_t ibeg, UInt_t iend);

      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      void UpdateCoefficients();

      // calculate average of responses of F
      Double_t CalcAverageResponse();
      Double_t CalcAverageResponseOLD(UInt_t ibeg, UInt_t iend);

      // calculate average of true response (initial estimate of a0)
      Double_t CalcAverageTruth(UInt_t ibeg, UInt_t iend);

      // calculate the average of each variable over the range
      void EvaluateAverage(UInt_t ibeg, UInt_t iend);

      // the same as above but for the various tau
      void MakeTstGradientVector(UInt_t ibeg, UInt_t iend);
      void UpdateTstCoefficients();
      void CalcTstAverageResponse();
      //      void CalcTstAverageResponse(UInt_t ibeg, UInt_t iend);

      RuleFit             * fRuleFit;      // rule fit
      RuleEnsemble        * fRuleEnsemble; // rule ensemble
      std::vector<const Event *>  fTrainingEvents; // ptr to training events
      //
      UInt_t                fNRules;       // number of rules
      UInt_t                fNLinear;      // number of linear terms
      //
      // Event indecis for path/validation - TODO: should let the user decide
      // Now it is just a simple one-fold cross validation.
      //
      UInt_t                fPathIdx1;       // first event index for path search
      UInt_t                fPathIdx2;       // last event index for path search
      UInt_t                fPerfIdx1;       // first event index for performance evaluation
      UInt_t                fPerfIdx2;       // last event index for performance evaluation
      std::vector<Double_t> fAverageSelector; // average of each variable over a range set by EvaluateAverage()
      std::vector<Double_t> fAverageRule;     // average of each rule over a range set by EvaluateAverage()

      std::vector<Double_t> fGradVec;        // gradient vector - dimension = number of rules in ensemble
      std::vector<Double_t> fGradVecLin;     // gradient vector - dimension = number of variables
      std::vector< std::vector<Double_t> > fGradVecTst;    // gradient vector - one per tau
      std::vector< std::vector<Double_t> > fGradVecLinTst; // gradient vector, linear terms - one per tau
      //
      std::vector<Double_t> fGDErrSum;  // accumulated errors per tau
      std::vector< std::vector<Double_t> > fGDCoefTst;    // rule coeffs - one per tau
      std::vector< std::vector<Double_t> > fGDCoefLinTst; // linear coeffs - one per tau
      std::vector<Double_t> fGDOfsTst;       // offset per tau
      //
      Double_t              fAverageTruth;   // average truth, ie sum(y)/N, y=+-1
      //
      std::vector< Double_t > fGDTauVec;     // the tau's
      UInt_t                fGDNTau;         // number of tau-paths
      UInt_t                fGDTauScan;      // number scan for tau-paths
      Double_t              fGDTauMin;       // min threshold parameter (tau in eq 26, ref 1)
      Double_t              fGDTauMax;       // max threshold parameter (tau in eq 26, ref 1)
      Double_t              fGDTau;          // selected threshold parameter (tau in eq 26, ref 1)
      Double_t              fGDPathStep;     // step size along path (delta nu in eq 22, ref 1)
      Int_t                 fGDNPathSteps;   // number of path steps
      Double_t              fGDErrScale;     // stop scan at error = scale*errmin
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
