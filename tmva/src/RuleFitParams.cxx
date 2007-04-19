// @(#)root/tmva $Id: RuleFitParams.cxx,v 1.9 2007/02/02 20:18:02 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RuleFitParams                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
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

#include <iostream>
#include <algorithm>
#include <numeric>

#include "TTree.h"
#include "TMath.h"

#include "TMVA/Timer.h"
#include "TMVA/RuleFitParams.h"
#include "TMVA/RuleFit.h"
#include "TMVA/RuleEnsemble.h"
#include "TMVA/MethodRuleFit.h"

Bool_t gFIRSTTST=kTRUE;
Bool_t gFIRSTORG=kTRUE;
//_______________________________________________________________________
TMVA::RuleFitParams::RuleFitParams()
   : fRuleFit ( 0 )
   , fRuleEnsemble ( 0 )
   , fPathIdx1 ( 0 )
   , fPathIdx2 ( 0 )
   , fPerfIdx1 ( 0 )
   , fPerfIdx2 ( 0 )
   , fGDNTau     ( 1 )
   , fGDTauScan  ( 300 )
   , fGDTauMin   ( 0.6 )
   , fGDTauMax   ( 0.6 )
   , fGDTau      ( 0.6 )
   , fGDPathStep ( 0.01 )
   , fGDNPathSteps ( 1000 )
   , fGDErrScale ( 1.1 )
   , fGDNtuple ( 0 )
   , fNTOffset ( 0 )
   , fNTCoeff ( 0 )
   , fNTLinCoeff ( 0 )
   , fLogger( "RuleFit" )
{
   // constructor
   Init();
}
//_______________________________________________________________________
TMVA::RuleFitParams::~RuleFitParams()
{
   // destructor
   if (fNTCoeff)     { delete fNTCoeff; fNTCoeff = 0; }
   if (fNTLinCoeff)  { delete fNTLinCoeff;fNTLinCoeff = 0; }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::Init()
{
   // Initializes all parameters using the RuleEnsemble and the training tree
   if (fRuleFit==0) return;
   //
   fRuleEnsemble   = fRuleFit->GetRuleEnsemblePtr();
   fNRules         = fRuleEnsemble->GetNRules();
   fNLinear        = fRuleEnsemble->GetNLinear();
   fTrainingEvents = fRuleFit->GetTrainingEvents();
//    fPathIdx1 = 0;
//    fPathIdx2 = (2*fTrainingEvents.size())/3;
//    fPerfIdx1 = fPathIdx2+1;
//    fPerfIdx2 = fTrainingEvents.size()-1;
   UInt_t   nsub    = fRuleFit->GetNSubsamples();
   Double_t fsubUse = fRuleFit->GetMethodRuleFit()->GetSubSampleFraction();
   if (fsubUse<0.0) fsubUse=0.0;
   if (fsubUse>1.0) fsubUse=1.0;
   UInt_t   nsubUse = static_cast<UInt_t>(fsubUse*static_cast<Double_t>(nsub)); //hmm probably overkill;)
   if (nsubUse==0) nsubUse=1;
   //
   // always use full training sample for evaluation
   //
   UInt_t dummy;
   fRuleFit->GetSubsampleEvents(0,      fPerfIdx1, dummy);
   fRuleFit->GetSubsampleEvents(nsub-1, dummy,     fPerfIdx2);
   //
   // use first nsubUse sub-samples for the GD path finding
   //
   fRuleFit->GetSubsampleEvents(0,         fPathIdx1, dummy);
   fRuleFit->GetSubsampleEvents(nsubUse-1, dummy,     fPathIdx2);
   //
   fLogger << kVERBOSE << "path constr. - event index range = [ " << fPathIdx1 << ", " << fPathIdx2 << " ]" << Endl;
   fLogger << kVERBOSE << "error estim. - event index range = [ " << fPerfIdx1 << ", " << fPerfIdx2 << " ]" << Endl;
   //
   if (fRuleEnsemble->DoRules()) 
      fLogger << kINFO << "number of rules in ensemble = " << fNRules << Endl;
   else 
      fLogger << kINFO << "rules are disabled " << Endl;

   if (fRuleEnsemble->DoLinear())
      fLogger << kINFO << "number of linear terms = " << fNLinear << Endl;
   else
      fLogger << kINFO << "linear terms are disabled " << Endl;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::InitNtuple()
{
   // initializes the ntuple

   fGDNtuple= new TTree("MonitorNtuple_RuleFitParams","RuleFit path search");
   fGDNtuple->Branch("risk",    &fNTRisk,     "risk/D");
   fGDNtuple->Branch("error",   &fNTErrorRate,"error/D");
   fGDNtuple->Branch("nuval",   &fNTNuval,    "nuval/D");
   fGDNtuple->Branch("coefrad", &fNTCoefRad,  "coefrad/D");
   fGDNtuple->Branch("offset",  &fNTOffset,   "offset/D");
   //
   fNTCoeff    = (fNRules >0 ? new Double_t[fNRules]  : 0);
   fNTLinCoeff = (fNLinear>0 ? new Double_t[fNLinear] : 0);

   for (UInt_t i=0; i<fNRules; i++) {
      fGDNtuple->Branch(Form("a%d",i+1),&fNTCoeff[i],Form("a%d/D",i+1));
   }
   for (UInt_t i=0; i<fNLinear; i++) {
      fGDNtuple->Branch(Form("b%d",i+1),&fNTLinCoeff[i],Form("b%d/D",i+1));
   }
}

//_______________________________________________________________________
const std::vector< Int_t >  *TMVA::RuleFitParams::GetSubsampleEvents() const
{
   // accessor to the subsamples
   return &(fRuleFit->GetSubsampleEvents());
}

//_______________________________________________________________________
void TMVA::RuleFitParams::GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const
{
   // calls the Subsample Events
   fRuleFit->GetSubsampleEvents(sub,ibeg,iend);
}

//_______________________________________________________________________
UInt_t TMVA::RuleFitParams::GetNSubsamples() const
{
   // get the number of subsamples
   return fRuleFit->GetNSubsamples();
}

//_______________________________________________________________________
const TMVA::Event *TMVA::RuleFitParams::GetTrainingEvent(UInt_t i, UInt_t isub) const
{
   // accesses a training event
   return fRuleFit->GetTrainingEvent(i,isub);
}

//_______________________________________________________________________
void TMVA::RuleFitParams::EvaluateAverage(UInt_t ibeg, UInt_t iend)
{
   // evaluate the average of each variable and f(x) in the given range - TODO: not very nice!
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kWARNING << "EvaluateAverage() - invalid start/end indices!" << Endl;
      return;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   Double_t val;
   //   const std::vector< Rule *> *rules= &(fRuleEnsemble->GetRulesConst());
   //   UInt_t nsel = fRuleEnsemble->GetMethodRuleFit()->GetNvar();
   if (fNLinear>0) fAverageSelector.resize(fNLinear,0);
   if (fNRules>0)  fAverageRule.resize(fNRules,0);
   //
   for ( UInt_t i=ibeg; i<iend+1; i++) {
      // first cache rule/lin response
      val = fRuleEnsemble->EvalLinEvent(*((*events)[i]));
      val = fRuleEnsemble->EvalEvent(*((*events)[i]));
      // loop over linear terms
      for ( UInt_t sel=0; sel<fNLinear; sel++ ) {
         fAverageSelector[sel] += fRuleEnsemble->EvalLinEvent(sel,kTRUE); //(*events)[i]->GetVal(sel);
      }
      // loop over rules
      for (UInt_t r=0; r<fNRules; r++) {
         fAverageRule[r] += fRuleEnsemble->GetEventRuleVal(r); //(*rules)[r]->EvalEvent( *(*events)[i] ); // exclude coeff.
      }
   }
   // average variable
   for ( UInt_t sel=0; sel<fNLinear; sel++ ) {
      fAverageSelector[sel] = fAverageSelector[sel] / neve;
      //      fLogger << kVERBOSE << "AVESEL: " << sel << " -> " << fAverageSelector[sel] << Endl;
   }
   // average rule response, excl coeff
   for (UInt_t r=0; r<fNRules; r++) {
      fAverageRule[r] = fAverageRule[r] / neve;
      //      fLogger << kVERBOSE << "AVERUL: " << r << " -> " << fAverageRule[r] << Endl;
   }
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::LossFunction( const TMVA::Event& e ) const
{
   // Implementation of squared-error ramp loss function (eq 39,40 in ref 1)
   // This is used for binary Classifications where y = {+1,-1} for (sig,bkg)
   Double_t h = max( -1.0, min(1.0,fRuleEnsemble->EvalEvent( e )) );
   Double_t diff = (e.IsSignal()?1:-1) - h;
   //
   return diff*diff;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::Risk(UInt_t ibeg, UInt_t iend) const
{
   // risk asessment
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kWARNING << "makeGradientVector() - invalid start/end indices!" << Endl;
      return 0;
   }
   Double_t rval=0;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   for ( UInt_t i=ibeg; i<iend+1; i++) {
      rval += LossFunction( *(*events)[i] );
   }
   rval = rval/Double_t(neve);
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::Penalty() const
{
   // This is the "lasso" penalty
   Double_t rval=0;
   const std::vector<Double_t> *lincoeff = & (fRuleEnsemble->GetLinCoefficients());
   for (UInt_t i=0; i<fNRules; i++) {
      rval += TMath::Abs(fRuleEnsemble->GetRules(i)->GetCoefficient());
   }
   for (UInt_t i=0; i<fNLinear; i++) {
      rval += TMath::Abs((*lincoeff)[i]);
   }
   return rval;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::InitGD()
{
   // Initialize GD path search
   if (fGDNTau<1) {
      fGDNTau    = 1;
      fGDTauScan = 1;
   }
   // set all taus
   fGDTauVec.resize( fGDNTau );
   if (fGDNTau==1) {
      fGDTauVec[0] = fGDTau;
   } else {
      // set tau vector - TODO: make a smarter choice of range in tau
      Double_t dtau = (fGDTauMax - fGDTauMin)/static_cast<Double_t>(fGDNTau-1);
      for (UInt_t itau=0; itau<fGDNTau; itau++) {
         fGDTauVec[itau] = static_cast<Double_t>(itau)*dtau + fGDTauMin;
         if (fGDTauVec[itau]>1.0) fGDTauVec[itau]=1.0;
      }
   }
   // inititalize path search vectors

   fGradVec.clear();
   fGradVecLin.clear();
   fGradVecTst.clear();
   fGradVecLinTst.clear();
   fGDErrSum.clear();
   fGDOfsTst.clear();
   fGDCoefTst.clear();
   fGDCoefLinTst.clear();
   //
   // rules
   //
   fGDCoefTst.resize(fGDNTau);
   fGradVec.resize(fNRules,0);
   fGradVecTst.resize(fGDNTau);
   for (UInt_t i=0; i<fGDNTau; i++) {
      fGradVecTst[i].resize(fNRules,0);
      fGDCoefTst[i].resize(fNRules,0);
   }
   //
   // linear terms
   //
   fGDCoefLinTst.resize(fGDNTau);
   fGradVecLin.resize(fNLinear,0);
   fGradVecLinTst.resize(fGDNTau);
   for (UInt_t i=0; i<fGDNTau; i++) {
      fGradVecLinTst[i].resize(fNLinear,0);
      fGDCoefLinTst[i].resize(fNLinear,0);
   }
   //
   // error, coefs etc
   //
   fGDErrSum.resize(fGDNTau,0);
   fGDOfsTst.resize(fGDNTau,0);
   //
   // calculate average selectors and rule responses for the path sample size
   //
   EvaluateAverage( fPathIdx1, fPathIdx2 );

}

//_______________________________________________________________________
Int_t TMVA::RuleFitParams::FindGDTau()
{
   // This finds the regularization parameter tau by scanning several different paths
   if (fGDNTau==0) return 0;
   if (fGDTauScan==0) return 0;

   if (fGDOfsTst.size()<1)
      fLogger << kFATAL << "BUG! FindGDTau() has been called BEFORE InitGD()." << Endl;
   //
   fLogger << kINFO << "estimating the regularization parameter tau"
           << Endl;
   // Find how many points to scan and how often to calculate the error
   Int_t nscan = fGDTauScan; //std::min(static_cast<Int_t>(fGDTauScan),fGDNPathSteps);
   Int_t netst = 100; //std::min(nscan/10,100);
   //
   // loop over paths
   //
   TMVA::Timer timer( nscan, "RuleFit" );
   for (Int_t ip=0; ip<nscan; ip++) {
      // make gradvec
      MakeTstGradientVector( fPathIdx1, fPathIdx2 );
      // update coefs
      UpdateTstCoefficients();
      // estimate error and do the sum
      // do this at index=0, netst-1, 2*netst-1 ...
      if ( (ip==0) || ((ip+1)%netst==0) ) {
         if (fLogger.GetMinType()>kVERBOSE)
            timer.DrawProgressBar(ip);
         ErrorRateRocTst( fPerfIdx1, fPerfIdx2 );
         UInt_t itauMin=0;
         Double_t errmin = fGDErrSum[0];
         fLogger << kDEBUG << Form("TAU: %4d  ",ip);
         for (UInt_t itau=0; itau<fGDNTau; itau++) {
            fLogger << kDEBUG << Form("%4.4f  ",fGDErrSum[itau]);
            if (fGDErrSum[itau]<errmin) {
               itauMin = itau;
               errmin = fGDErrSum[itau];
            }
         }
         fLogger << kDEBUG << Endl;
         fLogger << kVERBOSE << Form("%4d",ip+1) << ". tau = " << Form("%4.4f",fGDTauVec[itauMin])
                 << " => error sum = " << fGDErrSum[itauMin] << Endl;
      }
      gFIRSTTST=kFALSE;
   }
   //
   // loop over each tau and find which one gave the lowest error
   //
   Double_t errmin = 1000.0;
   UInt_t   itauMin=0;
   for (UInt_t itau=0; itau<fGDNTau; itau++) {
      if (fGDErrSum[itau]<errmin) {
         itauMin = itau;
         errmin = fGDErrSum[itau];
      }
   }
   //
   // Set tau and coefs
   // Downscale tau slightly in order to avoid numerical problems
   //
   fGDTau = fGDTauVec[itauMin];
   fRuleEnsemble->SetCoefficients( fGDCoefTst[itauMin] );
   fRuleEnsemble->SetLinCoefficients( fGDCoefLinTst[itauMin] );
   fRuleEnsemble->SetOffset( fGDOfsTst[itauMin] );
   fLogger << kINFO << "best path found with tau = " << Form("%4.4f",fGDTau)
           << " after " << timer.GetElapsedTime() << " s" << Endl;

   return nscan;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeGDPath()
{
   // The following finds the gradient directed path in parameter space.
   // More work is needed... FT, 24/9/2006
   // The algorithm is currently as follows:
   // <***if not otherwise stated, the sample used below is [fPathIdx1,fPathIdx2]***>
   // 1. Set offset to -average(y(true)) and all coefs=0 => average of F(x)==0
   // 2. FindGDTau() : start scanning using several paths defined by different tau
   //                  choose the tau yielding the best path               
   // 3. start the scanning the chosen path
   // 4. check error rate at a given frequency
   //    data used for check: [fPerfIdx1,fPerfIdx2]
   // 5. stop when either of the following onditions are fullfilled:
   //    a. loop index==fGDNPathSteps
   //    b. error > fGDErrScale*errmin
   //    c. only in DEBUG mode: risk is not monotoneously decreasing
   //
   // The algorithm will warn if:
   //   I. the error rate was still decreasing when loop finnished -> increase fGDNPathSteps!
   //  II. minimum was found at an early stage -> decrease fGDPathStep
   // III. DEBUG: risk > previous risk -> entered caotic region (regularization is too small)
   //

   fLogger << kINFO << "GD path scan - the scan stops when the max num. of steps is reached or a min is found"
           << Endl;
   fLogger << kVERBOSE << "number of events used per path step = " << fPathIdx2-fPathIdx1+1 << Endl;
   fLogger << kVERBOSE << "number of events used for error estimation = " << fPerfIdx2-fPerfIdx1+1 << Endl;

   // check if debug mode
   const Bool_t isVerbose = (fLogger.GetMinType()<=kVERBOSE);
   const Bool_t isDebug   = (fLogger.GetMinType()<=kDEBUG);

   // init GD parameters and clear coeff vectors
   InitGD();

   // initial estimate; all other a(i) are zero
   fLogger << kVERBOSE << "creating GD path"  << Endl;
   fLogger << kVERBOSE << "  N(steps)     = "   << fGDNPathSteps << Endl;
   fLogger << kVERBOSE << "  step         = "   << fGDPathStep   << Endl;
   fLogger << kVERBOSE << "  N(tau)       = "   << fGDNTau       << Endl;
   fLogger << kVERBOSE << "  N(tau steps) = "   << fGDTauScan    << Endl;
   fLogger << kVERBOSE << "  tau range    = [ " << fGDTauVec[0]  << " , " << fGDTauVec[fGDNTau-1] << " ]" << Endl;

   // init ntuple
   if (isDebug) InitNtuple();

   // DEBUG: risk scan
   Int_t    nbadrisk=0;                // number of points where risk(i+1)>risk(i)
   Double_t trisk=0;                   // time per risk evaluation
   Double_t strisk=0;                  // total time
   Double_t rprev=1e32;                // previous risk

   // parameters set at point with min error
   Double_t              errmin=1e32;  // min error
   Double_t              riskMin=0;    // risk
   Int_t                 indMin=-1;    // index
   std::vector<Double_t> coefsMin;     // rule coefs
   std::vector<Double_t> lincoefsMin;  // linear coefs
   Double_t              offsetMin;    // offset


   // DEBUG: timing
   clock_t  t0=0;
   Double_t tgradvec;
   Double_t tupgrade;
   Double_t tperf;
   Double_t stgradvec=0;
   Double_t stupgrade=0;
   Double_t stperf=0;

   // linear regression to estimate slope of error rate evolution
   const UInt_t npreg=5;
   std::vector<Double_t> valx;
   std::vector<Double_t> valy;
   std::vector<Double_t> valxy;

   // loop related
   Bool_t  docheck;          // true if an error rate check is to be done
   Int_t   iloop=0;          // loop index
   Bool_t  found=kFALSE;     // true if minimum is found
   Bool_t  riskFlat=kFALSE;  // DEBUG: flag is true if risk evolution behaves badly
   Bool_t  done = kFALSE;    // flag that the scan is done

   // calculate how often to check error rate
   int imod = fGDNPathSteps/100;
   if (imod<100) imod = std::min(100,fGDNPathSteps);
   if (imod>100) imod=100;

   // reset coefficients
   fAverageTruth = -CalcAverageTruth(fPathIdx1, fPathIdx2);
   offsetMin     = fAverageTruth;
   fRuleEnsemble->SetOffset(offsetMin);
   fRuleEnsemble->ClearCoefficients(0);
   fRuleEnsemble->ClearLinCoefficients(0);
   for (UInt_t i=0; i<fGDOfsTst.size(); i++) {
      fGDOfsTst[i] = offsetMin;
   }
   fLogger << kVERBOSE << "obtained initial offset = " << offsetMin << Endl;

   // find the best tau - returns the number of steps performed in scan
   Int_t nprescan = FindGDTau();

   //
   //
   // calculate F*
   //
   //   CalcFStar(fPerfIdx1, fPerfIdx2);
   //

   // set some ntuple values
   fNTRisk = rprev;
   fNTCoefRad = -1.0;
   fNTErrorRate = 0;

   // a local flag indicating for what reason the search was stopped
   Int_t stopCondition=0;

   // start loop with timer
   TMVA::Timer timer( fGDNPathSteps, "RuleFit" );
   while (!done) {
      // Make gradient vector (eq 44, ref 1)
      if (isVerbose) t0 = clock();
      MakeGradientVector(fPathIdx1, fPathIdx2);
      if (isVerbose) {
         tgradvec = Double_t(clock()-t0)/CLOCKS_PER_SEC;
         stgradvec += tgradvec;
      }
      
      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      if (isVerbose) t0 = clock();
      UpdateCoefficients();
      if (isVerbose) {
         tupgrade = Double_t(clock()-t0)/CLOCKS_PER_SEC;
         stupgrade += tupgrade;
      }

      // don't check error rate every loop
      docheck = ((iloop==0) ||((iloop+1)%imod==0));

      if (docheck) {
         // draw progressbar only if not debug
         if (!isVerbose)
            timer.DrawProgressBar(iloop);
         fNTNuval = Double_t(iloop)*fGDPathStep;
         fNTRisk = 0.0;

         // check risk evolution

         if (isDebug) FillCoefficients();
         fNTCoefRad = fRuleEnsemble->CoefficientRadius();
         
         // calculate risk
         t0 = clock();
         fNTRisk = Risk(fPathIdx1, fPathIdx2);
         trisk =  Double_t(clock()-t0)/CLOCKS_PER_SEC;
         strisk += trisk;
         //
         // Check for an increase in risk.
         // Such an increase would imply that the regularization is too small.
         // Stop the iteration if this happens.
         //
         if (fNTRisk>=rprev) {
            if (fNTRisk>rprev) {
               nbadrisk++;
               fLogger << kWARNING << "Risk(i+1)>=Risk(i) in path" << Endl;
               riskFlat=(nbadrisk>3);
               if (riskFlat) {
                  fLogger << kWARNING << "chaotic behaviour of risk evolution => the regularization is too small" << Endl;
                  fLogger << kWARNING << "--- STOPPING MINIMIZATION ---" << Endl;
                  fLogger << kWARNING << "this may be OK if minimum is already found" << Endl;
               }
            }
         }
         rprev = fNTRisk;
         //
         // Estimate the error rate using cross validation
         // Well, not quite full cross validation since we only
         // use ONE model.
         //
         if (isVerbose) t0 = clock();
         fNTErrorRate = 0;

         // Check error rate
         Double_t errroc  = ErrorRateRoc(fPerfIdx1, fPerfIdx2);

         //
         fNTErrorRate = errroc;
         //
         if (isVerbose) {
            tperf = Double_t(clock()-t0)/CLOCKS_PER_SEC;
            stperf +=tperf;
         }
         //
         // Always take the last min.
         // For each step the risk is reduced.
         //
         if (fNTErrorRate<=errmin) {
            errmin  = fNTErrorRate;
            riskMin = fNTRisk;
            indMin  = iloop;
            fRuleEnsemble->GetCoefficients(coefsMin);
            lincoefsMin = fRuleEnsemble->GetLinCoefficients();
            offsetMin   = fRuleEnsemble->GetOffset();
         }
         if ( fNTErrorRate > fGDErrScale*errmin) found = kTRUE;
         //
         // check slope of last couple of points
         //
         if (valx.size()==npreg) {
            valx.erase(valx.begin());
            valy.erase(valy.begin());
            valxy.erase(valxy.begin());
         }
         valx.push_back(fNTNuval);
         valy.push_back(fNTErrorRate);
         valxy.push_back(fNTErrorRate*fNTNuval);

         gFIRSTORG=kFALSE;

         //
         if (isDebug) fGDNtuple->Fill();
         if (isVerbose) {
            fLogger << kVERBOSE << "ParamsIRE : "
                    << setw(10)
                    << Form("%8d",iloop+1) << " "
                    << Form("%4.4f",fNTRisk) << " "
                    << Form("%4.4f",errroc)  << " "
                    << Form("%4.4f",fsigave+fbkgave) << " "
                    << Form("%4.4f",fsigave) << " "
                    << Form("%4.4f",fsigrms) << " "
                    << Form("%4.4f",fbkgave) << " "
                    << Form("%4.4f",fbkgrms) << " "
                    << Form("%4.4f",fRuleEnsemble->CoefficientRadius())
                    << Endl;
         }
      }
      iloop++;
      // Stop iteration under various conditions
      // * The condition R(i+1)<R(i) is no longer true (when then implicit regularization is too weak)
      // * If the current error estimate is > factor*errmin (factor = 1.1)
      // * We have reach the last step...
      Bool_t endOfLoop = (iloop==fGDNPathSteps);
      if ( ((riskFlat) || (endOfLoop)) && (!found) ) {
         if (riskFlat) {
            stopCondition = 1;
         } else if (endOfLoop) {
            stopCondition = 2;
         }
         if (indMin<0) {
            fLogger << kWARNING << "BUG TRAP: should not be here - still, this bug is harmless;)" << Endl;
            errmin  = fNTErrorRate;
            riskMin = fNTRisk;
            indMin  = iloop;
            fRuleEnsemble->GetCoefficients(coefsMin);
            lincoefsMin = fRuleEnsemble->GetLinCoefficients();
            offsetMin   = fRuleEnsemble->GetOffset();
         }
         found = kTRUE;
      }
      done = (found);
   }
   fLogger << kINFO << "minimization elapsed time : " << timer.GetElapsedTime() << " s" << Endl;
   fLogger << kINFO << "------------------------------------------------------------------" << Endl;
   fLogger << kINFO << "Found minimum at step " << indMin+1 << " with area under ROC = " << errmin << Endl;
   fLogger << kINFO << "Reason for ending loop : ";
   switch (stopCondition) {
   case 0:
      fLogger << kINFO << "clear minima found";
      break;
   case 1:
      fLogger << kINFO << "chaotic behaviour of risk";
      break;
   case 2:
      fLogger << kINFO << "end of loop reached";
      break;
   default:
      fLogger << kINFO << "unknown!";
      break;
   }
   fLogger << Endl;
   fLogger << kINFO << "------------------------------------------------------------------" << Endl;

   // check if early minima - might be an indication of too large stepsize
   if ( Double_t(indMin)/Double_t(nprescan+fGDNPathSteps) < 0.05 ) {
      fLogger << kWARNING << "reached minimum early in the search - check results and maybe decrease GDStep size"
              << Endl;
   }
   //
   // quick check of the sign of the slope for the last npreg points
   //
   Double_t sumx  = std::accumulate( valx.begin(), valx.end(), Double_t() );
   Double_t sumxy = std::accumulate( valxy.begin(), valxy.end(), Double_t() );
   Double_t sumy  = std::accumulate( valy.begin(), valy.end(), Double_t() );
   Double_t slope = Double_t(valx.size())*sumxy - sumx*sumy;
   if (slope<0) {
      fLogger << kWARNING << "the error rate was still decreasing when the end of the path was reached;" << Endl;
      fLogger << kWARNING << "increase number of steps (GDNSteps)." << Endl;
   }
   //
   // set coefficients
   //
   if (found) {
      fRuleEnsemble->SetCoefficients( coefsMin );
      fRuleEnsemble->SetLinCoefficients( lincoefsMin );
      fRuleEnsemble->SetOffset( offsetMin );
   } else {
      fLogger << kFATAL << "BUG TRAP: minimum not found in MakeGDPath()" << Endl;
   }

   //
   // print timing info (VERBOSE mode)
   //
   if (isVerbose) {
      Double_t stloop  = strisk +stupgrade + stgradvec + stperf;
      fLogger << kVERBOSE << "Timing per loop (ms):" << Endl;
      fLogger << kVERBOSE << "   gradvec = " << 1000*stgradvec/iloop << Endl;
      fLogger << kVERBOSE << "   upgrade = " << 1000*stupgrade/iloop << Endl;
      fLogger << kVERBOSE << "   risk    = " << 1000*strisk/iloop    << Endl;
      fLogger << kVERBOSE << "   perf    = " << 1000*stperf/iloop    << Endl;
      fLogger << kVERBOSE << "   loop    = " << 1000*stloop/iloop    << Endl;
   }
   // write ntuple (DEBUG)
   if (isDebug) fGDNtuple->Write();
}

//_______________________________________________________________________
void TMVA::RuleFitParams::FillCoefficients()
{
   // helper function to store the rule coefficients in local arrays

   fNTOffset = fRuleEnsemble->GetOffset();
   //
   for (UInt_t i=0; i<fNRules; i++) {
      fNTCoeff[i] = fRuleEnsemble->GetRules(i)->GetCoefficient();
   }
   for (UInt_t i=0; i<fNLinear; i++) {
      fNTLinCoeff[i] = fRuleEnsemble->GetLinCoefficients(i);
   }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::CalcFStar(UInt_t ibeg, UInt_t iend)
{
   // Estimates F* (optimum scoring function) for all events for the given sets.
   // The result is used in ErrorRateReg().
   //
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<CalcFStar> invalid start/end indices!" << Endl;
      return;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   fFstar.clear();
   std::vector<Double_t> fstarSorted;
   Double_t fstarVal;
   // loop over all events and estimate F* for each event
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      fstarVal = fRuleEnsemble->FStar(e);
      fFstar.push_back(fstarVal);
      fstarSorted.push_back(fstarVal);
      if (isnan(fstarVal)) fLogger << kFATAL << "F* is NAN!" << Endl;
   }
   // sort F* and find median
   std::sort( fstarSorted.begin(), fstarSorted.end() );
   UInt_t ind = neve/2;
   if (neve&1) { // odd number of events
      fFstarMedian = 0.5*(fstarSorted[ind]+fstarSorted[ind-1]);
   } 
   else { // even
      fFstarMedian = fstarSorted[ind];
   }
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateReg(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters
   // This code is pretty messy at the moment.
   // Cleanup is needed.
   //
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<ErrorRateReg> invalid start/end indices!" << Endl;
      return 1000000.0;
   }
   if (fFstar.size()!=neve) {
      fLogger << kFATAL << "--- RuleFitParams::ErrorRateReg() - F* not initialised! BUG!!!"
              << " Fstar.size() = " << fFstar.size() << " , N(events) = " << neve << Endl;
   }
   //
   Double_t sF;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   Double_t sumdf = 0;
   Double_t sumdfmed = 0;
   //
   // A bit messy here.
   // I believe the binary error classification is appropriate here.
   // The problem is stability.
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      sF = fRuleEnsemble->EvalEvent( e );
      // scaled abs error, eq 20 in RuleFit paper
      sumdf += TMath::Abs(fFstar[i-ibeg] - sF);
      sumdfmed += TMath::Abs(fFstar[i-ibeg] - fFstarMedian);
   }
   // scaled abs error, eq 20
   // This error (df) is large - need to think on how to compensate...
   //
   return sumdf/sumdfmed;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateRisk(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters
   // This code is pretty messy at the moment.
   // Cleanup is needed.
   //
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<ErrorRateRisk> invalid start/end indices!" << Endl;
      return 1000000.0;
   }
   //
   Double_t sF, sFstar, ytrue;
   Double_t sR;
   const std::vector<const Event *> *events = GetTrainingEvents();
   Double_t sumRstar=0;
   Double_t sumR=0;
   //
   // A bit messy here.
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      sF     = std::max(-1.0,std::min(fRuleEnsemble->EvalEvent( e ),1.0));
      sFstar = std::max(-1.0,std::min(fRuleEnsemble->FStar(),1.0));
      ytrue = (e.IsSignal() ? +1.0:-1.0);
      //
      sR        = (ytrue-sFstar);
      sumRstar += sR*sR;
      sR        = (ytrue-sF);
      sumR     += sR*sR;
   }

   return TMath::Abs(sumR-sumRstar)/sumRstar;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateBin(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters
   // It uses a binary estimate of (y-F*(x))
   // (y-F*(x)) = (Num of events where sign(F)!=sign(y))/Neve
   // y = {+1 if event is signal, -1 otherwise}
   //
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<ErrorRateBin> invalid start/end indices!" << Endl;
      return 1000000.0;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   Double_t sumdfbin = 0;
   Double_t dneve = Double_t(neve);
   Int_t signF, signy;
   Double_t sF;
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      sF     = fRuleEnsemble->EvalEvent( e );
      //      Double_t sFstar = fRuleEnsemble->FStar(e); // THIS CAN BE CALCULATED ONCE!
      signF = (sF>0 ? +1:-1);
      //      signy = (sFStar>0 ? +1:-1);
      signy = (e.IsSignal() ? +1:-1);
      sumdfbin += TMath::Abs(Double_t(signF-signy))*0.5;
   }
   Double_t f = sumdfbin/dneve;
   //   Double_t   df = f*sqrt((1.0/sumdfbin) + (1.0/dneve));
   return f;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateRocRaw( std::vector<Double_t> & sFsig,
                                               std::vector<Double_t> & sFbkg )

{
   //
   // Estimates the error rate with the current set of parameters.
   // It calculates the area under the bkg rejection vs signal efficiency curve.
   // The value returned is 1-area.
   //
   std::sort(sFsig.begin(), sFsig.end());
   std::sort(sFbkg.begin(), sFbkg.end());
   const Double_t minsig = sFsig.front();
   const Double_t minbkg = sFbkg.front();
   const Double_t maxsig = sFsig.back();
   const Double_t maxbkg = sFbkg.back();
   const Double_t minf = std::min(minsig,minbkg);
   const Double_t maxf = std::max(maxsig,maxbkg);
   const Int_t    nsig = Int_t(sFsig.size());
   const Int_t    nbkg = Int_t(sFbkg.size());
   const Int_t    np   = std::min((nsig+nbkg)/4,50);
   const Double_t df   = (maxf-minf)/(np-1);
   //
   // calculate area under rejection/efficiency curve
   //
   Double_t fcut;
   std::vector<Double_t>::const_iterator indit;
   Int_t nrbkg;
   Int_t nesig;
   Int_t pnesig=0;
   Double_t rejb=0;
   Double_t effs=1.0;
   Double_t prejb=0;
   Double_t peffs=1.0;
   Double_t drejb;
   Double_t deffs;
   Double_t area=0;
   Int_t    npok=0;
   //
   // loop over range of F [minf,maxf]
   //
   for (Int_t i=0; i<np; i++) {
      fcut = minf + df*Double_t(i);
      indit = std::find_if( sFsig.begin(), sFsig.end(), std::bind2nd(std::greater_equal<Double_t>(), fcut));
      nesig = sFsig.end()-indit; // number of sig accepted with F>cut
      if (TMath::Abs(pnesig-nesig)>0) {
         npok++;
         indit = std::find_if( sFbkg.begin(), sFbkg.end(), std::bind2nd(std::greater_equal<Double_t>(), fcut));
         nrbkg = indit-sFbkg.begin(); // number of bkg rejected with F>cut
         rejb = Double_t(nrbkg)/Double_t(nbkg);
         effs = Double_t(nesig)/Double_t(nsig);
         //
         drejb = rejb-prejb;
         deffs = effs-peffs;
         area += 0.5*TMath::Abs(deffs)*(rejb+prejb); // trapezoid
         prejb = rejb;
         peffs = effs;
      }
      pnesig = nesig;
   }
   area += 0.5*(1+rejb)*effs; // extrapolate to the end point

   return (1.0-area);
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateRoc(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters.
   // It calculates the area under the bkg rejection vs signal efficiency curve.
   // The value returned is 1-area.
   //

   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<ErrorRateRoc> invalid start/end indices!" << Endl;
      return 1000000.0;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   Double_t sF;
   //
   std::vector<Double_t> sFsig;
   std::vector<Double_t> sFbkg;
   Double_t sumfsig=0;
   Double_t sumfbkg=0;
   Double_t sumf2sig=0;
   Double_t sumf2bkg=0;
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      sF     = fRuleEnsemble->EvalEvent( e );
      if (e.IsSignal()) {
         sFsig.push_back(sF);
         sumfsig  +=sF;
         sumf2sig +=sF*sF;
      } else {
         sFbkg.push_back(sF);
         sumfbkg  +=sF;
         sumf2bkg +=sF*sF;
      }
   }
   fsigave = sumfsig/sFsig.size();
   fbkgave = sumfbkg/sFbkg.size();
   fsigrms = sqrt((sumf2sig - (sumfsig*sumfsig/sFsig.size()))/(sFsig.size()-1));
   fbkgrms = sqrt((sumf2bkg - (sumfbkg*sumfbkg/sFbkg.size()))/(sFbkg.size()-1));
   //
   return ErrorRateRocRaw( sFsig, sFbkg );
}

//_______________________________________________________________________
void TMVA::RuleFitParams::ErrorRateRocTst(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters.
   // It calculates the area under the bkg rejection vs signal efficiency curve.
   // The value returned is 1-area.
   //
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<ErrorRateRocTst> invalid start/end indices!" << Endl;
      return;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   //   std::vector<Double_t> sF;
   Double_t sF;
   std::vector< std::vector<Double_t> > sFsig;
   std::vector< std::vector<Double_t> > sFbkg;
   //
   sFsig.resize( fGDNTau );
   sFbkg.resize( fGDNTau );
   //   sF.resize( fGDNTau ); 

   for (UInt_t i=ibeg; i<iend+1; i++) {
      for (UInt_t itau=0; itau<fGDNTau; itau++) {
         if (itau==0) sF = fRuleEnsemble->EvalEvent( *(*events)[i], fGDOfsTst[itau], fGDCoefTst[itau], fGDCoefLinTst[itau] );
         else         sF = fRuleEnsemble->EvalEvent(                fGDOfsTst[itau], fGDCoefTst[itau], fGDCoefLinTst[itau] );
         if ((*events)[i]->IsSignal()) {
            sFsig[itau].push_back(sF);
         } else {
            sFbkg[itau].push_back(sF);
         }
      }
   }
   Double_t err;
   // name is fGDErrSum but it is the current value -> CHANGE NAME!
   for (UInt_t itau=0; itau<fGDNTau; itau++) {
      err = ErrorRateRocRaw( sFsig[itau], sFbkg[itau] );
      fGDErrSum[itau] = err;
   }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeTstGradientVector( UInt_t ibeg, UInt_t iend )
{
   // make test gradient vector for all tau
   // same algorithm as MakeGradientVector()
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<MakeTstGradientVector> invalid start/end indices!" << Endl;
      return;
   }
   //
   Double_t norm   = 2.0/Double_t(neve);
   //
   const std::vector<const Event *> *events = GetTrainingEvents();

   // Clear gradient vectors
   for (UInt_t itau=0; itau<fGDNTau; itau++) {
      for (UInt_t ir=0; ir<fNRules; ir++) {
         fGradVecTst[itau][ir]=0;
      }
      for (UInt_t il=0; il<fNLinear; il++) {
         fGradVecLinTst[itau][il]=0;
      }
   }
   //
   Double_t val; // temp store
   Double_t sF;   // score function value
   Double_t r;   // eq 35, ref 1
   Double_t y;   // true score (+1 or -1)
   //
   // Loop over all events
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event *e = (*events)[i];
      for (UInt_t itau=0; itau<fGDNTau; itau++) { // loop over tau
         if (itau==0) sF = fRuleEnsemble->EvalEvent( *e, fGDOfsTst[itau], fGDCoefTst[itau], fGDCoefLinTst[itau] );
         else         sF = fRuleEnsemble->EvalEvent(     fGDOfsTst[itau], fGDCoefTst[itau], fGDCoefLinTst[itau] );
         if (TMath::Abs(sF)<1.0) {
            r = 0;
            y = (e->IsSignal()?1.0:-1.0);
            r = y - sF;

            // rule gradient vector
            for (UInt_t ir=0; ir<fNRules; ir++) {
               val = fRuleEnsemble->GetEventRuleVal(ir); // filled by EvalEvent() call above
               if (val>0) fGradVecTst[itau][ir] += norm*r*val;
            }
            // linear terms
            for (UInt_t il=0; il<fNLinear; il++) {
               fGradVecLinTst[itau][il] += norm*r*fRuleEnsemble->EvalLinEvent( il, kTRUE );
            }
         } // if (TMath::Abs(F)<xxx)
      }
   }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::UpdateTstCoefficients()
{
   // Establish maximum gradient for rules, linear terms and the offset
   // for all taus TODO: do not need index range!
   //
   Double_t maxr, maxl, cthresh, val;
   // loop over all taus
   for (UInt_t itau=0; itau<fGDNTau; itau++) {
      // find max gradient
      maxr = ( (fNRules>0 ? 
                TMath::Abs(*(std::max_element( fGradVecTst[itau].begin(), fGradVecTst[itau].end(), TMVA::AbsValue()))):0) );
      maxl = ( (fNLinear>0 ? 
                TMath::Abs(*(std::max_element( fGradVecLinTst[itau].begin(), fGradVecLinTst[itau].end(), TMVA::AbsValue()))):0) );

      // Use the maximum as a threshold
      Double_t maxv = (maxr>maxl ? maxr:maxl);
      cthresh = maxv * fGDTauVec[itau];

      // Add to offset, if gradient is large enough:
      // Loop over the gradient vector and move to next set of coefficients
      // size of GradVec (and GradVecLin) should be 0 if learner is disabled
      //
      // Step-size is divided by 10 when looking for the best path.
      //
      if (maxv>0) {
         const Double_t stepScale=1.0;
         for (UInt_t i=0; i<fNRules; i++) {
            val = fGradVecTst[itau][i];

            if (TMath::Abs(val)>=cthresh) {
               fGDCoefTst[itau][i] += fGDPathStep*val*stepScale;
            }
         }
         // Loop over the gradient vector for the linear part and move to next set of coefficients
         for (UInt_t i=0; i<fNLinear; i++) {
            val = fGradVecLinTst[itau][i];
            if (TMath::Abs(val)>=cthresh) {
               fGDCoefLinTst[itau][i] += fGDPathStep*val*stepScale/fRuleEnsemble->GetLinNorm(i);
            }
         }
      }
      // set the offset
      //   CalcTstAverageResponse( ibeg, iend );
      CalcTstAverageResponse();
   }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeGradientVector( UInt_t ibeg, UInt_t iend )
{
   // make gradient vector
   //
   //  clock_t t0=clock();
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<MakeGradientVector> invalid start/end indices!" << Endl;
      return;
   }
   //
   Double_t norm   = 2.0/Double_t(neve);
   //
   const std::vector<const Event *> *events = GetTrainingEvents();

   // Clear gradient vectors
   for (UInt_t ir=0; ir<fNRules; ir++) {
      fGradVec[ir]=0;
   }
   for (UInt_t il=0; il<fNLinear; il++) {
      fGradVecLin[il]=0;
   }
   //
   Double_t val; // temp store
   Double_t sF;   // score function value
   Double_t r;   // eq 35, ref 1
   Double_t y;   // true score (+1 or -1)
   //
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event *e = (*events)[i];
      sF = fRuleEnsemble->EvalEvent( *e );
      if (TMath::Abs(sF)<1.0) {
         r = 0;
         y = (e->IsSignal()?1.0:-1.0);
         r = y - sF;

         // rule gradient vector
         for (UInt_t ir=0; ir<fNRules; ir++) {
            val = fRuleEnsemble->GetEventRuleVal(ir); // filled by EvalEvent() call above
            if (val>0) fGradVec[ir] += norm*r*val;
         }
         // linear terms
         for (UInt_t il=0; il<fNLinear; il++) {
            fGradVecLin[il] += norm*r*fRuleEnsemble->EvalLinEvent( il, kTRUE );
         }
      } // if (TMath::Abs(F)<xxx)
   }
}


//_______________________________________________________________________
void TMVA::RuleFitParams::UpdateCoefficients()
{
   // Establish maximum gradient for rules, linear terms and the offset
   //
   Double_t maxr = ( (fRuleEnsemble->DoRules() ? 
                      TMath::Abs(*(std::max_element( fGradVec.begin(), fGradVec.end(), TMVA::AbsValue()))):0) );
   Double_t maxl = ( (fRuleEnsemble->DoLinear() ? 
                      TMath::Abs(*(std::max_element( fGradVecLin.begin(), fGradVecLin.end(), TMVA::AbsValue()))):0) );
   // Use the maximum as a threshold
   Double_t maxv = (maxr>maxl ? maxr:maxl);
   Double_t cthresh = maxv * fGDTau;

   Double_t useRThresh;
   Double_t useLThresh;
   //
   // Choose threshholds.
   //
   useRThresh = cthresh;
   useLThresh = cthresh;

   Double_t gval, lval, coef, lcoef;

   // Add to offset, if gradient is large enough:
   // Loop over the gradient vector and move to next set of coefficients
   // size of GradVec (and GradVecLin) should be 0 if learner is disabled
   if (maxv>0) {
      for (UInt_t i=0; i<fGradVec.size(); i++) {
         gval = fGradVec[i];
         if (TMath::Abs(gval)>=useRThresh) {
            coef = fRuleEnsemble->GetRulesConst(i)->GetCoefficient() + fGDPathStep*gval;
            fRuleEnsemble->GetRules(i)->SetCoefficient(coef);
         }
      }

      // Loop over the gradient vector for the linear part and move to next set of coefficients
      for (UInt_t i=0; i<fGradVecLin.size(); i++) {
         lval = fGradVecLin[i];
         if (TMath::Abs(lval)>=useLThresh) {
            lcoef = fRuleEnsemble->GetLinCoefficients(i) + (fGDPathStep*lval/fRuleEnsemble->GetLinNorm(i));
            fRuleEnsemble->SetLinCoefficient(i,lcoef);
         }
      }
   // Set the offset
   // Double_t ofs     = fRuleEnsemble->GetOffset();
   //   Double_t respons = CalcAverageResponse( ibeg, iend );
      Double_t offset = CalcAverageResponse();
   //   fRuleEnsemble->SetOffset( fAverageTruth + ofs - respons );
   //   fRuleEnsemble->SetOffset( ofs - respons );
      fRuleEnsemble->SetOffset( offset );
   //   respons = CalcAverageResponse( ibeg, iend );
   }
}

//_______________________________________________________________________
void TMVA::RuleFitParams::CalcTstAverageResponse()
{
   // calc average response for all test paths - TODO: see comment under CalcAverageResponse()
   // note that 0 offset is used
   for (UInt_t itau=0; itau<fGDNTau; itau++) {
      fGDOfsTst[itau] = 0;
      for (UInt_t s=0; s<fNLinear; s++) {
         fGDOfsTst[itau] -= fGDCoefLinTst[itau][s] * fAverageSelector[s];
      }
      for (UInt_t r=0; r<fNRules; r++) {
         fGDOfsTst[itau] -= fGDCoefTst[itau][r] * fAverageRule[r];
      }
   }
   //
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcAverageResponse()
{
   // calulate the average response - TODO : rewrite bad dependancy on EvaluateAverage() !
   //
   // note that 0 offset is used
   Double_t ofs = 0;
   for (UInt_t s=0; s<fNLinear; s++) {
      ofs -= fRuleEnsemble->GetLinCoefficients(s) * fAverageSelector[s];
   }
   for (UInt_t r=0; r<fNRules; r++) {
      ofs -= fRuleEnsemble->GetRules(r)->GetCoefficient() * fAverageRule[r];
   }
   return ofs;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcAverageResponseOLD(UInt_t ibeg, UInt_t iend)
{
   // calulate the average response

   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<CalcAverageResponse> invalid start/end indices!" << Endl;
      return 0;
   }
   Double_t sum=0;
   const std::vector<const Event *> *events = GetTrainingEvents();
   for (UInt_t i=ibeg; i<iend+1; i++) {
      sum += fRuleEnsemble->EvalEvent( *(*events)[i] );
   }
   return sum/neve;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcAverageTruth(UInt_t ibeg, UInt_t iend)
{
   // calulate the average truth

   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<CalcAverageTruth> invalid start/end indices!" << Endl;
      return 0;
   }
   Double_t sum=0;
   Int_t nsig=0;
   Int_t nbkg=0;
   const std::vector<const Event *> *events = GetTrainingEvents();
   for (UInt_t i=ibeg; i<iend+1; i++) {
      if ((*events)[i]->IsSignal()) nsig++;
      else                          nbkg++;
      sum += Double_t((*events)[i]->IsSignal()?1:-1);
   }
   fLogger << kVERBOSE << "num of signal / background = " << nsig << " / " << nbkg << Endl;

   return sum/neve;
}
