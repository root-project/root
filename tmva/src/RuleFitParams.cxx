// @(#)root/tmva $Id: RuleFitParams.cxx,v 1.18 2006/10/03 17:49:10 tegen Exp $
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <algorithm>
#include <numeric>

#include "TMVA/Timer.h"

#include "TMVA/RuleFitParams.h"

#include "TMVA/RuleFit.h"

#include "TMVA/RuleEnsemble.h"

// Uncomment this in order to get detailed printout on the path search
#define DEBUG_RULEFITPARAMS
#undef  DEBUG_RULEFITPARAMS

//_______________________________________________________________________
TMVA::RuleFitParams::RuleFitParams()
{
   fRuleFit = 0;
   fRuleEnsemble = 0;
   fGDTau      = 0.0;
   fGDPathStep = 0.01;
   fGDNPathSteps = 100;
   fGDNtuple = 0;
   fGDErrNsigma = 1.0;
   fNTLinCoeff = 0;
   fNTCoeff = 0;
   fNTOffset = 0;
   Init();
}
//_______________________________________________________________________
TMVA::RuleFitParams::~RuleFitParams()
{
   //   if (fGDNtuple)    delete fGDNtuple;
   if (fNTCoeff)     delete fNTCoeff;
   if (fNTLinCoeff)  delete fNTLinCoeff;
}
//_______________________________________________________________________
Double_t TMVA::RuleFitParams::LinearModel( const TMVA::Event& e ) const
{
   return fRuleEnsemble->EvalEvent( e );
}

//_______________________________________________________________________
void TMVA::RuleFitParams::Init()
{
   //
   if (fRuleFit==0) return;
   fRuleEnsemble = fRuleFit->GetRuleEnsemblePtr();
   UInt_t nrules = fRuleEnsemble->GetNRules();
   UInt_t nvars  = fRuleEnsemble->GetLinNorm().size(); // HERE: Define a GetNVars() or similar...
   //
   fGradVec.clear();
   fGradVecLin.clear();
   fGradVecMin.clear();
   //
   if (fRuleEnsemble->DoRules()) {
      std::cout << "--- RuleFitParams: Number of rules in ensemble = " << nrules << std::endl;
      fGradVec.resize(nrules,0);
      fGradVecMin.resize(nrules,0); // HERE: Not yet used
   } else {
      std::cout << "--- RuleFitParams: Rules are disabled " << std::endl;
   }
   if (fRuleEnsemble->DoLinear()) {
      std::cout << "--- RuleFitParams: Number of linear terms = " << nvars << std::endl;
      fGradVecLin.resize(nvars,0);
   } else {
      std::cout << "--- RuleFitParams: Linear terms are disabled " << std::endl;
   }

}

//_______________________________________________________________________
void TMVA::RuleFitParams::InitNtuple()
{
   const UInt_t nrules = fRuleEnsemble->GetNRules();
   const UInt_t nlin   = fRuleEnsemble->GetLinNorm().size();
   //
   fGDNtuple= new TTree("MonitorNtuple_RuleFitParams","RuleFit path search");
   fGDNtuple->Branch("risk",    &fNTRisk,     "risk/D");
   fGDNtuple->Branch("error",   &fNTErrorRate,"error/D");
   fGDNtuple->Branch("nuval",   &fNTNuval,    "nuval/D");
   fGDNtuple->Branch("coefrad", &fNTCoefRad,  "coefrad/D");
   fGDNtuple->Branch("offset",  &fNTOffset,   "offset/D");
   //
   fNTCoeff    = new Double_t[nrules];
   fNTLinCoeff = new Double_t[nlin];

   for (UInt_t i=0; i<nrules; i++) {
      fGDNtuple->Branch(Form("a%d",i+1),&fNTCoeff[i],Form("a%d/D",i+1));
   }
   for (UInt_t i=0; i<nlin; i++) {
      fGDNtuple->Branch(Form("b%d",i+1),&fNTLinCoeff[i],Form("b%d/D",i+1));
   }
}

//_______________________________________________________________________
const std::vector<const TMVA::Event *>  *TMVA::RuleFitParams::GetTrainingEvents()  const
{
   return &(fRuleFit->GetTrainingEvents());
}

//_______________________________________________________________________
const std::vector< Int_t >  *TMVA::RuleFitParams::GetSubsampleEvents() const
{
   return &(fRuleFit->GetSubsampleEvents());
}

//_______________________________________________________________________
void TMVA::RuleFitParams::GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const
{
   fRuleFit->GetSubsampleEvents(sub,ibeg,iend);
}

//_______________________________________________________________________
const UInt_t TMVA::RuleFitParams::GetNSubsamples() const
{
   return fRuleFit->GetNSubsamples();
}

//_______________________________________________________________________
const TMVA::Event *TMVA::RuleFitParams::GetTrainingEvent(UInt_t i) const
{
   return fRuleFit->GetTrainingEvent(i);
}

//_______________________________________________________________________
const TMVA::Event *TMVA::RuleFitParams::GetTrainingEvent(UInt_t i, UInt_t isub) const
{
   return fRuleFit->GetTrainingEvent(i,isub);
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::Risk() const
{
   UInt_t iend,ibeg;
   GetSubsampleEvents(0,ibeg,iend);
   UInt_t neve = iend-ibeg;
   if (neve<1) {
      std::cout << "--- RuleFitParams::MakeGradientVector() - invalid start/end indices!" << std::endl;
      return 0;
   }
   Double_t rval=0;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   for ( UInt_t i=ibeg; i<iend; i++) {
      rval += LossFunction( *(*events)[i] );
   }
   rval = rval/Double_t(neve);
   //   if (fRegularization>0) rval += fRegularization*Penalty();
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::LossFunction( const TMVA::Event& e ) const
{
   // Implementation of squared-error ramp loss function (eq 39,40 in ref 1)
   // This is used for binary Classifications where y = {+1,-1} for (sig,bkg)
   Double_t F = LinearModel( e );
   Double_t H = max( -1.0, min(1.0,F) );
   Double_t diff = (e.IsSignal()?1:-1) - H;
   //
   return diff*diff;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::Penalty() const
{
   // This is the "lasso" penalty
   Double_t rval=0;
   UInt_t nrules = fRuleEnsemble->GetNRules();
   const std::vector<Double_t> *lincoeff = & (fRuleEnsemble->GetLinCoefficients());
   if (fRuleEnsemble->DoRules()) {
      for (UInt_t i=0; i<nrules; i++) {
         rval += fabs(fRuleEnsemble->GetRules(i)->GetCoefficient());
      }
   }
   if (fRuleEnsemble->DoLinear()) {
      for (UInt_t i=0; i<lincoeff->size(); i++) {
         rval += fabs((*lincoeff)[i]);
      }
   }
   return rval;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeGDPath()
{
   // The following finds the gradient directed path in parameter space.
   // More work is needed... FT, 24/9/2006
   //
   //   std::cout << "--- RuleFitParams: n(events) = " << nevents << std::endl;
   // initial estimate; all other a(i) are zero
   std::cout << "--- RuleFitParams: Creating GD path" << std::endl;
   std::cout << "--- RuleFitParams: N(steps)   = " << fGDNPathSteps << std::endl;
   std::cout << "--- RuleFitParams: Step       = " << fGDPathStep   << std::endl;
   std::cout << "--- RuleFitParams: Tau        = " << fGDTau        << std::endl;
   InitNtuple();
   Double_t a0 = CalcAverageResponse();
   fRuleEnsemble->SetOffset(a0);
   fRuleEnsemble->ClearCoefficients(0);
   fRuleEnsemble->ClearLinCoefficients(0);

   std::cout << "--- RuleFitParams: Obtained a0 = " << a0 << std::endl;
   // Loop over the paths
   int imod = fGDNPathSteps/100;
   if (imod<1) imod = 1;
   if (imod>1000) imod=1000;
   //
   Double_t rprev=1e32;
   UInt_t iend,ibeg;
   GetSubsampleEvents(0,ibeg,iend);
   //
   std::cout << "--- RuleFitParams: Number of events used per path step = " << iend-ibeg+1 << std::endl;
   Double_t errmin = 1e32;
   Double_t riskMin=0;
   Int_t indMin=-1;
   std::vector<Double_t> coefsMin;
   std::vector<Double_t> lincoefsMin;

   fPerfUsedSet = -1;
   fFstarValid = kFALSE;
   Bool_t done = kFALSE;

   std::cout << "--- RuleFitParams: GD path scan - the scan stops when the max number of steps is reached or a minimum is found"
             << std::endl;

#ifndef DEBUG_RULEFITPARAMS
   TMVA::Timer timer( fGDNPathSteps, "RuleFitParams" );
#endif
   //   for (int i=0; i<fGDNPathSteps; i++) {
   int i=0;
   clock_t t0;
   clock_t tloop;
   clock_t tgradvec;
   clock_t tupgrade;
   clock_t trisk;
   clock_t tperf;
   clock_t stgradvec=0;
   clock_t stupgrade=0;
   clock_t strisk=0;
   clock_t stperf=0;
   
   Bool_t found=kFALSE;
   Bool_t riskFlat=kFALSE;
   Int_t nsubsamples = GetNSubsamples();
   Int_t nset  = (nsubsamples>1 ? 1:0);
   Double_t df=0;
   Double_t erd;
   // linear regression
   const UInt_t npreg=10;
   std::vector<Double_t> valx;
   std::vector<Double_t> valy;
   std::vector<Double_t> valxy;
   //
   fNTRisk = rprev;
   fNTCoefRad = -1.0;
   fNTErrorRate = 0;

   //
   while (!done) {
      // Make gradient vector (eq 44, ref 1)
      t0 = clock();
      MakeGradientVector();
      tgradvec = clock()-t0;
      stgradvec += tgradvec;
      
      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      t0 = clock();
      UpdateCoefficients();
      tupgrade = clock()-t0;
      stupgrade += tupgrade;

      // Calculate error
      if ((i==0) || (i%imod==0)) {
#ifndef DEBUG_RULEFITPARAMS
         timer.DrawProgressBar(i);
#endif

#ifdef DEBUG_RULEFITPARAMS
         fNTCoefRad = fRuleEnsemble->CoefficientRadius();
#endif

         fNTNuval = Double_t(i)*fGDPathStep;
         FillCoefficients();
         // Calculate risk
         t0 = clock();
         fNTRisk = Risk();
         trisk = clock()-t0;
         strisk += trisk;
         //
         // Check for an increase in risk.
         // Such an increase would imply that the regularization is too small.
         // Stop the iteration if this happens.
         //
         if (fNTRisk>=rprev) {
#ifdef DEBUG_RULEFITPARAMS
            if (fNTRisk>rprev) std::cout << "--- RuleFitParams: WARNING: R(i+1)>=R(i) in path - something is wrong." << std::endl;
#endif
            riskFlat=kTRUE;
         }
         rprev = fNTRisk;
         //
         fNTErrorRate = ErrorRateBin(nset,df);
         if (fNTErrorRate<errmin) {
            errmin=fNTErrorRate; riskMin = fNTRisk; indMin = i;
            fRuleEnsemble->GetCoefficients(coefsMin);
            lincoefsMin = fRuleEnsemble->GetLinCoefficients();
         }
         erd = (fNTErrorRate-errmin)/df;
         if ( erd > fGDErrNsigma) found = kTRUE;  //TODO: 1.1 as an option
         // check slope of last couple of points
         if (valx.size()==npreg) {
            valx.erase(valx.begin());
            valy.erase(valy.begin());
            valxy.erase(valxy.begin());
         }
         valx.push_back(fNTNuval);
         valy.push_back(erd);
         valxy.push_back(fNTErrorRate*fNTNuval);

         //
         fGDNtuple->Fill();
         //
#ifdef DEBUG_RULEFITPARAMS
         std::cout << "--- RuleFitParamsIRE : " << i << " " << fNTRisk << " " << fNTErrorRate << " "
                   << fRuleEnsemble->CoefficientRadius() << " "
                   << std::accumulate( valy.begin(), valy.end(), Double_t() )/valy.size();
         if (fRuleEnsemble->GetLinCoefficients().size()>3) {
            std::cout << " "
                      << fRuleEnsemble->GetLinCoefficients(0) << " "
                      << fRuleEnsemble->GetLinCoefficients(1) << " "
                      << fRuleEnsemble->GetLinCoefficients(2) << " "
                      << fRuleEnsemble->GetLinCoefficients(3) << " ";
         }
         std::cout << std::endl;
#endif
         tperf = clock()-t0;
         stperf +=tperf;
      }
      i++;
      // Stop iteration under various conditions
      // * The condition R(i+1)<R(i) is no longer true (when then implicit regularization is too weak)
      // * If the current error estimate is > factor*errmin (factor = 1.1)
      // * We have reach the last step...
      if ( ((riskFlat) || (i==fGDNPathSteps)) && (!found) ) {
         if (indMin<0) {
            errmin=fNTErrorRate; riskMin = fNTRisk; indMin = i;
            fRuleEnsemble->GetCoefficients(coefsMin);
            lincoefsMin = fRuleEnsemble->GetLinCoefficients();
         }
         found = kTRUE;
      }
      done = (found); //((i==fGDNPathSteps) || (coefRad>1.0));//)(err>1.1*errmin)); // CHANGE THIS
   }
   std::cout << "\n--- RuleFitParams: min error rate = " << errmin << " at step " << indMin << std::endl;
   if ( Double_t(indMin)/Double_t(fGDNPathSteps) < 0.05 ) {
      std::cout << "--- RuleFitParams: WARNING - reached minimum early in the search." << std::endl;
      std::cout << "                   Decrease the step size (GDStep)." << std::endl;
   }
   //
   // quick check of the sign of the slope for the last npreg points
   //
   Double_t sumx  = std::accumulate( valx.begin(), valx.end(), Double_t() );
   Double_t sumxy = std::accumulate( valxy.begin(), valxy.end(), Double_t() );
   Double_t sumy  = std::accumulate( valy.begin(), valy.end(), Double_t() );
   Double_t slope = Double_t(valx.size())*sumxy - sumx*sumy;
   if (slope<0) {
      std::cout << "--- RuleFitParams: WARNING - the error rate was still decreasing when the end of the path was reached." << std::endl;
      std::cout << "                   Increase the number of steps (GDNSteps). Slope = " << slope << std::endl;
   }
   //
   if (found) {//indMin>=0) {
      fRuleEnsemble->SetCoefficients(coefsMin);
      //      lincoefsMin[0] = -0.787; // TEMP - setting to exact Fisher solution
      //      lincoefsMin[1] = -0.876;
      //      lincoefsMin[2] = -3.041;
      //      lincoefsMin[3] =  6.692;
      fRuleEnsemble->SetLinCoefficients(lincoefsMin);
   }
   tloop  = strisk +stupgrade + stgradvec + stperf;
   std::cout << "--- RuleFitParams: " 
             << "Time gradvec = " << stgradvec/i
             << "     upgrade = " << stupgrade/i
             << "     risk    = " << strisk/i
             << "     perf    = " << stperf/i << std::endl;
   fGDNtuple->Write();
}
//_______________________________________________________________________
void TMVA::RuleFitParams::FillCoefficients()
{
   const UInt_t nrules = fRuleEnsemble->GetNRules();
   const UInt_t nlin   = fRuleEnsemble->GetLinNorm().size();
   //
   fNTOffset = fRuleEnsemble->GetOffset();
   //
   for (UInt_t i=0; i<nrules; i++) {
      fNTCoeff[i] = fRuleEnsemble->GetRules(i)->GetCoefficient();
   }
   for (UInt_t i=0; i<nlin; i++) {
      fNTLinCoeff[i] = fRuleEnsemble->GetLinCoefficients(i);
   }
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcOffset(Int_t set)
{
   //
   // Calculate an offset = - sum (Fs/Ns + Fb/Nb) / 2
   //
   UInt_t iend,ibeg;

   if (set!=fPerfUsedSet) fFstarValid=kFALSE;
   //
   GetSubsampleEvents(set,ibeg,iend);
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      std::cout << "--- RuleFitParams::Performance() - invalid start/end indices!" << std::endl;
      fFstarValid = kFALSE;
      return 1000000.0;
   }
   //
   Double_t F;//,iF;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   Double_t sumS = 0.0;
   Double_t sumB = 0.0;
   Int_t    nB = 0;
   Int_t    nS = 0;
   for (UInt_t i=ibeg; i<iend; i++) {
      const TMVA::Event& e = *(*events)[i];
      F = LinearModel( e );
      if (e.IsSignal()) {
         sumS += F;
         nS++;
      } else {
         sumB += F;
         nB++;
      }
   }
   return -0.5*( (sumS/Double_t(nS)) + (sumB/Double_t(nB)) );
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateReg(Int_t set)
{
   //
   // Estimates the error rate with the current set of parameters
   // This code is pretty messy at the moment.
   // Cleanup is needed.
   // The whole idea with Fstar is wrong.
   // Remove it when sure... FT 29/9/2006
   //
   UInt_t iend,ibeg;

   if (set!=fPerfUsedSet) fFstarValid=kFALSE;
   //
   GetSubsampleEvents(set,ibeg,iend);
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      std::cout << "--- RuleFitParams::Performance() - invalid start/end indices!" << std::endl;
      fFstarValid = kFALSE;
      return 1000000.0;
   }
   //
   Double_t F;//,iF;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   if (!fFstarValid) {
      fFstar.clear();
      std::vector<Double_t> fstarSorted;
      Double_t fstarVal;
      //
      for (UInt_t i=ibeg; i<iend; i++) {
         const TMVA::Event& e = *(*events)[i];
         fstarVal = fRuleEnsemble->FStar(e);
         //         std::cout << "F* = " << fstarVal << std::endl;
         fFstar.push_back(fstarVal);
         fstarSorted.push_back(fstarVal);
         if (isnan(fstarVal)) std::cout << "WARNING: F* is NAN!" << std::endl;
      }
      std::sort( fstarSorted.begin(), fstarSorted.end() );
      UInt_t ind = neve/2;
      //   for (UInt_t i=0; i<neve; i++) {
      //      if ( ((i>ind-10) && (i<ind+10)) || (i<5) || (i>neve-5) )
      //         std::cout << "F*sorted : " << i << " -> " << fstarSorted[i] << std::endl;
      //   }
      if (neve&1) { // odd number of events
         fFstarMedian = 0.5*(fstarSorted[ind]+fstarSorted[ind-1]);
      } else { // even
         fFstarMedian = fstarSorted[ind];
      }
      
      fFstarValid = kTRUE;
   }
   //
   //   std::cout << "fFstarMedian = " << fFstarMedian << std::endl;
   Double_t sumdf = 0;
   Double_t sumdfmed = 0;
   //
   // A bit messy here.
   // I believe the binary error classification is appropriate here.
   // The problem is stability.
   //
   for (UInt_t i=ibeg; i<iend; i++) {
      const TMVA::Event& e = *(*events)[i];
      F = LinearModel( e );
      // scaled abs error, eq 20 in RuleFit paper
      sumdf += fabs(fFstar[i-ibeg] - F);
      sumdfmed += fabs(fFstar[i-ibeg] - fFstarMedian);
   }
   //   std::cout << "median F* = " << fFstarMedian << " ; sumdfmed = " << sumdfmed << " ; sumdf = " << sumdf << std::endl;
   // scaled abs error, eq 20
   // This error (df) is large - need to think on how to compensate...
   //
   fPerfUsedSet = set;
   return sumdf/sumdfmed;
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::ErrorRateBin(Int_t set, Double_t & df)
{
   //
   // Estimates the error rate with the current set of parameters
   // It uses a binary estimate of (y-F*(x))
   // (y-F*(x)) = (Num of events where sign(F)!=sign(y))/Neve
   // y = {+1 if event is signal, -1 otherwise}
   //
   UInt_t iend,ibeg;
   //
   GetSubsampleEvents(set,ibeg,iend);
   UInt_t neve = iend-ibeg;
   if (neve<1) {
      std::cout << "--- RuleFitParams::Performance() - invalid start/end indices!" << std::endl;
      fFstarValid = kFALSE;
      return 1000000.0;
   }
   //
   const std::vector<const Event *> *events = GetTrainingEvents();
   //
   Double_t sumdfbin = 0;
   Double_t dneve = Double_t(neve);
   Int_t signF, signy;
   //   Double_t FStar;
   Double_t F;
   //
   for (UInt_t i=ibeg; i<iend; i++) {
      const TMVA::Event& e = *(*events)[i];
      F     = LinearModel( e );
      //      FStar = fRuleEnsemble->FStar(e);
      signF = (F>0 ? +1:-1);
      //      signy = (FStar>0 ? +1:-1);
      signy = (e.IsSignal() ? +1:-1);
      sumdfbin += fabs(Double_t(signF-signy))*0.5;
   }
   Double_t f = sumdfbin/dneve;
   df = f*sqrt((1.0/sumdfbin) + (1.0/dneve));
   //   std::cout << "f = " << f << " +- " << df << " N(err) = " << sumdfbin << ",  N = " << neve<< std::endl;
   return f;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeGradientVector()
{
   UInt_t iend,ibeg;
   GetSubsampleEvents(0,ibeg,iend);
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      std::cout << "--- RuleFitParams::MakeGradientVector() - invalid start/end indices!" << std::endl;
      return;
   }
   //
   Double_t F;//,iF;
   Double_t r; // eq 35, ref 1
   UInt_t nrules = (fRuleEnsemble->DoRules() ? fGradVec.size():0);
   UInt_t nlin   = (fRuleEnsemble->DoLinear() ? fGradVecLin.size():0);
   //
   const TMVA::Rule *rule;
   Double_t norm = 2.0/Double_t(neve);
   //   Double_t sig;
   //
   const std::vector<const Event *> *events = GetTrainingEvents();

   // Clear gradient vectors
   for (UInt_t ir=0; ir<nrules; ir++) {
      fGradVec[ir]=0;
   }
   for (UInt_t il=0; il<nlin; il++) {
      fGradVecLin[il]=0;
   }
   fGradOfs = 0;
   // Loop over all events and calculate gradient
   for (UInt_t i=ibeg; i<iend; i++) {
      const TMVA::Event& e = *(*events)[i];
      F = LinearModel( e );
      //      iF = ((fabs(F)<1) ? 1.0:0.0);
      //      if ((i==ibeg) || (fabs(F)<100000.0)) {
      r=0;
      if (fabs(F)<1.0) {
         r = (e.IsSignal()?1.:-1.) - F;
         fGradOfs = norm*r;
         // Loop over all rules and calculate grad vector
         for (UInt_t ir=0; ir<nrules; ir++) {
            rule = fRuleEnsemble->GetRulesConst(ir);
            //               sig = fRuleEnsemble->GetSupport(ir);
            //               sig = (sig>0 ? 1.0/sig : 1.0);
            fGradVec[ir] += norm*r*rule->EvalEvent(e); // eval event without coeff
         }
         for (UInt_t il=0; il<nlin; il++) {
            fGradVecLin[il] += norm*r*fRuleEnsemble->EvalLinEvent( il, e, kTRUE );
         }
      } // if (fabs(F)<xxx)
   }
//    Double_t sum2=0;
//    for (UInt_t r=0; r<nrules; r++) {
//       sum2 += fGradVec[r]*fGradVec[r];
//    }
   //   std::cout << "|fGradVec| = " << sqrt(sum2) << std::endl;
}

//_______________________________________________________________________
void TMVA::RuleFitParams::UpdateCoefficients()
{
   // Establish maximum gradient for rules, linear terms and the offset
   std::vector<Double_t> maxgrads;
   maxgrads.push_back( (fRuleEnsemble->DoRules() ? 
                        fabs(*(std::max_element( fGradVec.begin(), fGradVec.end(), TMVA::AbsValue()))):0) );
   maxgrads.push_back( (fRuleEnsemble->DoLinear() ? 
                        fabs(*(std::max_element( fGradVecLin.begin(), fGradVecLin.end(), TMVA::AbsValue()))):0) );
   //   maxgrads.push_back(  fabs(fGradOfs) );
   // Use the maximum as a threshold
   Double_t cthresh = (*std::max_element( maxgrads.begin(),maxgrads.end() )) * fGDTau;
   //   if (maxl>maxr) std::cout << "Lincoeff is max: " << cthresh << std::endl;
   //   std::cout << (maxr>maxl ? " Rulecoeff is max : " : " Linear coeff is max : ") << cthresh << std::endl;
   //   std::cout << "  Other = " << (maxr>maxl ? maxl:maxr )*fGDTau/cthresh << std::endl;
   //
   // Individual thresholds for linear and rules - not what we want...
   //   Double_t rthresh = maxr*fGDTau;
   //   Double_t lthresh = maxl*fGDTau;
   const Double_t useRThresh = cthresh;
   const Double_t useLThresh = cthresh;
   //   const Double_t useOThresh = cthresh;

   Double_t gval, lval, coef, lcoef;

   // Add to offset, if gradient is large enough:
   //   if (fabs(fGradOfs)>useOThresh) fRuleEnsemble->AddOffset(fGradOfs*fGDPathStep); REMOVE PERHAPS!
   // Loop over the gradient vector and move to next set of coefficients
   // size of GradVec (and GradVecLin) should be 0 if learner is disabled
   for (UInt_t i=0; i<fGradVec.size(); i++) {
      gval = fGradVec[i];
      //      std::cout << "RC = " << gval << " ; thresh = " << useThresh << std::endl;
      if (fabs(gval)>=useRThresh) {
         //      if (fabs(gval)>=gthresh) {
         coef = fRuleEnsemble->GetRulesConst(i)->GetCoefficient() + fGDPathStep*gval;
         fRuleEnsemble->GetRules(i)->SetCoefficient(coef);
      }
   }

   // Loop over the gradient vector for the linear part and move to next set of coefficients
   for (UInt_t i=0; i<fGradVecLin.size(); i++) {
      lval = fGradVecLin[i];
      //      std::cout << "LC = " << lval << " ; thresh = " << useThresh << std::endl;
      if (fabs(lval)>=useLThresh) {
         //         if (fabs(lval)>=lthresh) {
         lcoef = fRuleEnsemble->GetLinCoefficients(i) + fGDPathStep*lval;
         fRuleEnsemble->SetLinCoefficient(i,lcoef);
      }
   }

   // Calculate offset
   fRuleEnsemble->SetOffset( CalcOffset(0) );
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcAverageResponse()
{
   UInt_t iend,ibeg;
   GetSubsampleEvents(0,ibeg,iend);
   UInt_t neve = iend-ibeg;
   if (neve<1) {
      std::cout << "--- RuleFitParams::CalcAverageResponse() - invalid start/end indices!" << std::endl;
      return 0;
   }
   Double_t sum=0;
   Int_t nsig=0;
   Int_t nbkg=0;
   const std::vector<const Event *> *events = GetTrainingEvents();
   for (UInt_t i=ibeg; i<iend; i++) {
      if ((*events)[i]->IsSignal()) {
         nsig++;
      } else {
         nbkg++;
      }
      sum += Double_t((*events)[i]->IsSignal()?1:-1);
   }
   std::cout << "--- RuleFitParams: Num of sig = " << nsig << std::endl;
   std::cout << "--- RuleFitParams: Num of bkg = " << nbkg << std::endl;
   return sum/neve;
}
