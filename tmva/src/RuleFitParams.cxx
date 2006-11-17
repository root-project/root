// @(#)root/tmva $Id: RuleFitParams.cxx,v 1.32 2006/11/16 22:51:59 helgevoss Exp $
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

#include "TMVA/Timer.h"
#include "TMVA/RuleFitParams.h"
#include "TMVA/RuleFit.h"
#include "TMVA/RuleEnsemble.h"

// Uncomment this in order to get detailed printout on the path search
#define DEBUG_RULEFITPARAMS
#undef  DEBUG_RULEFITPARAMS

//_______________________________________________________________________
TMVA::RuleFitParams::RuleFitParams()
   : fRuleFit ( 0 )
   , fRuleEnsemble ( 0 )
   , fPathIdx1 ( 0 )
   , fPathIdx2 ( 0 )
   , fPerfIdx1 ( 0 )
   , fPerfIdx2 ( 0 )
   , fGDTau      ( 0.0 )
   , fGDPathStep ( 0.01 )
   , fGDNPathSteps ( 100 )
   , fGDErrNsigma ( 1.0 )
   , fGDNtuple ( 0 )
   , fNTOffset ( 0 )
   , fNTCoeff ( 0 )
   , fNTLinCoeff ( 0 )
   , fLogger( "RuleFitParams" )
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
   fRuleEnsemble = fRuleFit->GetRuleEnsemblePtr();
   UInt_t nrules = fRuleEnsemble->GetNRules();
   UInt_t nvars  = fRuleEnsemble->GetLinNorm().size(); // HERE: Define a GetNVars() or similar...
   fTrainingEvents = fRuleFit->GetTrainingEvents();
   fPathIdx1 = 0;
   fPathIdx2 = (2*fTrainingEvents.size())/3;
   fPerfIdx1 = fPathIdx2+1;
   fPerfIdx2 = fTrainingEvents.size()-1;
   //
   fGradVec.clear();
   fGradVecLin.clear();
   fGradVecMin.clear();
   //
   if (fRuleEnsemble->DoRules()) {
      fLogger << kINFO << "number of rules in ensemble = " << nrules << Endl;
      fGradVec.resize(nrules,0);
      fGradVecMin.resize(nrules,0); // HERE: Not yet used
   } 
   else fLogger << kINFO << "rules are disabled " << Endl;

   if (fRuleEnsemble->DoLinear()) {
      fLogger << kINFO << "number of linear terms = " << nvars << Endl;
      fGradVecLin.resize(nvars,0);
   } 
   else fLogger << kINFO << "linear terms are disabled " << Endl;

}

//_______________________________________________________________________
void TMVA::RuleFitParams::InitNtuple()
{
   // initializes the ntuple

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
// const std::vector<const TMVA::Event *>  *TMVA::RuleFitParams::GetTrainingEvents()  const
// {
//    return &(fRuleFit->GetTrainingEvents());
// }

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
const UInt_t TMVA::RuleFitParams::GetNSubsamples() const
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
   UInt_t nrules = fRuleEnsemble->GetNRules();
   const std::vector<Double_t> *lincoeff = & (fRuleEnsemble->GetLinCoefficients());
   if (fRuleEnsemble->DoRules()) {
      for (UInt_t i=0; i<nrules; i++) {
         rval += TMath::Abs(fRuleEnsemble->GetRules(i)->GetCoefficient());
      }
   }
   if (fRuleEnsemble->DoLinear()) {
      for (UInt_t i=0; i<lincoeff->size(); i++) {
         rval += TMath::Abs((*lincoeff)[i]);
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
#ifdef DEBUG_RULEFITPARAMS
   Int_t nbadrisk=0;
#endif

   // initial estimate; all other a(i) are zero
   fLogger << kINFO << "creating GD path" << Endl;
   fLogger << kINFO << "  N(steps)   = " << fGDNPathSteps << Endl;
   fLogger << kINFO << "  step       = " << fGDPathStep   << Endl;
   fLogger << kINFO << "  tau        = " << fGDTau        << Endl;
   InitNtuple();
   //
   // Loop over the paths
   int imod = fGDNPathSteps/100;
   if (imod<100) imod = std::min(100,fGDNPathSteps);
   if (imod>100) imod=100;
   //
   Double_t rprev=1e32;
   Double_t errmin = 1e32;
   Double_t riskMin=0;
   Int_t indMin=-1;
   std::vector<Double_t> coefsMin;
   std::vector<Double_t> lincoefsMin;

   Bool_t done = kFALSE;

   fLogger << kINFO << "GD path scan - the scan stops when the max num. of steps is reached or a min is found"
           << Endl;

#ifndef DEBUG_RULEFITPARAMS
   TMVA::Timer timer( fGDNPathSteps, "RuleFitParams" );
#endif
   //   for (int i=0; i<fGDNPathSteps; i++) {
   int i=0;
   clock_t t0;
   clock_t tloop;
   Double_t tgradvec;
   Double_t tupgrade;
   Double_t trisk;
   Double_t tperf;
   Double_t stgradvec=0;
   Double_t stupgrade=0;
   Double_t strisk=0;
   Double_t stperf=0;
   
   Bool_t found=kFALSE;
   Bool_t riskFlat=kFALSE;
   //
   fLogger << kINFO << "number of events used per path step = " << fPathIdx2-fPathIdx1+1 << Endl;
   Double_t a0 = CalcAverageResponse(fPathIdx1, fPathIdx2);
   //
   fRuleEnsemble->SetOffset(a0);
   fRuleEnsemble->ClearCoefficients(0);
   fRuleEnsemble->ClearLinCoefficients(0);

   fLogger << kINFO << "obtained a0 = " << a0 << Endl;
   CalcFStar(fPerfIdx1, fPerfIdx2);
   //
   Double_t df=0;
   Double_t erd;
   // linear regression
   const UInt_t npreg=10;
   std::vector<Double_t> valx;
   std::vector<Double_t> valy;
   std::vector<Double_t> valxy;
   //
   Int_t ncheck=0;
   Bool_t docheck;
   //
   fNTRisk = rprev;
   fNTCoefRad = -1.0;
   fNTErrorRate = 0;
   //
   tloop = clock();
   while (!done) {
      // Make gradient vector (eq 44, ref 1)
      t0 = clock();
      MakeGradientVector(fPathIdx1, fPathIdx2);
      tgradvec = Double_t(clock()-t0)/CLOCKS_PER_SEC;
      stgradvec += tgradvec;
      
      // Calculate the direction in parameter space (eq 25, ref 1) and update coeffs (eq 22, ref 1)
      t0 = clock();
      UpdateCoefficients();
      tupgrade = Double_t(clock()-t0)/CLOCKS_PER_SEC;
      stupgrade += tupgrade;

      if ((i==0) || (i%imod==0)) {
         ncheck++;
         docheck = ((i==0) || (ncheck>4));
      } else {
         docheck = kFALSE;
      }
      // Calculate error
      if (docheck) {
         ncheck++;
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
         fNTRisk = Risk(fPathIdx1, fPathIdx2);
         trisk =  Double_t(clock()-t0)/CLOCKS_PER_SEC;
         strisk += trisk;
         //
         // Check for an increase in risk.
         // Such an increase would imply that the regularization is too small.
         // Stop the iteration if this happens.
         //
         if (fNTRisk>=rprev) {
#ifdef DEBUG_RULEFITPARAMS
            if (fNTRisk>rprev) {
               nbadrisk++;
               fLogger << kWARNING << "R(i+1)>=R(i) in path." << Endl;
               riskFlat=(nbadrisk>5);
            }
#endif
         }
         rprev = fNTRisk;
         //
         // Estimate the error rate using cross validation
         // Well, not quite full cross validation since we only
         // use ONE model.
         //
         t0 = clock();
         df = 0;
         fNTErrorRate = 0;

#ifdef DEBUG_RULEFITPARAMS
         Double_t errbin  = ErrorRateBin(fPerfIdx1, fPerfIdx2);
         Double_t errrisk = ErrorRateRisk(fPerfIdx1, fPerfIdx2);
         Double_t errreg  = ErrorRateReg(fPerfIdx1, fPerfIdx2);
#endif
         Double_t errroc  = ErrorRateRoc(fPerfIdx1, fPerfIdx2);
         //
         fNTErrorRate = errroc;
         //
         tperf = Double_t(clock()-t0)/CLOCKS_PER_SEC;
         stperf +=tperf;
         //
         // Always take the last min.
         // For each step the risk is reduced.
         //
         //         if (i==9000) fNTErrorRate = 0.0;
         if (ncheck>3) {
            if (fNTErrorRate<=errmin) {
               errmin=fNTErrorRate; riskMin = fNTRisk; indMin = i;
               fRuleEnsemble->GetCoefficients(coefsMin);
               lincoefsMin = fRuleEnsemble->GetLinCoefficients();
            }
            if ( fNTErrorRate > fGDErrNsigma*errmin) found = kTRUE;  //TODO: 1.1 as an option
         }
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
#ifdef DEBUG_RULEFITPARAMS
         fGDNtuple->Fill();
         fLogger << kINFO << "ParamsIRE : "
                 << setw(10)
                 << Form("%8d",i) << " "
                 << Form("%4.4f",fNTRisk) << " "
                 << Form("%4.4f",errroc)  << " "
                 << Form("%4.4f",errrisk) << " "
                 << Form("%4.4f",errbin)  << " "
                 << Form("%4.4f",errreg)  << " "
                 << Form("%4.4f",fsigave) << " "
                 << Form("%4.4f",fsigrms) << " "
                 << Form("%4.4f",fbkgave) << " "
                 << Form("%4.4f",fbkgrms) << " "
                 << Form("%4.4f",fRuleEnsemble->CoefficientRadius());
         if (fRuleEnsemble->GetLinCoefficients().size()>3) {
            fLogger << kINFO << " "
                    << Form("%4.4f",fRuleEnsemble->GetLinCoefficients(0)) << " "
                    << Form("%4.4f",fRuleEnsemble->GetLinCoefficients(1)) << " "
                    << Form("%4.4f",fRuleEnsemble->GetLinCoefficients(2)) << " "
                    << Form("%4.4f",fRuleEnsemble->GetLinCoefficients(3)) << " ";
         }
         fLogger << kINFO << Endl;
#endif
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
   fLogger << kINFO << "in error rate = " << errmin << " at step " << indMin << Endl;
   if ( Double_t(indMin)/Double_t(fGDNPathSteps) < 0.05 ) {
      fLogger << kWARNING << "reached minimum early in the search - decrease step size (GDStep)" << Endl;
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
   if (found) {
      fRuleEnsemble->SetCoefficients(coefsMin);
      fRuleEnsemble->SetLinCoefficients(lincoefsMin);
   }
#ifdef DEBUG_RULEFITPARAMS
   Double_t stloop  = strisk +stupgrade + stgradvec + stperf;
   fLogger << kINFO << "Params: " 
           << "Time gradvec = " << stgradvec/i
           << "     upgrade = " << stupgrade/i
           << "     risk    = " << strisk/i
           << "     perf    = " << stperf/i
           << "     loop    = " << stloop/i
           << Endl;
#endif
   fGDNtuple->Write();
}
//_______________________________________________________________________
void TMVA::RuleFitParams::FillCoefficients()
{
   // helper function to store the rule coefficients in local arrays

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
Double_t TMVA::RuleFitParams::ErrorRateRoc(UInt_t ibeg, UInt_t iend)
{
   //
   // Estimates the error rate with the current set of parameters.
   // It calculates the area under the bkg rejection vs signal efficiency curve.
   // The value returned is 1-area.
   //
   static int cntloop=0;

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
      //      if (cntloop==20) std::cout << "SCORE: " << cntloop << "   " << sF << "   " << (e.IsSignal() ? 1:-1) << std::endl;
   }
   fsigave = sumfsig/sFsig.size();
   fbkgave = sumfbkg/sFbkg.size();
   fsigrms = sqrt((sumf2sig - (sumfsig*sumfsig/sFsig.size()))/(sFsig.size()-1));
   fbkgrms = sqrt((sumf2bkg - (sumfbkg*sumfbkg/sFbkg.size()))/(sFbkg.size()-1));
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
   // calculate are under rejection/efficiency curve
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
      //      std::cout << "EFF: " << cntloop << "   " << effs << "   " << rejb << std::endl;
   }
   area += 0.5*(1+rejb)*effs; // extrapolate to the end point
   cntloop++;
   return (1.0-area);
}

//_______________________________________________________________________
void TMVA::RuleFitParams::MakeGradientVector( UInt_t ibeg, UInt_t iend )
{
   //   clock_t t0=clock();
   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<MakeGradientVector> invalid start/end indices!" << Endl;
      return;
   }
   //
   UInt_t   nrules = (fRuleEnsemble->DoRules() ? fGradVec.size():0);
   UInt_t   nlin   = (fRuleEnsemble->DoLinear() ? fGradVecLin.size():0);
   Double_t norm   = 2.0/Double_t(neve);
   //
   const std::vector<const Event *> *events = GetTrainingEvents();

   // Clear gradient vectors
   for (UInt_t ir=0; ir<nrules; ir++) {
      fGradVec[ir]=0;
   }
   for (UInt_t il=0; il<nlin; il++) {
      fGradVecLin[il]=0;
   }
   fGradOfs=0;
   //
   Double_t val; // temp store
   Double_t sF;   // score function value
   Double_t a0;  // offset
   Double_t r;   // eq 35, ref 1
   Double_t y;   // true score (+1 or -1)
   Double_t sumr = 0.0; // sum of all r+a0
   //
   a0 = fRuleEnsemble->GetOffset();
   for (UInt_t i=ibeg; i<iend+1; i++) {
      const TMVA::Event& e = *(*events)[i];
      sF = fRuleEnsemble->EvalEvent( e );
      if (TMath::Abs(sF)<1.0) {
         r = 0;
         y = (e.IsSignal()?1.0:-1.0);
         r = y - sF;
         // Here we want to calculate a0.
         // See top of page 8 in the Gradient directed regularization ppr by Friedman et al.
         //  a0 = sum ( y - f(x) )/Neve
         //  F = a0 + f(x)
         sumr += r+a0;
         //
         // rule gradient vector
         for (UInt_t ir=0; ir<nrules; ir++) {
            val = fRuleEnsemble->GetEventRuleVal(ir); // filled by EvalEvent() call above
            if (val>0) fGradVec[ir] += norm*r*val;
         }
         // linear terms
         for (UInt_t il=0; il<nlin; il++) {
            fGradVecLin[il] += norm*r*fRuleEnsemble->EvalLinEvent( il, kTRUE );
         }
      } // if (TMath::Abs(F)<xxx)
   }
   fGradOfs = sumr/neve;
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
   Double_t cthresh = (maxr>maxl ? maxr:maxl) * fGDTau;

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
         lcoef = fRuleEnsemble->GetLinCoefficients(i) + fGDPathStep*lval;
         fRuleEnsemble->SetLinCoefficient(i,lcoef);
      }
   }

   // Set the offset - it is calculated in MakeGradientVector()
   fRuleEnsemble->SetOffset( fGradOfs );
}

//_______________________________________________________________________
Double_t TMVA::RuleFitParams::CalcAverageResponse(UInt_t ibeg, UInt_t iend)
{
   // calulate the average response

   UInt_t neve = iend-ibeg+1;
   if (neve<1) {
      fLogger << kFATAL << "<CalcAverageResponse> invalid start/end indices!" << Endl;
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
   fLogger << kINFO << "num of signal / background = " << nsig << " / " << nbkg << Endl;

   return sum/neve;
}
