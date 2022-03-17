// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::RatioOfProfiledLikelihoodsTestStat
    \ingroup Roostats

TestStatistic that returns the ratio of profiled likelihoods.

By default the calculation is:

\f[
        \log{    \frac{ \lambda(\mu_{alt}  , {conditional \: MLE \: for \: alt  \: nuisance}) }
                      { \lambda(\mu_{null} , {conditional \: MLE \: for \: null \: nuisance}) } }
\f]

where \f$ \lambda \f$ is the profile likelihood ratio, so the
MLE for the null and alternate are subtracted off.

If `SetSubtractMLE(false)` then it calculates:

\f[
        \log{    \frac{ L(\mu_{alt}  , {conditional \: MLE \: for \: alt  \: nuisance}) }
                      { L(\mu_{null} , {conditional \: MLE \: for \: null \: nuisance}) } }
\f]

where \f$ L \f$ is the Likelihood function.

The values of the parameters of interest for the alternative
hypothesis are taken at the time of the construction.
If empty, it treats all free parameters as nuisance parameters.

The value of the parameters of interest for the null hypotheses
are given at each call of Evaluate.

This test statistic is often called the Tevatron test statistic, because it has
been used by the Tevatron experiments.
*/

#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"

#include "RooStats/ProfileLikelihoodTestStat.h"

#include "RooArgSet.h"
#include "RooAbsData.h"
#include "TMath.h"
#include "RooMsgService.h"
#include "RooGlobalFunc.h"


Bool_t RooStats::RatioOfProfiledLikelihoodsTestStat::fgAlwaysReuseNll = kTRUE ;

void RooStats::RatioOfProfiledLikelihoodsTestStat::SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }

////////////////////////////////////////////////////////////////////////////////
/// returns -logL(poi, conditional MLE of nuisance params)
/// subtract off the global MLE or not depending on the option
/// It is the numerator or the denominator of the ratio (depending on the pdf)
///
/// L.M. : not sure why this method is needed now

Double_t RooStats::RatioOfProfiledLikelihoodsTestStat::ProfiledLikelihood(RooAbsData& data, RooArgSet& poi, RooAbsPdf& pdf) {
   int type = (fSubtractMLE) ? 0 : 2;

   // null
   if ( &pdf == fNullProfile.GetPdf() )
      return fNullProfile.EvaluateProfileLikelihood(type, data, poi);
   else if (&pdf == fAltProfile.GetPdf() )
      return fAltProfile.EvaluateProfileLikelihood(type, data, poi);

   oocoutE((TObject*)NULL,InputArguments) << "RatioOfProfiledLikelihoods::ProfileLikelihood - invalid pdf used for computing the profiled likelihood - return NaN"
                         << std::endl;

   return TMath::QuietNaN();

}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the ratio of profile likelihood

Double_t  RooStats::RatioOfProfiledLikelihoodsTestStat::Evaluate(RooAbsData& data, RooArgSet& nullParamsOfInterest) {

   int type = (fSubtractMLE) ? 0 : 2;

   // null
   double nullNLL = fNullProfile.EvaluateProfileLikelihood(type, data, nullParamsOfInterest);
   const RooArgSet *nullset = fNullProfile.GetDetailedOutput();

   // alt
   double altNLL = fAltProfile.EvaluateProfileLikelihood(type, data, *fAltPOI);
   const RooArgSet *altset = fAltProfile.GetDetailedOutput();

   if (fDetailedOutput != NULL) {
      delete fDetailedOutput;
      fDetailedOutput = NULL;
   }
   if (fDetailedOutputEnabled) {
      fDetailedOutput = new RooArgSet();
      for (auto const *var : static_range_cast<RooRealVar *>(*nullset)) {
         RooRealVar* cloneVar = new RooRealVar(TString::Format("nullprof_%s", var->GetName()),
                                               TString::Format("%s for null", var->GetTitle()), var->getVal());
         fDetailedOutput->addOwned(*cloneVar);
      }
      for (auto const *var : static_range_cast<RooRealVar *>(*altset)) {
         RooRealVar* cloneVar = new RooRealVar(TString::Format("altprof_%s", var->GetName()),
                                               TString::Format("%s for null", var->GetTitle()), var->getVal());
         fDetailedOutput->addOwned(*cloneVar);
      }
   }

/*
// set variables back to where they were
nullParamsOfInterest = *saveNullPOI;
*allVars = *saveAll;
delete saveAll;
delete allVars;
*/

   return nullNLL -altNLL;
}
