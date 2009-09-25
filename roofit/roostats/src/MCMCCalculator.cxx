// @(#)root/roostats:$Id: MCMCCalculator.cxx 28978 2009-06-17 14:33:31Z kbelasco $
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
MCMCCalculator is a concrete implementation of IntervalCalculator.
It uses a MetropolisHastings object to construct a Markov Chain of data points in the
parameter space.  From this Markov Chain, this class can generate a MCMCInterval as
per user specification.
</p>

<p>
The interface allows one to pass the model, data, and parameters via a workspace and
then specify them with names.
</p>

<p>
After configuring the calculator, one only needs to ask GetInterval(), which will
return an ConfInterval (MCMCInterval in this case).
</p>
END_HTML
*/
//_________________________________________________

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROO_GLOBAL_FUNC
#include "RooGlobalFunc.h"
#endif
#ifndef ROO_ABS_REAL
#include "RooAbsReal.h"
#endif
#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_ARG_LIST
#include "RooArgList.h"
#endif
#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif
#ifndef ROOSTATS_MCMCCalculator
#include "RooStats/MCMCCalculator.h"
#endif
#ifndef ROOSTATS_MetropolisHastings
#include "RooStats/MetropolisHastings.h"
#endif
#ifndef ROOSTATS_MarkovChain
#include "RooStats/MarkovChain.h"
#endif
#ifndef RooStats_MCMCInterval
#include "RooStats/MCMCInterval.h"
#endif
#ifndef ROOT_TIterator
#include "TIterator.h"
#endif
#ifndef ROOSTATS_UniformProposal
#include "RooStats/UniformProposal.h"
#endif
#ifndef ROOSTATS_PdfProposal
#include "RooStats/PdfProposal.h"
#endif

ClassImp(RooStats::MCMCCalculator);

using namespace RooFit;
using namespace RooStats;

// default constructor
MCMCCalculator::MCMCCalculator()
{
   // default constructor
   fWS = NULL;
   fPOI = NULL;
   fNuisParams = NULL;
   fOwnsWorkspace = kFALSE;
   fPropFunc = NULL;
   fPdfName = NULL;
   fDataName = NULL;
   fNumIters = 0;
   fNumBurnInSteps = 0;
   fNumBins = 0;
   fAxes = NULL;
   fUseKeys = kFALSE;
   fUseSparseHist = kFALSE;
}

// Constructor for automatic configuration with basic settings.  Uses a
// UniformProposal,10,000 iterations, 40 burn in steps, 50 bins for each
// RooRealVar, determines interval by keys, and turns on sparse histogram
// mode in the MCMCInterval.  Finds a 95% confidence interval.
MCMCCalculator::MCMCCalculator(RooAbsData& data, RooAbsPdf& pdf,
      RooArgSet& paramsOfInterest)
{
   fWS = new RooWorkspace();
   fOwnsWorkspace = true;
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);

   SetupBasicUsage();
}

// Constructor for automatic configuration with basic settings.  Uses a
// UniformProposal,10,000 iterations, 40 burn in steps, 50 bins for each
// RooRealVar, determines interval by keys, and turns on sparse histogram
// mode in the MCMCInterval.  Finds a 95% confidence interval.
MCMCCalculator::MCMCCalculator(RooWorkspace& ws, RooAbsData& data,
      RooAbsPdf& pdf, RooArgSet& paramsOfInterest)
{
   fOwnsWorkspace = false;
   SetWorkspace(ws);
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);

   SetupBasicUsage();
}

// alternate constructor, specifying many arguments
MCMCCalculator::MCMCCalculator(RooWorkspace& ws, RooAbsData& data,
                RooAbsPdf& pdf, RooArgSet& paramsOfInterest,
                ProposalFunction& proposalFunction, Int_t numIters,
                RooArgList* axes, Double_t size)
{
   fOwnsWorkspace = false;
   SetWorkspace(ws);
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);
   SetTestSize(size);
   SetProposalFunction(proposalFunction);
   fNumIters = numIters;
   fNumBurnInSteps = 0;
   fNumBins = 0;
   fAxes = axes;
   fUseKeys = kFALSE;
   fUseSparseHist = kFALSE;
}

// alternate constructor, specifying many arguments
MCMCCalculator::MCMCCalculator(RooAbsData& data, RooAbsPdf& pdf,
                RooArgSet& paramsOfInterest, ProposalFunction& proposalFunction,
                Int_t numIters, RooArgList* axes, Double_t size)
{
   // alternate constructor
   fWS = new RooWorkspace();
   fOwnsWorkspace = true;
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);
   SetTestSize(size);
   SetProposalFunction(proposalFunction);
   fNumIters = numIters;
   fNumBurnInSteps = 0;
   fNumBins = 0;
   fAxes = axes;
   fUseKeys = kFALSE;
   fUseSparseHist = kFALSE;
}

// Constructor for automatic configuration with basic settings.  Uses a
// UniformProposal,10,000 iterations, 40 burn in steps, 50 bins for each
// RooRealVar, determines interval by keys, and turns on sparse histogram
// mode in the MCMCInterval.  Finds a 95% confidence interval.
void MCMCCalculator::SetupBasicUsage()
{
   fPropFunc = new UniformProposal();
   fNumIters = 10000;
   fNumBurnInSteps = 40;
   //fNumBins = 0;
   fNumBins = 50;
   fUseKeys = kTRUE;
   fUseSparseHist = kTRUE;
   SetTestSize(0.05);
}

MCMCInterval* MCMCCalculator::GetInterval() const
{
   // Main interface to get a RooStats::ConfInterval.  

   RooAbsPdf* pdf = fWS->pdf(fPdfName);
   RooAbsData* data = fWS->data(fDataName);
   if (!data || !pdf || !fPOI || !fPropFunc) return 0;

   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RooAbsReal* nll = pdf->createNLL(*data, Constrain(*constrainedParams));
   delete constrainedParams;

   RooArgSet* params = nll->getParameters(*data);
   RemoveConstantParameters(params);
   if (fNumBins > 0) {
      SetBins(*params, fNumBins);
      SetBins(*fPOI, fNumBins);
      if (dynamic_cast<PdfProposal*>(fPropFunc)) {
         RooArgSet* proposalVars = ((PdfProposal*)fPropFunc)->GetPdf()->
                                               getParameters((RooAbsData*)NULL);
         SetBins(*proposalVars, fNumBins);
      }
   }

   MetropolisHastings mh;
   mh.SetFunction(*nll);
   mh.SetType(MetropolisHastings::kLog);
   mh.SetSign(MetropolisHastings::kNegative);
   mh.SetParameters(*params);
   mh.SetProposalFunction(*fPropFunc);
   mh.SetNumIters(fNumIters);

   MarkovChain* chain = mh.ConstructChain();

   MCMCInterval* interval = new MCMCInterval("mcmcinterval", "MCMCInterval",
                                             *fPOI, *chain);
   if (fAxes != NULL)
      interval->SetAxes(*fAxes);
   if (fNumBurnInSteps > 0)
      interval->SetNumBurnInSteps(fNumBurnInSteps);
   interval->SetUseKeys(fUseKeys);
   interval->SetUseSparseHist(fUseSparseHist);
   interval->SetConfidenceLevel(1.0 - fSize);
   return interval;
}
