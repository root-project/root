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
#ifndef ROOSTATS_ModelConfig
#include "RooStats/ModelConfig.h"
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
MCMCCalculator::MCMCCalculator() : 
   fPOI(0),
   fPropFunc(0), 
   fPdf(0), 
   fData(0),
   fAxes(0)
{
   // default constructor
   fNumIters = 0;
   fNumBurnInSteps = 0;
   fNumBins = 0;
   fUseKeys = kFALSE;
   fUseSparseHist = kFALSE;
}

// Constructor for automatic configuration with basic settings.  Uses a
// UniformProposal,10,000 iterations, 40 burn in steps, 50 bins for each
// RooRealVar, determines interval by keys, and turns on sparse histogram
// mode in the MCMCInterval.  Finds a 95% confidence interval.
MCMCCalculator::MCMCCalculator(RooAbsData& data, RooAbsPdf& pdf,
                               const RooArgSet& paramsOfInterest) : 
   fPOI(&paramsOfInterest),
   fPropFunc(0), 
   fPdf(&pdf), 
   fData(&data),
   fAxes(0)
{
   SetupBasicUsage();
}

// Constructor for automatic configuration with basic settings.  Uses a
// UniformProposal,10,000 iterations, 40 burn in steps, 50 bins for each
// RooRealVar, determines interval by keys, and turns on sparse histogram
// mode in the MCMCInterval.  Finds a 95% confidence interval.
MCMCCalculator::MCMCCalculator(RooAbsData& data, const ModelConfig & model) :
   fPropFunc(0), 
   fPdf(model.GetPdf()), 
   fData(&data),
   fAxes(0)
{
   SetModel(model);
   SetupBasicUsage();
}

// alternate constructor, specifying many arguments
MCMCCalculator::MCMCCalculator(RooAbsData& data, const ModelConfig & model,
                               ProposalFunction& proposalFunction, Int_t numIters,
                               RooArgList* axes, Double_t size) : 
   fPropFunc(&proposalFunction), 
   fPdf(model.GetPdf()), 
   fData(&data), 
   fAxes(axes)
{
   SetModel(model);
   SetTestSize(size);
   fNumIters = numIters;
   fNumBurnInSteps = 0;
   fNumBins = 0;
   fUseKeys = kFALSE;
   fUseSparseHist = kFALSE;
}

void MCMCCalculator::SetModel(const ModelConfig & model) { 
   // set the model
   fPdf = model.GetPdf();  
   if (model.GetParametersOfInterest() ) fPOI = model.GetParametersOfInterest();
   if (model.GetNuisanceParameters() )   fNuisParams = model.GetNuisanceParameters();
}

// alternate constructor, specifying many arguments
MCMCCalculator::MCMCCalculator(RooAbsData& data, RooAbsPdf& pdf,
                               const RooArgSet& paramsOfInterest, ProposalFunction& proposalFunction,
                               Int_t numIters, RooArgList* axes, Double_t size) : 
   fPOI(&paramsOfInterest),
   fPropFunc(&proposalFunction), 
   fPdf(&pdf), 
   fData(&data), 
   fAxes(axes)
{
   SetTestSize(size);
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
   fPropFunc = 0;
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

   if (!fData || !fPdf || !fPOI  ) return 0;

   // if a proposal function has not been specified create a default one
   bool useDefaultPropFunc = (fPropFunc == 0); 
   if (useDefaultPropFunc) fPropFunc = new UniformProposal(); 

   RooArgSet* constrainedParams = fPdf->getParameters(*fData);
   RooAbsReal* nll = fPdf->createNLL(*fData, Constrain(*constrainedParams));
   delete constrainedParams;

   RooArgSet* params = nll->getParameters(*fData);
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

   TString name = TString("MCMCInterval_") + TString(GetName() ); 
   MCMCInterval* interval = new MCMCInterval(name, name, *fPOI, *chain);
   if (fAxes != NULL)
      interval->SetAxes(*fAxes);
   if (fNumBurnInSteps > 0)
      interval->SetNumBurnInSteps(fNumBurnInSteps);
   interval->SetUseKeys(fUseKeys);
   interval->SetUseSparseHist(fUseSparseHist);
   interval->SetConfidenceLevel(1.0 - fSize);

   if (useDefaultPropFunc) delete fPropFunc; 
   
   return interval;
}
