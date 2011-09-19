// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_MetropolisHastings
#define ROOSTATS_MetropolisHastings

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROOSTATS_ProposalFunction
#include "RooStats/ProposalFunction.h"
#endif
#ifndef ROOSTATS_MarkovChain
#include "RooStats/MarkovChain.h"
#endif

namespace RooStats {

   class MetropolisHastings :  public TObject {


   public:
      enum FunctionSign {kNegative, kPositive, kSignUnset};
      enum FunctionType {kRegular, kLog, kTypeUnset};

      // default constructor
      MetropolisHastings();

      // alternate constructor
      MetropolisHastings(RooAbsReal& function, const RooArgSet& paramsOfInterest,
            ProposalFunction& proposalFunction, Int_t numIters);

      virtual ~MetropolisHastings() {}

      // main purpose of MetropolisHastings - run Metropolis-Hastings
      // algorithm to generate Markov Chain of points in the parameter space
      virtual MarkovChain* ConstructChain();

      // specify the parameters of interest in the interval
      // only the POI will be added in the chain
      virtual void SetParametersOfInterest(const RooArgSet& set)
      { fPOI.removeAll();  fPOI.add(set);  RemoveConstantParameters(&fPOI); }      
      // specify all the parameters of interest in the interval
      virtual void SetParameters(const RooArgSet& set)
      { fParameters.removeAll();  fParameters.add(set);  RemoveConstantParameters(&fParameters); }
      // set the proposal function for suggesting new points for the MCMC
      virtual void SetProposalFunction(ProposalFunction& proposalFunction)
      { fPropFunc = &proposalFunction; }
      // set the number of iterations to run the metropolis algorithm
      virtual void SetNumIters(Int_t numIters)
      { fNumIters = numIters; }
      // set the number of steps in the chain to discard as burn-in,
      // starting from the first
      virtual void SetNumBurnInSteps(Int_t numBurnInSteps)
      { fNumBurnInSteps = numBurnInSteps; }
      // set the (likelihood) function
      virtual void SetFunction(RooAbsReal& function) { fFunction = &function; }
      // set the sign of the function
      virtual void SetSign(enum FunctionSign sign) { fSign = sign; }
      // set the type of the function
      virtual void SetType(enum FunctionType type) { fType = type; }


   protected:
      RooAbsReal* fFunction; // function that will generate likelihood values
      RooArgSet fParameters; // RooRealVars that define all parameter space
      RooArgSet fPOI; // RooRealVars that define only the POI
      ProposalFunction* fPropFunc; // Proposal function for MCMC integration
      Int_t fNumIters; // number of iterations to run metropolis algorithm
      Int_t fNumBurnInSteps; // number of iterations to discard as burn-in, starting from the first
      enum FunctionSign fSign; // whether the likelihood is negative (like NLL) or positive
      enum FunctionType fType; // whether the likelihood is on a regular, log, (or other) scale

      // whether we should take the step, based on the value of d, fSign, fType
      virtual Bool_t ShouldTakeStep(Double_t d);
      virtual Double_t CalcNLL(Double_t xL);

      ClassDef(MetropolisHastings,1) // Markov Chain Monte Carlo calculator for Bayesian credible intervals
   };
}


#endif
