// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProfileLikelihoodCalculator
#define ROOSTATS_ProfileLikelihoodCalculator

#include "RooStats/CombinedCalculator.h"

#include "RooStats/LikelihoodInterval.h"

namespace RooStats {

   class LikelihoodInterval;

 class ProfileLikelihoodCalculator : public CombinedCalculator {

   public:

      /// Default constructor (needed for I/O)
      ProfileLikelihoodCalculator();

      /// Constructor from data, from a full model pdf describing both parameter of interest and nuisance parameters
      /// and from the set specifying the parameter of interest (POI).
      /// There is no need to specify the nuisance parameters since they are all other parameters of the model.
      /// When using the calculator for performing an hypothesis test one needs to provide also a snapshot (a copy)
      /// defining the null parameters and their value. There is no need to pass the alternate parameters. These
      /// will be obtained by the value maximizing the likelihood function
      ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest,
                                  double size = 0.05, const RooArgSet* nullParams = nullptr );


      /// Constructor from data and a model configuration
      /// If the ModelConfig defines a prior pdf for any of the parameters those will be included as constrained terms in the
      /// likelihood function
      ProfileLikelihoodCalculator(RooAbsData& data, ModelConfig & model, double size = 0.05);


      ~ProfileLikelihoodCalculator() override;

      /// Return a likelihood interval. A global fit to the likelihood is performed and
      /// the interval is constructed using the profile likelihood ratio function of the POI.
      LikelihoodInterval* GetInterval() const override ;

      /// Return the hypothesis test result obtained from the likelihood ratio of the
      /// maximum likelihood value with the null parameters fixed to their values, with respect to keeping all parameters
      /// floating (global maximum likelihood value).
      HypoTestResult* GetHypoTest() const override;



   protected:

    /// clear internal fit result
    void DoReset() const;

    /// perform a global fit
    RooFit::OwningPtr<RooAbsReal> DoGlobalFit() const;

    /// minimize likelihood
    static RooFitResult * DoMinimizeNLL(RooAbsReal * nll);


    mutable RooFitResult * fFitResult;  //// internal  result of global fit
    mutable bool fGlobalFitDone;        ///< flag to control if a global fit has been done


    ClassDefOverride(ProfileLikelihoodCalculator,2) // A concrete implementation of CombinedCalculator that uses the ProfileLikelihood ratio.

   };
}
#endif
