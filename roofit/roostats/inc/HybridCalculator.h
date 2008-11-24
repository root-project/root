// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HybridCalculator
#define ROOSTATS_HybridCalculator

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#include <vector>

#include "TH1.h"

#include "RooStats/HybridResult.h"

namespace RooStats {

   class HybridCalculator : /*public HypoTestCalculator ,*/ public TNamed {

   public:
      /// Constructor for HybridCalculator
      HybridCalculator(const char *name,
                       const char *title,
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgList& observables,
                       RooArgSet& nuisance_parameters,
                       RooAbsPdf& prior_pdf);

      /// Destructor of HybridCalculator
      virtual ~HybridCalculator();

      void SetTestStatistics(int index);
      HybridResult* Calculate(TH1& data, unsigned int nToys, bool usePriors);
      HybridResult* Calculate(RooTreeData& data, unsigned int nToys, bool usePriors);
      HybridResult* Calculate(unsigned int nToys, bool usePriors);
      void PrintMore(const char* options);

   private:
      void RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors);

      RooAbsPdf& fSbModel; // The pdf of the signal+background model
      RooAbsPdf& fBModel; // The pdf of the background model
      RooArgList& fObservables; // Collection of the observables of the model
      RooArgSet& fParameters; // Collection of the nuisance parameters in the model
      RooAbsPdf& fPriorPdf; // Prior PDF of the nuisance parameters
      unsigned int fTestStatisticsIdx; // Index of the test statistics to use

   protected:
      ClassDef(HybridCalculator,1)  // Hypothesis test calculator using a Bayesian-frequentist hybrid method
   };
}

#endif
