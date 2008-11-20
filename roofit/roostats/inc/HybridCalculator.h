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


namespace RooStats {

   class HybridCalculator /*: public HypoTestCalculator*/ {  /// TO DO: inheritance

   public:
      /// Constructor for HybridCalculator
      HybridCalculator(const char *name,
                       const char *title,
                       RooAbsPdf& sb_model,
                       RooAbsPdf& b_model,
                       RooArgList& observables,
                       RooArgSet& parameters,
                       RooAbsPdf& prior_pdf);

      /// Destructor of HybridCalculator
      virtual ~HybridCalculator();

      void SetTestStatistics(int index);
      void Calculate(RooAbsData& data, unsigned int nToys, bool usePriors);
      void RunToys(unsigned int nToys, bool usePriors); // private?
      void Print(const char* options);

   private:
      const char* fName; /// TO DO: put to TNamed inherited
      const char* fTitle; /// TO DO: put to TNamed inherited
      RooAbsPdf& fSbModel;
      RooAbsPdf& fBModel;
      RooArgList& fObservables;
      RooArgSet& fParameters;
      RooAbsPdf& fPriorPdf;
      unsigned int fTestStatisticsIdx;

   protected:
      ClassDef(HybridCalculator,1)  // Hypothesis test calculator using a Bayesian-frequentist hybrid method
   };
}

#endif
