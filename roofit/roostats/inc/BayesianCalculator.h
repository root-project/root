// @(#)root/roostats:$Id: ModelConfig.h 27519 2009-02-19 13:31:41Z pellicci $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_BayesianCalculator
#define ROOSTATS_BayesianCalculator

#include "TNamed.h"

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

#ifndef ROOSTATS_IntervalCalculator
#include "RooStats/IntervalCalculator.h"
#endif

#ifndef ROOSTATS_SimpleInterval
#include "RooStats/SimpleInterval.h"
#endif

class RooAbsData; 
class RooAbsPdf; 
class RooPlot; 

namespace RooStats {

   class ModelConfig; 
   class SimpleInterval; 

   class BayesianCalculator : public IntervalCalculator, public TNamed {

   public:

      // constructor
      BayesianCalculator( ) {}

      BayesianCalculator( RooAbsData& data,
                          RooAbsPdf& pdf,
                          const RooArgSet & POI,
                          RooAbsPdf& priorPOI,
                          const RooArgSet* nuisanceParameters = 0 );

      BayesianCalculator( RooAbsData& data,
                          ModelConfig & model);

      // destructor
      virtual ~BayesianCalculator() ;

      RooPlot* PlotPosterior() ; 

      virtual SimpleInterval* GetInterval() const ; 

   private:
    
      RooAbsData* fData;
      RooAbsPdf* fPdf;
      RooArgSet fPOI;
      RooAbsPdf* fPriorPOI;
      RooArgSet fNuisanceParameters;

      mutable Double_t fLowerLimit;
      mutable Double_t fUpperLimit;

   protected:

      ClassDef(BayesianCalculator,1)  // BayesianCalculator class

   };
}

#endif
