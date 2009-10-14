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
class RooAbsReal;

namespace RooStats {

   class ModelConfig; 
   class SimpleInterval; 

   class BayesianCalculator : public IntervalCalculator, public TNamed {

   public:

      // constructor
      BayesianCalculator( );

      BayesianCalculator( RooAbsData& data,
                          RooAbsPdf& pdf,
                          const RooArgSet & POI,
                          RooAbsPdf& priorPOI,
                          const RooArgSet* nuisanceParameters = 0 );

      BayesianCalculator( RooAbsData& data,
                          ModelConfig & model);

      // destructor
      virtual ~BayesianCalculator() ;

      RooPlot* GetPosteriorPlot() const; 

      // return posterior pdf (object is managed by the BayesianCalculator class)
      RooAbsPdf * GetPosteriorPdf(const char * = 0) const; 

      virtual SimpleInterval* GetInterval() const ; 

      virtual void  SetData(RooAbsData & data) { fData = &data; }

      virtual void SetModel(const ModelConfig & model); 

      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) {
         fSize = size;
         if (fInterval) delete fInterval; fInterval = 0;  
      }
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) { SetTestSize( 1. - cl); }
      // Get the size of the test (eg. rate of Type I error)
      virtual Double_t Size() const {return fSize;}
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}

   protected:

      void ClearAll() const; 
   
   private:
    
      RooAbsData* fData;
      RooAbsPdf* fPdf;
      RooArgSet fPOI;
      RooAbsPdf* fPriorPOI;
      RooArgSet fNuisanceParameters;

      mutable RooAbsPdf* fProductPdf; 
      mutable RooAbsReal* fLogLike; 
      mutable RooAbsReal* fLikelihood; 
      mutable RooAbsReal* fIntegratedLikelihood; 
      mutable RooAbsPdf* fPosteriorPdf; 
      mutable SimpleInterval* fInterval;     // cached pointer to resulting interval

      double fSize; 


   protected:

      ClassDef(BayesianCalculator,1)  // BayesianCalculator class

   };
}

#endif
