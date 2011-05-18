// @(#)root/roostats:$Id$
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

#include "Math/IFunctionfwd.h"

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
class TF1;

namespace RooStats {

   class ModelConfig; 
   class SimpleInterval; 

   class BayesianCalculator : public IntervalCalculator, public TNamed {

   public:

      // constructor
      BayesianCalculator( );

      BayesianCalculator( RooAbsData& data,
                          RooAbsPdf& pdf,
                          const RooArgSet& POI,
                          RooAbsPdf& priorPOI,
                          const RooArgSet* nuisanceParameters = 0 );

      BayesianCalculator( RooAbsData& data,
                          ModelConfig& model );

      // destructor
      virtual ~BayesianCalculator();

      // get the plot with option to get it normalized 
      RooPlot* GetPosteriorPlot(bool norm = false, double precision = 0.01) const; 

      // return posterior pdf (object is managed by the BayesianCalculator class)
      RooAbsPdf* GetPosteriorPdf() const; 
      // return posterior function (object is managed by the BayesianCalculator class)
      RooAbsReal* GetPosteriorFunction() const; 

      // compute the interval. By Default a central interval is computed 
      // By using SetLeftTileFraction can control if central/ upper/lower interval
      // For shortest interval use SetShortestInterval(true)
      virtual SimpleInterval* GetInterval() const ; 

      virtual void SetData( RooAbsData & data ) {
         fData = &data;
         ClearAll();
      }


      // set the model via the ModelConfig
      virtual void SetModel( const ModelConfig& model ); 

      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize( Double_t size ) {
         fSize = size;
         fValidInterval = false; 
      }
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel( Double_t cl ) { SetTestSize(1.-cl); }
      // Get the size of the test (eg. rate of Type I error)
      virtual Double_t Size() const { return fSize; }
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel() const { return 1.-fSize; }

      // set the fraction of probability content on the left tail
      // Central limits use 0.5 (default case)  
      // for upper limits it is 0 and 1 for lower limit
      // For shortest intervals a negative value (i.e. -1) must be given
      void SetLeftSideTailFraction(Double_t leftSideFraction )  {fLeftSideFraction = leftSideFraction;} 

      // set the Bayesian calculator to compute the shorest interval (default is central interval) 
      // to switch off SetLeftSideTailFraction to the rght value
      void SetShortestInterval() { fLeftSideFraction = -1; }

      // set the precision of the Root Finder 
      void SetBrfPrecision( double precision ) { fBrfPrecision = precision; }

      // use directly the approximate posterior function obtained by binning it in nbins
      // by default the cdf is used by integrating the posterior
      // if a value of nbin <= 0 the cdf function will be used
      void SetScanOfPosterior(int nbin = 100) { fNScanBins = nbin; }


      // set the integration type (possible type are) 
      // 1D: adaptive, gauss, nonadaptive
      // multidim: adaptive, vegas, miser, plain. These last 3 are based on MC integration
      // if type = 0 use default specified via class IntegratorMultiDimOptions::SetDefaultIntegrator
      void SetIntegrationType(const char * type); 

      // return the mode (most probable value of the posterior function) 
      double GetMode() const; 

   protected:

      void ClearAll() const; 

      void ApproximatePosterior() const; 

      void ComputeIntervalFromApproxPosterior(double c1, double c2) const;

      void ComputeIntervalFromCdf(double c1, double c2) const; 

      void ComputeIntervalUsingRooFit(double c1, double c2) const;

      void ComputeShortestInterval() const; 
   
   private:

      // plan to replace the above: return a SimpleInterval integrating 
      // over all other parameters except the one specified as argument
      //virtual SimpleInterval* GetInterval( RooRealVar* parameter  ) const { return 0; }
    
      RooAbsData* fData;
      RooAbsPdf* fPdf;
      RooArgSet fPOI;
      RooAbsPdf* fPriorPOI;
      RooArgSet fNuisanceParameters;

      mutable RooAbsPdf* fProductPdf;              // internal pointer to model * prior
      mutable RooAbsReal* fLogLike;                // internal pointer to log likelihood function
      mutable RooAbsReal* fLikelihood;             // internal pointer to likelihood function 
      mutable RooAbsReal* fIntegratedLikelihood;   // integrated likelihood function, i.e - unnormalized posterior function  
      mutable RooAbsPdf* fPosteriorPdf;             // normalized (on the poi) posterior pdf 
      mutable ROOT::Math::IGenFunction * fPosteriorFunction;   // function representing the posterior
      mutable TF1 * fApproxPosterior;    // TF1 representing the scanned posterior function
      mutable Double_t  fLower;    // computer lower interval bound
      mutable Double_t  fUpper;    // upper interval bound
      mutable Double_t  fNLLMin;   // minimum value of Nll 
      double fSize;  // size used for getting the interval
      double fLeftSideFraction;    // fraction of probability content on left side of interval
      double fBrfPrecision;     // root finder precision
      int fNScanBins;            // number of bins to scan, if = -1 no scan is done (default)
      mutable Bool_t    fValidInterval; 
     


      TString fIntegrationType; 

   protected:

      ClassDef(BayesianCalculator,1)  // BayesianCalculator class

   };
}

#endif
