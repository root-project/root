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
                          RooAbsPdf& priorPdf,
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

      // specify the parameters of interest in the interval
      virtual void SetParameters(const RooArgSet& set) { fPOI.removeAll(); fPOI.add(set); }

      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet& set) {fNuisanceParameters.removeAll(); fNuisanceParameters.add(set);}

      // Set only the Prior Pdf 
      virtual void SetPriorPdf(RooAbsPdf& pdf) { fPriorPdf = &pdf; }

      // set the conditional observables which will be used when creating the NLL
      // so the pdf's will not be normalized on the conditional observables when computing the NLL 
      virtual void SetConditionalObservables(const RooArgSet& set) {fConditionalObs.removeAll(); fConditionalObs.add(set);}

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

      // set the number of iterations when running a MC integration algorithm
      // If not set use default algorithmic values 
      // In case of ToyMC sampling of the nuisance the value is 100
      // In case of using the GSL MCintegrations types the default value is 
      // defined in ROOT::Math::IntegratorMultiDimOptions::DefaultNCalls()
      virtual void SetNumIters(Int_t numIters)  { fNumIterations = numIters; }

      // set the integration type (possible type are) : 
      void SetIntegrationType(const char * type); 

      // return the mode (most probable value of the posterior function) 
      double GetMode() const; 

      // force the nuisance pdf when using the toy mc sampling
      void ForceNuisancePdf(RooAbsPdf & pdf) { fNuisancePdf = &pdf; }

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
    
      RooAbsData* fData;                          // data set 
      RooAbsPdf* fPdf;                           // model pdf  (could contain the nuisance pdf as constraint term)
      RooArgSet fPOI;                            // POI
      RooAbsPdf* fPriorPdf;                      // prior pdf (typically for the POI)
      RooAbsPdf* fNuisancePdf;                   // nuisance pdf (needed when using nuisance sampling technique)
      RooArgSet fNuisanceParameters;             // nuisance parameters
      RooArgSet fConditionalObs    ;             // conditional observables

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
      mutable int fNScanBins;            // number of bins to scan, if = -1 no scan is done (default)
      int fNumIterations;        // number of iterations (when using ToyMC)
      mutable Bool_t    fValidInterval; 
      


      TString fIntegrationType; 

   protected:

      ClassDef(BayesianCalculator,2)  // BayesianCalculator class

   };
}

#endif
