// @(#)root/roostats:$Id: MCMCInterval.h 26805 2009-06-17 14:31:02Z kbelasco $
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_MCMCInterval
#define RooStats_MCMCInterval

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef RooStats_ConfInterval
#include "RooStats/ConfInterval.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "TH1.h"


namespace RooStats {

   class MCMCInterval : public ConfInterval {


   public:
      MCMCInterval();
      MCMCInterval(const char* name);
      MCMCInterval(const char* name, const char* title);
      MCMCInterval(const char* name, const char* title, RooArgSet& parameters,
                   RooDataSet& chain);

      enum {DEFAULT_NUM_BINS = 50};

      virtual ~MCMCInterval()
      {
         delete[] fAxes;
         delete fHist;
         delete fData;
         delete[] fNumBins;
      }
        
      // determine whether this point is in the confidence interval
      virtual Bool_t IsInInterval(RooArgSet& point);

      // set the desired confidence level (see GetActualConfidenceLevel())
      // Note: calling this function triggers the algorithm that determines
      // the interval, so call this after initializing all other aspects
      // of this IntervalCalculator
      // Also, calling this function again with a different confidence level
      // retriggers the calculation of the interval
      virtual void SetConfidenceLevel(Double_t cl);

      // get the desired confidence level (see GetActualConfidenceLevel())
      virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 
      // do we want it to return list of parameters
      virtual RooArgSet* GetParameters() const;

      // get the cutoff bin height for being considered in the
      // confidence interval
      virtual Double_t GetCutoff() { return fCutoff; }

      // get the actual value of the confidence level for this interval.
      // It is >= the specified confidence level because the interval contains
      // all bins with the cutoff height (or higher), until at least the desired
      // confidence level is reached.
      // Returns (Sum of bin heights in interval) / (Sum of all bin heights)
      virtual Double_t GetActualConfidenceLevel()
      { return fIntervalSum/fHist->GetSumOfWeights(); }

      // whether the specified confidence level is a floor for the actual
      // confidence level (strict), or a ceiling (not strict)
      virtual void SetStrict(Bool_t isStrict) { fIsStrict = isStrict; }

      // check if parameters are correct. (dummy implementation to start)
      Bool_t CheckParameters(RooArgSet& point) const;

      // Set the parameters of interest for this interval
      // and change other internal data members accordingly
      virtual void SetParameters(RooArgSet& parameters);

      // Set which parameters go on which axis.  The first list element
      // goes on the x axis, second (if it exists) on y, third (if it
      // exists) on z.
      virtual void SetAxes(RooArgList& axes);

      // get the lower limit of param in the confidence interval
      // Note that this works better for some distributions (ones with exactly
      // one maximum) than others, and sometimes has little value.
      virtual Double_t LowerLimit(RooRealVar& param);

      // get the upper limit of param in the confidence interval
      // Note that this works better for some distributions (ones with exactly
      // one maximum) than others, and sometimes has little value.
      virtual Double_t UpperLimit(RooRealVar& param);

      // set the number of steps in the chain to discard as burn-in,
      // starting from the first
      virtual void SetNumBurnInSteps(Int_t numBurnInSteps)
      { fNumBurnInSteps = numBurnInSteps; }

      // set the number of bins to use (same for all axes, for now)
      virtual void SetNumBins(Int_t numBins);

      // Get a clone of the histogram of the posterior
      virtual TH1* GetPosteriorHist();

      // Get the markov chain on which this interval is based
      virtual const RooDataSet* GetChain() { return fData; }

   protected:
      // data members
      RooArgSet* fParameters; // parameters of interest for this interval
      RooDataSet* fData; // the markov chain
      TH1* fHist; // histogram generated from data to determine binning
      Double_t fConfidenceLevel; // Requested confidence level (eg. 0.95 for 95% CL)
      Double_t fCutoff; // cutoff bin size to be in interval
      Bool_t fIsStrict; // whether the specified confidence level is a floor
                        // for the actual confidence level (strict), or a 
                        // ceiling (not strict)
      Int_t fDimension; // number of variables
      Int_t fNumBurnInSteps; // number of steps to discard as burn in, starting from the first
      Int_t* fNumBins; // number of bins for each dimension
      Double_t fIntervalSum; // sum of heights of bins in the interval
      RooRealVar** fAxes; // array of pointers to RooRealVars representing
                          // the axes of the histogram
                          // fAxes[0] represents x-axis, [1] y, [2] z
      Int_t fPreferredNumBins; // number of bins client wants

      // functions
      virtual void DetermineInterval();
      virtual void CreateHistogram();

      ClassDef(MCMCInterval,1)  // Concrete implementation of a ConfInterval based on MCMC calculation
      
   };
}

#endif
