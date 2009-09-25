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

#ifndef ROOSTATS_ConfInterval
#include "RooStats/ConfInterval.h"
#endif
#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_ARG_LIST
#include "RooArgList.h"
#endif
#ifndef ROO_DATA_HIST
#include "RooDataHist.h"
#endif
#ifndef ROO_DATA_SET
#include "RooDataSet.h"
#endif
#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif
#ifndef ROO_KEYS_PDF
#include "RooNDKeysPdf.h"
#endif
#ifndef ROOSTATS_MarkovChain
#include "RooStats/MarkovChain.h"
#endif
#ifndef ROOT_TH1
#include "TH1.h"
#endif
#ifndef ROO_PRODUCT
#include "RooProduct.h"
#endif
#ifndef RooStats_Heavyside
#include "RooStats/Heavyside.h"
#endif
#ifndef ROO_PRODUCT
#include "RooProduct.h"
#endif
#ifndef ROOT_THnSparse
#include "THnSparse.h"
#endif

namespace RooStats {

   class MCMCInterval : public ConfInterval {


   public:
      MCMCInterval();
      MCMCInterval(const char* name);
      MCMCInterval(const char* name, const char* title);
      MCMCInterval(const char* name, const char* title, RooArgSet& parameters,
                   MarkovChain& chain);

      enum {DEFAULT_NUM_BINS = 50};

      virtual ~MCMCInterval()
      {
         delete[] fAxes;
         delete fHist;
         delete fChain;
         // kbelasco: check here for memory management errors
         delete fDataHist;
         delete fSparseHist;
         delete fKeysPdf;
         delete fProduct;
         delete fHeavyside;
         delete fKeysDataHist;
         delete fCutoffVar;
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
 
      // return a set containing the parameters of this interval
      // the caller owns the returned RooArgSet*
      virtual RooArgSet* GetParameters() const;

      // get the cutoff bin height for being considered in the
      // confidence interval
      virtual Double_t GetHistCutoff();

      // get the cutoff RooNDKeysPdf value for being considered in the
      // confidence interval
      virtual Double_t GetKeysPdfCutoff();
      //virtual Double_t GetKeysPdfCutoff() { return fKeysCutoff; }

      // get the actual value of the confidence level for this interval.
      virtual Double_t GetActualConfidenceLevel();

      // get the sum of all bin weights
      virtual Double_t GetSumOfWeights() const
      { return fDataHist->sum(kFALSE); }

      // whether the specified confidence level is a floor for the actual
      // confidence level (strict), or a ceiling (not strict)
      virtual void SetHistStrict(Bool_t isHistStrict) { fIsHistStrict = isHistStrict; }

      // check if parameters are correct. (dummy implementation to start)
      Bool_t CheckParameters(RooArgSet& point) const;

      // Set the parameters of interest for this interval
      // and change other internal data members accordingly
      virtual void SetParameters(RooArgSet& parameters);

      // Set the MarkovChain that this interval is based on
      virtual void SetChain(MarkovChain& chain) { fChain = &chain; }

      // Set which parameters go on which axis.  The first list element
      // goes on the x axis, second (if it exists) on y, third (if it
      // exists) on z, etc
      virtual void SetAxes(RooArgList& axes);

      // return a list of RooRealVars representing the axes
      // you own the returned RooArgList
      virtual RooArgList* GetAxes()
      {
         RooArgList* axes = new RooArgList();
         for (Int_t i = 0; i < fDimension; i++)
            axes->addClone(*fAxes[i]);
         return axes;
      }

      // get the lower limit of param in the confidence interval
      // Note that this works better for some distributions (ones with exactly
      // one maximum) than others, and sometimes has little value.
      virtual Double_t LowerLimit(RooRealVar& param);

      // determine lower limit using keys pdf
      virtual Double_t LowerLimitByKeys(RooRealVar& param);

      // determine upper limit using keys pdf
      virtual Double_t UpperLimitByKeys(RooRealVar& param);

      // determine lower limit using histogram
      virtual Double_t LowerLimitByHist(RooRealVar& param);

      // determine lower limit using histogram
      virtual Double_t LowerLimitBySparseHist(RooRealVar& param);

      // determine lower limit using histogram
      virtual Double_t LowerLimitByDataHist(RooRealVar& param);

      // determine upper limit using histogram
      virtual Double_t UpperLimitByHist(RooRealVar& param);

      // determine upper limit using histogram
      virtual Double_t UpperLimitBySparseHist(RooRealVar& param);

      // determine upper limit using histogram
      virtual Double_t UpperLimitByDataHist(RooRealVar& param);

      // get the upper limit of param in the confidence interval
      // Note that this works better for some distributions (ones with exactly
      // one maximum) than others, and sometimes has little value.
      virtual Double_t UpperLimit(RooRealVar& param);

      // set the number of steps in the chain to discard as burn-in,
      // starting from the first
      virtual void SetNumBurnInSteps(Int_t numBurnInSteps)
      { fNumBurnInSteps = numBurnInSteps; }

      // set whether to use kernel estimation to determine the interval
      virtual void SetUseKeys(Bool_t useKeys) { fUseKeys = useKeys; }

      // set whether to use a sparse histogram.  you MUST also call
      // SetUseKeys(kFALSE) to use a histogram.
      virtual void SetUseSparseHist(Bool_t useSparseHist)
      { fUseSparseHist = useSparseHist; }

      // get whether we used kernel estimation to determine the interval
      virtual Bool_t GetUseKeys() { return fUseKeys; }

      // get the number of steps in the chain to disard as burn-in,

      // get the number of steps in the chain to disard as burn-in,
      // starting from the first
      virtual Int_t GetNumBurnInSteps() { return fNumBurnInSteps; }

      // set the number of bins to use (same for all axes, for now)
      //virtual void SetNumBins(Int_t numBins);

      // Get a clone of the histogram of the posterior
      virtual TH1* GetPosteriorHist();

      // Get a clone of the keys pdf of the posterior
      virtual RooNDKeysPdf* GetPosteriorKeysPdf();

      // Get a clone of the (keyspdf * heavyside) product of the posterior
      virtual RooProduct* GetPosteriorKeysProduct();

      // Get the number of parameters of interest in this interval
      virtual Int_t GetDimension() const { return fDimension; }

      // Get the markov chain on which this interval is based
      // You do not own the returned MarkovChain*
      virtual const MarkovChain* GetChain() { return fChain; }

      // Get a clone of the markov chain on which this interval is based
      // as a RooDataSet.  You own the returned RooDataSet*
      virtual RooDataSet* GetChainAsDataSet(RooArgSet* whichVars = NULL)
      { return fChain->GetAsDataSet(whichVars); }

      // Get the markov chain on which this interval is based
      // as a RooDataSet.  You do not own the returned RooDataSet*
      virtual const RooDataSet* GetChainAsConstDataSet()
      { return fChain->GetAsConstDataSet(); }

      // Get a clone of the markov chain on which this interval is based
      // as a RooDataHist.  You own the returned RooDataHist*
      virtual RooDataHist* GetChainAsDataHist(RooArgSet* whichVars = NULL)
      { return fChain->GetAsDataHist(whichVars); }

      // Get a clone of the markov chain on which this interval is based
      // as a THnSparse.  You own the returned THnSparse*
      virtual THnSparse* GetChainAsSparseHist(RooArgSet* whichVars = NULL)
      { return fChain->GetAsSparseHist(whichVars); }

      // Get a clone of the NLL variable from the markov chain
      virtual RooRealVar* GetNLLVar() const { return fChain->GetNLLVar(); }

      // Get a clone of the weight variable from the markov chain
      virtual RooRealVar* GetWeightVar() const { return fChain->GetWeightVar(); }

      // set the acceptable level or error for Keys interval determination
      virtual void SetEpsilon(Double_t epsilon)
      {
         if (epsilon < 0)
            coutE(InputArguments) << "MCMCInterval::SetEpsilon will not allow "
                                  << "negative epsilon value" << endl;
         else
            fEpsilon = epsilon;
      }

   private:
      inline Bool_t AcceptableConfLevel(Double_t confLevel);

   protected:
      // data members
      RooArgSet* fParameters; // parameters of interest for this interval
      MarkovChain* fChain; // the markov chain
      RooDataHist* fDataHist; // the binned Markov Chain data
      RooNDKeysPdf* fKeysPdf; // the kernel estimation pdf
      RooProduct* fProduct; // the (keysPdf * heavyside) product
      Heavyside* fHeavyside; // the Heavyside function
      RooDataHist* fKeysDataHist; // data hist representing product
      TH1* fHist; // the binned Markov Chain data
      THnSparse* fSparseHist; // the binned Markov Chain data
      Double_t fConfidenceLevel; // Requested confidence level (eg. 0.95 for 95% CL)
      Double_t fHistConfLevel; // the actual conf level determined by hist
      Double_t fKeysConfLevel; // the actual conf level determined by keys
      Double_t fHistCutoff; // cutoff bin size to be in interval
      Double_t fKeysCutoff; // cutoff keys pdf value to be in interval
      Double_t fFull; // Value of intergral of fProduct
      RooRealVar* fCutoffVar; // cutoff variable to use for integrating keys pdf
      Bool_t fUseKeys; // whether to use kernel estimation
      Bool_t fUseSparseHist; // whether to use sparse hist (vs. RooDataHist)
      Bool_t fIsHistStrict; // whether the specified confidence level is a floor
                            // for the actual confidence level (strict), or a 
                            // ceiling (not strict) for determination by histogram
      Int_t fDimension; // number of variables
      Int_t fNumBurnInSteps; // number of steps to discard as burn in, starting from the first
      Double_t fIntervalSum; // sum of heights of bins in the interval
      RooRealVar** fAxes; // array of pointers to RooRealVars representing
                          // the axes of the histogram
                          // fAxes[0] represents x-axis, [1] y, [2] z, etc
      Double_t fEpsilon; // acceptable error for Keys interval determination

      // functions
      virtual void DetermineInterval();
      virtual void DetermineByHist();
      virtual void DetermineBySparseHist();
      virtual void DetermineByDataHist();
      virtual void DetermineByKeys();
      virtual void CreateHist();
      virtual void CreateSparseHist();
      virtual void CreateDataHist();
      virtual void CreateKeysPdf();
      virtual void CreateKeysDataHist();
      inline virtual Double_t CalcConfLevel(Double_t cutoff, Double_t full);

      ClassDef(MCMCInterval,1)  // Concrete implementation of a ConfInterval based on MCMC calculation
      
   };
}

#endif
