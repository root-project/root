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

#ifndef RooStats_MCMCInterval
#define RooStats_MCMCInterval

#include "Rtypes.h"

#include "RooStats/ConfInterval.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooMsgService.h"
#include "RooStats/MarkovChain.h"

#include <vector>

class RooNDKeysPdf;
class RooProduct;


namespace RooStats {

   class Heaviside;

   class MCMCInterval : public ConfInterval {


   public:

      /// default constructor
      explicit MCMCInterval(const char* name = 0);

      /// constructor from parameter of interest and Markov chain object
      MCMCInterval(const char* name, const RooArgSet& parameters,
                   MarkovChain& chain);

      enum {DEFAULT_NUM_BINS = 50};
      enum IntervalType {kShortest, kTailFraction};

      ~MCMCInterval() override;

      /// determine whether this point is in the confidence interval
      bool IsInInterval(const RooArgSet& point) const override;

      /// set the desired confidence level (see GetActualConfidenceLevel())
      /// Note: calling this function triggers the algorithm that determines
      /// the interval, so call this after initializing all other aspects
      /// of this IntervalCalculator
      /// Also, calling this function again with a different confidence level
      /// re-triggers the calculation of the interval
      void SetConfidenceLevel(double cl) override;

      /// get the desired confidence level (see GetActualConfidenceLevel())
      double ConfidenceLevel() const override {return fConfidenceLevel;}

      /// return a set containing the parameters of this interval
      /// the caller owns the returned RooArgSet*
      RooArgSet* GetParameters() const override;

      /// get the cutoff bin height for being considered in the
      /// confidence interval
      virtual double GetHistCutoff();

      /// get the cutoff RooNDKeysPdf value for being considered in the
      /// confidence interval
      virtual double GetKeysPdfCutoff();
      ///virtual double GetKeysPdfCutoff() { return fKeysCutoff; }

      /// get the actual value of the confidence level for this interval.
      virtual double GetActualConfidenceLevel();

      /// whether the specified confidence level is a floor for the actual
      /// confidence level (strict), or a ceiling (not strict)
      virtual void SetHistStrict(bool isHistStrict)
      { fIsHistStrict = isHistStrict; }

      /// check if parameters are correct. (dummy implementation to start)
      bool CheckParameters(const RooArgSet& point) const override;

      /// Set the parameters of interest for this interval
      /// and change other internal data members accordingly
      virtual void SetParameters(const RooArgSet& parameters);

      /// Set the MarkovChain that this interval is based on
      virtual void SetChain(MarkovChain& chain) { fChain = &chain; }

      /// Set which parameters go on which axis.  The first list element
      /// goes on the x axis, second (if it exists) on y, third (if it
      /// exists) on z, etc
      virtual void SetAxes(RooArgList& axes);

      /// return a list of RooRealVars representing the axes
      /// you own the returned RooArgList
      virtual RooArgList* GetAxes()
      {
         RooArgList* axes = new RooArgList();
         for (Int_t i = 0; i < fDimension; i++)
            axes->addClone(*fAxes[i]);
         return axes;
      }

      /// get the lowest value of param that is within the confidence interval
      virtual double LowerLimit(RooRealVar& param);

      /// determine lower limit of the lower confidence interval
      virtual double LowerLimitTailFraction(RooRealVar& param);

      /// get the lower limit of param in the shortest confidence interval
      /// Note that this works better for some distributions (ones with exactly
      /// one maximum) than others, and sometimes has little value.
      virtual double LowerLimitShortest(RooRealVar& param);

      /// determine lower limit in the shortest interval by using keys pdf
      virtual double LowerLimitByKeys(RooRealVar& param);

      /// determine lower limit using histogram
      virtual double LowerLimitByHist(RooRealVar& param);

      /// determine lower limit using histogram
      virtual double LowerLimitBySparseHist(RooRealVar& param);

      /// determine lower limit using histogram
      virtual double LowerLimitByDataHist(RooRealVar& param);

      /// get the highest value of param that is within the confidence interval
      virtual double UpperLimit(RooRealVar& param);

      /// determine upper limit of the lower confidence interval
      virtual double UpperLimitTailFraction(RooRealVar& param);

      /// get the upper limit of param in the confidence interval
      /// Note that this works better for some distributions (ones with exactly
      /// one maximum) than others, and sometimes has little value.
      virtual double UpperLimitShortest(RooRealVar& param);

      /// determine upper limit in the shortest interval by using keys pdf
      virtual double UpperLimitByKeys(RooRealVar& param);

      /// determine upper limit using histogram
      virtual double UpperLimitByHist(RooRealVar& param);

      /// determine upper limit using histogram
      virtual double UpperLimitBySparseHist(RooRealVar& param);

      /// determine upper limit using histogram
      virtual double UpperLimitByDataHist(RooRealVar& param);

      /// Determine the approximate maximum value of the Keys PDF
      double GetKeysMax();

      /// set the number of steps in the chain to discard as burn-in,
      /// starting from the first
      virtual void SetNumBurnInSteps(Int_t numBurnInSteps)
      { fNumBurnInSteps = numBurnInSteps; }

      /// set whether to use kernel estimation to determine the interval
      virtual void SetUseKeys(bool useKeys) { fUseKeys = useKeys; }

      /// set whether to use a sparse histogram.  you MUST also call
      /// SetUseKeys(false) to use a histogram.
      virtual void SetUseSparseHist(bool useSparseHist)
      { fUseSparseHist = useSparseHist; }

      /// get whether we used kernel estimation to determine the interval
      virtual bool GetUseKeys() { return fUseKeys; }

      /// get the number of steps in the chain to discard as burn-in,

      /// get the number of steps in the chain to discard as burn-in,
      /// starting from the first
      virtual Int_t GetNumBurnInSteps() { return fNumBurnInSteps; }

      /// set the number of bins to use (same for all axes, for now)
      ///virtual void SetNumBins(Int_t numBins);

      /// Get a clone of the histogram of the posterior
      virtual TH1* GetPosteriorHist();

      /// Get a clone of the keys pdf of the posterior
      virtual RooNDKeysPdf* GetPosteriorKeysPdf();

      /// Get a clone of the (keyspdf * heaviside) product of the posterior
      virtual RooProduct* GetPosteriorKeysProduct();

      /// Get the number of parameters of interest in this interval
      virtual Int_t GetDimension() const { return fDimension; }

      /// Get the markov chain on which this interval is based
      /// You do not own the returned MarkovChain*
      virtual const MarkovChain* GetChain() { return fChain; }

      /// Get a clone of the markov chain on which this interval is based
      /// as a RooDataSet.  You own the returned RooDataSet*
      virtual RooDataSet* GetChainAsDataSet(RooArgSet* whichVars = NULL)
      { return fChain->GetAsDataSet(whichVars); }

      /// Get the markov chain on which this interval is based
      /// as a RooDataSet.  You do not own the returned RooDataSet*
      virtual const RooDataSet* GetChainAsConstDataSet()
      { return fChain->GetAsConstDataSet(); }

      /// Get a clone of the markov chain on which this interval is based
      /// as a RooDataHist.  You own the returned RooDataHist*
      virtual RooDataHist* GetChainAsDataHist(RooArgSet* whichVars = NULL)
      { return fChain->GetAsDataHist(whichVars); }

      /// Get a clone of the markov chain on which this interval is based
      /// as a THnSparse.  You own the returned THnSparse*
      virtual THnSparse* GetChainAsSparseHist(RooArgSet* whichVars = NULL)
      { return fChain->GetAsSparseHist(whichVars); }

      /// Get a clone of the NLL variable from the markov chain
      virtual RooRealVar* GetNLLVar() const
      { return fChain->GetNLLVar(); }

      /// Get a clone of the weight variable from the markov chain
      virtual RooRealVar* GetWeightVar() const
      { return fChain->GetWeightVar(); }

      /// set the acceptable level or error for Keys interval determination
      virtual void SetEpsilon(double epsilon)
      {
         if (epsilon < 0)
            coutE(InputArguments) << "MCMCInterval::SetEpsilon will not allow "
                                  << "negative epsilon value" << std::endl;
         else
            fEpsilon = epsilon;
      }

      /// Set the type of interval to find.  This will only have an effect for
      /// 1-D intervals.  If is more than 1 parameter of interest, then a
      /// "shortest" interval will always be used, since it generalizes directly
      /// to N dimensions
      virtual void SetIntervalType(enum IntervalType intervalType)
      { fIntervalType = intervalType; }
      virtual void SetShortestInterval() { SetIntervalType(kShortest); }

      /// Return the type of this interval
      virtual enum IntervalType GetIntervalType() { return fIntervalType; }

      /// set the left-side tail fraction for a tail-fraction interval
      virtual void SetLeftSideTailFraction(double a) {
         fIntervalType = kTailFraction;
         fLeftSideTF = a;
      }

      /// kbelasco: The inner-workings of the class really should not be exposed
      /// like this in a comment, but it seems to be the only way to give
      /// the user any control over this process, if they desire it
      ///
      /// Set the fraction delta such that
      /// topCutoff (a) is considered == bottomCutoff (b) iff
      /// (TMath::Abs(a - b) < TMath::Abs(fDelta * (a + b)/2))
      /// when determining the confidence interval by Keys
      virtual void SetDelta(double delta)
      {
         if (delta < 0.)
            coutE(InputArguments) << "MCMCInterval::SetDelta will not allow "
                                  << "negative delta value" << std::endl;
         else
            fDelta = delta;
      }

   private:
      inline bool AcceptableConfLevel(double confLevel);
      inline bool WithinDeltaFraction(double a, double b);

   protected:
      // data members
      RooArgSet  fParameters;     ///< parameters of interest for this interval
      MarkovChain* fChain;        ///< the markov chain
      double fConfidenceLevel;  ///< Requested confidence level (eg. 0.95 for 95% CL)

      RooDataHist* fDataHist;     ///< the binned Markov Chain data
      THnSparse* fSparseHist;     ///< the binned Markov Chain data
      double fHistConfLevel;    ///< the actual conf level determined by hist
      double fHistCutoff;       ///< cutoff bin size to be in interval

      RooNDKeysPdf* fKeysPdf;     ///< the kernel estimation pdf
      RooProduct* fProduct;       ///< the (keysPdf * heaviside) product
      Heaviside* fHeaviside;      ///< the Heaviside function
      RooDataHist* fKeysDataHist; ///< data hist representing product
      RooRealVar* fCutoffVar;     ///< cutoff variable to use for integrating keys pdf
      double fKeysConfLevel;    ///< the actual conf level determined by keys
      double fKeysCutoff;       ///< cutoff keys pdf value to be in interval
      double fFull;             ///< Value of intergral of fProduct

      double fLeftSideTF;       ///< left side tail-fraction for interval
      double fTFConfLevel;      ///< the actual conf level of tail-fraction interval
      std::vector<Int_t> fVector; ///< vector containing the Markov chain data
      double fVecWeight;        ///< sum of weights of all entries in fVector
      double fTFLower;          ///< lower limit of the tail-fraction interval
      double fTFUpper;          ///< upper limit of the tail-fraction interval

      TH1* fHist;                 ///< the binned Markov Chain data

      bool fUseKeys;            ///< whether to use kernel estimation
      bool fUseSparseHist;      ///< whether to use sparse hist (vs. RooDataHist)
      bool fIsHistStrict;       ///< whether the specified confidence level is a
                                  ///< floor for the actual confidence level (strict),
                                  ///< or a ceiling (not strict) for determination by
                                  ///< histogram
      Int_t fDimension;           ///< number of variables
      Int_t fNumBurnInSteps;      ///< number of steps to discard as burn in, starting
                                  ///< from the first
      RooRealVar** fAxes;         ///< array of pointers to RooRealVars representing
                                  ///< the axes of the histogram
                                  ///< fAxes[0] represents x-axis, [1] y, [2] z, etc

      double fEpsilon;          ///< acceptable error for Keys interval determination

      double fDelta;            ///< topCutoff (a) considered == bottomCutoff (b) iff
                                  ///< (TMath::Abs(a - b) < TMath::Abs(fDelta * (a + b)/2));
                                  ///< Theoretically, the Abs is not needed here, but
                                  ///< floating-point arithmetic does not always work
                                  ///< perfectly, and the Abs doesn't hurt
      enum IntervalType fIntervalType;


      // functions
      virtual void DetermineInterval();
      virtual void DetermineShortestInterval();
      virtual void DetermineTailFractionInterval();
      virtual void DetermineByHist();
      virtual void DetermineBySparseHist();
      virtual void DetermineByDataHist();
      virtual void DetermineByKeys();
      virtual void CreateHist();
      virtual void CreateSparseHist();
      virtual void CreateDataHist();
      virtual void CreateKeysPdf();
      virtual void CreateKeysDataHist();
      virtual void CreateVector(RooRealVar* param);
      inline virtual double CalcConfLevel(double cutoff, double full);

      ClassDefOverride(MCMCInterval,1)  // Concrete implementation of a ConfInterval based on MCMC calculation

   };
}

#endif
