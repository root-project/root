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

#ifndef ROOSTATS_SamplingDistribution
#define ROOSTATS_SamplingDistribution

#include "TNamed.h"

#include "Rtypes.h"
#include "RooDataSet.h"

#include <vector>

namespace RooStats {

 class SamplingDistribution : public TNamed {

   public:

    /// Constructor for SamplingDistribution
    SamplingDistribution(const char *name,const char *title, std::vector<Double_t>& samplingDist, const char * varName = 0);
    SamplingDistribution(const char *name,const char *title,
          std::vector<Double_t>& samplingDist, std::vector<Double_t>& sampleWeights, const char * varName = 0);


    SamplingDistribution(const char *name,const char *title, const char * varName = 0);

    SamplingDistribution(const char *name,const char *title, RooDataSet& dataSet, const char * columnName = 0, const char * varName = 0);

    /// Default constructor for SamplingDistribution
    SamplingDistribution();

    /// Destructor of SamplingDistribution
    ~SamplingDistribution() override;

    /// get the inverse of the Cumulative distribution function
    Double_t InverseCDF(Double_t pvalue);

    /// get the inverse of the Cumulative distribution function
    Double_t InverseCDFInterpolate(Double_t pvalue);

    /// get the inverse of the Cumulative distribution function
    /// together with the inverse based on sampling variation
    Double_t InverseCDF(Double_t pvalue, Double_t sigmaVariaton, Double_t& inverseVariation);

    /// merge two sampling distributions
    void Add(const SamplingDistribution* other);

    /// size of samples
    Int_t GetSize() const{return fSamplingDist.size();}

    /// Get test statistics values
    const std::vector<Double_t> & GetSamplingDistribution() const {return fSamplingDist;}
    /// Get the sampling weights
    const std::vector<Double_t> & GetSampleWeights() const {return fSampleWeights;}

    const TString GetVarName() const {return fVarName;}

    /// numerical integral in these limits
    Double_t Integral(Double_t low, Double_t high, bool normalize = true, bool lowClosed = true, bool highClosed = false) const;

    /// numerical integral in these limits including error estimation
    Double_t IntegralAndError(Double_t & error, Double_t low, Double_t high, bool normalize = true,
                              bool lowClosed = true, bool highClosed = false) const;

    /// calculate CDF as a special case of Integral(...) with lower limit equal to -inf
    Double_t CDF(Double_t x) const;

  private:

    mutable std::vector<Double_t> fSamplingDist;  ///< vector of points for the sampling distribution
    mutable std::vector<Double_t> fSampleWeights; ///< vector of weights for the samples

    TString fVarName;

    mutable std::vector<Double_t> fSumW;   ///<! Cached vector with sum of the weight used to compute integral
    mutable std::vector<Double_t> fSumW2;  ///<! Cached vector with sum of the weight used to compute integral error

  protected:

    /// internal function to sort values
    void SortValues() const;

    ClassDefOverride(SamplingDistribution,2)  /// Class containing the results of the HybridCalculator
  };
}

#endif
