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
    SamplingDistribution(const char *name,const char *title, std::vector<double>& samplingDist, const char * varName = nullptr);
    SamplingDistribution(const char *name,const char *title,
          std::vector<double>& samplingDist, std::vector<double>& sampleWeights, const char * varName = nullptr);


    SamplingDistribution(const char *name,const char *title, const char * varName = nullptr);

    SamplingDistribution(const char *name,const char *title, RooDataSet& dataSet, const char * columnName = nullptr, const char * varName = nullptr);

    /// Default constructor for SamplingDistribution
    SamplingDistribution();

    /// Destructor of SamplingDistribution
    ~SamplingDistribution() override;

    /// get the inverse of the Cumulative distribution function
    double InverseCDF(double pvalue);

    /// get the inverse of the Cumulative distribution function
    double InverseCDFInterpolate(double pvalue);

    /// get the inverse of the Cumulative distribution function
    /// together with the inverse based on sampling variation
    double InverseCDF(double pvalue, double sigmaVariaton, double& inverseVariation);

    /// merge two sampling distributions
    void Add(const SamplingDistribution* other);

    /// size of samples
    Int_t GetSize() const{return fSamplingDist.size();}

    /// Get test statistics values
    const std::vector<double> & GetSamplingDistribution() const {return fSamplingDist;}
    /// Get the sampling weights
    const std::vector<double> & GetSampleWeights() const {return fSampleWeights;}

    const TString GetVarName() const {return fVarName;}

    /// numerical integral in these limits
    double Integral(double low, double high, bool normalize = true, bool lowClosed = true, bool highClosed = false) const;

    /// numerical integral in these limits including error estimation
    double IntegralAndError(double & error, double low, double high, bool normalize = true,
                              bool lowClosed = true, bool highClosed = false) const;

    /// calculate CDF as a special case of Integral(...) with lower limit equal to -inf
    double CDF(double x) const;

  private:

    mutable std::vector<double> fSamplingDist;  ///< vector of points for the sampling distribution
    mutable std::vector<double> fSampleWeights; ///< vector of weights for the samples

    TString fVarName;

    mutable std::vector<double> fSumW;   ///<! Cached vector with sum of the weight used to compute integral
    mutable std::vector<double> fSumW2;  ///<! Cached vector with sum of the weight used to compute integral error

  protected:

    /// internal function to sort values
    void SortValues() const;

    ClassDefOverride(SamplingDistribution,2)  /// Class containing the results of the HybridCalculator
  };
}

#endif
