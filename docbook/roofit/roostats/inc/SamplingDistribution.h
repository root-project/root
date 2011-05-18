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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#include "Rtypes.h"
#include "RooDataSet.h"


#include <vector>


namespace RooStats {

 class SamplingDistribution : public TNamed {

   public:

    // Constructor for SamplingDistribution
    SamplingDistribution(const char *name,const char *title, std::vector<Double_t>& samplingDist, const char * varName = 0);
    SamplingDistribution(const char *name,const char *title,
			 std::vector<Double_t>& samplingDist, std::vector<Double_t>& sampleWeights, const char * varName = 0);


    SamplingDistribution(const char *name,const char *title, const char * varName = 0);

    SamplingDistribution(const char *name,const char *title, RooDataSet& dataSet, const char * varName = 0);

    // Default constructor for SamplingDistribution
    SamplingDistribution();
    
    // Destructor of SamplingDistribution
    virtual ~SamplingDistribution();
    
    // get the inverse of the Cumulative distribution function
    Double_t InverseCDF(Double_t pvalue);

    // get the inverse of the Cumulative distribution function
    Double_t InverseCDFInterpolate(Double_t pvalue);

    // get the inverse of the Cumulative distribution function
    // together with the inverse based on sampling variation
    Double_t InverseCDF(Double_t pvalue, Double_t sigmaVariaton, Double_t& inverseVariation);
  
    // merge two sampling distributions
    void Add(const SamplingDistribution* other);
    
    // size of samples
    Int_t GetSize() const{return fSamplingDist.size();}

    // Get test statistics values
    const std::vector<Double_t> & GetSamplingDistribution() const {return fSamplingDist;}
    // Get the sampling weights 
    const std::vector<Double_t> & GetSampleWeights() const {return fSampleWeights;}

    const TString GetVarName() const {return fVarName;}
    
    // numerical integral in these limits
    Double_t Integral(Double_t low, Double_t high, Bool_t normalize = kTRUE, Bool_t lowClosed = kTRUE, Bool_t highClosed = kFALSE) const;

    // calculate CDF as a special case of Integral(...) with lower limit equal to -inf
    Double_t CDF(Double_t x) const;

  private:
    std::vector<Double_t> fSamplingDist; // vector of points for the sampling distribution
    std::vector<Double_t> fSampleWeights; // vector of weights for the samples
    // store a RooRealVar that this distribution corresponds to?

    TString fVarName;
    
  protected:
    
    ClassDef(SamplingDistribution,1)  // Class containing the results of the HybridCalculator
  };
}

#endif
