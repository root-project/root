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

//____________________________________________________________________
/*
SamplingDistribution : 

This class simply holds a sampling distribution of some test statistic.  
The distribution can either be an empirical distribution (eg. the samples themselves) or
a weighted set of points (eg. for the FFT method).
The class supports merging.
*/

#include "RooStats/SamplingDistribution.h"
#include "RooNumber.h"
#include "math.h"
#include <algorithm>
#include <iostream>
using namespace std ;

/// ClassImp for building the THtml documentation of the class 
ClassImp(RooStats::SamplingDistribution)

using namespace RooStats;

//_______________________________________________________
SamplingDistribution::SamplingDistribution( const char *name, const char *title,
					    std::vector<Double_t>& samplingDist, const char * varName) :
  TNamed(name,title)
{
  // SamplingDistribution constructor
  fSamplingDist = samplingDist;
  // need to check STL stuff here.  Will this = operator work as wanted, or do we need:
  //  std::copy(samplingDist.begin(), samplingDist.end(), fSamplingDist.begin());

  // WVE must fill sampleWeights vector here otherwise append behavior potentially undefined
  fSampleWeights.resize(fSamplingDist.size(),1.0) ;  

  fVarName = varName;
}

//_______________________________________________________
SamplingDistribution::SamplingDistribution( const char *name, const char *title,
					    std::vector<Double_t>& samplingDist, std::vector<Double_t>& sampleWeights, const char * varName) :
  TNamed(name,title)
{
  // SamplingDistribution constructor
  fSamplingDist = samplingDist;
  fSampleWeights = sampleWeights;
  // need to check STL stuff here.  Will this = operator work as wanted, or do we need:
  //  std::copy(samplingDist.begin(), samplingDist.end(), fSamplingDist.begin());

  fVarName = varName;
}

//_______________________________________________________
SamplingDistribution::SamplingDistribution( const char *name, const char *title, const char * varName) :
  TNamed(name,title)
{
   // SamplingDistribution constructor (with name and title)
  fVarName = varName;
}


SamplingDistribution::SamplingDistribution(
   const char *name,
   const char *title,
   RooDataSet& dataSet,
   const char * varName
) : TNamed(name, title) {
   // Creates a SamplingDistribution from a RooDataSet for debugging
   // purposes; e.g. if you need a Gaussian type SamplingDistribution
   // you can generate it from a Gaussian pdf and use the resulting
   // RooDataSet with this constructor.
   //
   // The result is the projected distribution onto varName
   // marginalizing the other variables.
   //
   // If varName is not given, the first variable will be used.
   // This is useful mostly for RooDataSets with only one observable.

   fVarName = varName;
   if(fVarName.Length() == 0) {
      // no leak. none of these transfers ownership.
      fVarName = dataSet.get()->first()->GetName();
   }

   for(Int_t i=0; i < dataSet.numEntries(); i++) {
      fSamplingDist.push_back(dataSet.get(i)->getRealValue(fVarName));
      fSampleWeights.push_back(dataSet.weight());
   }
}


//_______________________________________________________
SamplingDistribution::SamplingDistribution( ) :
  TNamed("SamplingDistribution_DefaultName","SamplingDistribution")
{
   // SamplingDistribution default constructor
}

//_______________________________________________________
SamplingDistribution::~SamplingDistribution()
{
   // SamplingDistribution destructor

   fSamplingDist.clear();
   fSampleWeights.clear();
}


//_______________________________________________________
void SamplingDistribution::Add(const SamplingDistribution* other)
{
   // Merge SamplingDistributions (does nothing if NULL is given).
   // If variable name was not set before, it is copied from the added
   // SamplingDistribution.

   if(!other) return;

  std::vector<double> newSamplingDist = other->fSamplingDist;
  std::vector<double> newSampleWeights = other->fSampleWeights;
  // need to check STL stuff here.  Will this = operator work as wanted, or do we need:
  //  std::copy(samplingDist.begin(), samplingDist.end(), fSamplingDist.begin());
  // need to look into STL, do it the easy way for now

  // reserve memory
  fSamplingDist.reserve(fSamplingDist.size()+newSamplingDist.size());
  fSampleWeights.reserve(fSampleWeights.size()+newSampleWeights.size());

  // push back elements
  for(unsigned int i=0; i<newSamplingDist.size(); ++i){
    fSamplingDist.push_back(newSamplingDist[i]);
    fSampleWeights.push_back(newSampleWeights[i]);
  }


  if(GetVarName().Length() == 0  &&  other->GetVarName().Length() > 0)
     fVarName = other->GetVarName();

  if(strlen(GetName()) == 0  &&  strlen(other->GetName()) > 0)
     SetName(other->GetName());
  if(strlen(GetTitle()) == 0  &&  strlen(other->GetTitle()) > 0)
     SetTitle(other->GetTitle());

}



//_______________________________________________________
Double_t SamplingDistribution::Integral(Double_t low, Double_t high, Bool_t normalize, Bool_t lowClosed, Bool_t highClosed) const
{
   // Returns the integral in the open/closed/mixed interval. Default is [low,high) interval.
   // Normalization can be turned off.

   Double_t sum = 0;
   for(unsigned int i=0; i<fSamplingDist.size(); i++) {
      double value = fSamplingDist[i];

      if((lowClosed  ? value >= low  : value > low)  &&
         (highClosed ? value <= high : value < high))
      {
         sum += fSampleWeights[i];
      }
   }

   if(normalize) {
      Double_t norm = 0;
      for(unsigned int i=0; i<fSamplingDist.size(); i++) {
         norm += fSampleWeights[i];
      }
      sum /= norm;
   }

   return sum;
}

//_______________________________________________________
Double_t SamplingDistribution::CDF(Double_t x) const {
   // returns the closed integral [-inf,x]
   return Integral(-RooNumber::infinity(), x, kTRUE, kTRUE, kTRUE);
}



//_______________________________________________________
Double_t SamplingDistribution::InverseCDF(Double_t pvalue)
{
   // returns the inverse of the cumulative distribution function

  Double_t dummy=0;
  return InverseCDF(pvalue,0,dummy);
}
//_______________________________________________________
Double_t SamplingDistribution::InverseCDF(Double_t pvalue, 
					  Double_t sigmaVariation, 
					  Double_t& inverseWithVariation)
{
   // returns the inverse of the cumulative distribution function, with variations depending on number of samples

  // will need to deal with weights, but for now:
  std::sort(fSamplingDist.begin(), fSamplingDist.end());


  // Acceptance regions are meant to be inclusive of (1-\alpha) of the probability
  // so the returned values of the CDF should make this easy.
  // in particular:
  //   if finding the critical value for a lower bound
  //     when p_i < p < p_j, one should return the value associated with i
  //     if i=0, then one should return -infinity
  //   if finding the critical value for an upper bound
  //     when p_i < p < p_j, one should return the value associated with j
  //     if i = size-1, then one should return +infinity
  //   use pvalue < 0.5 to indicate a lower bound is requested
  
  // casting will round down, eg. give i
  int nominal = (unsigned int) (pvalue*fSamplingDist.size());

  if(nominal <= 0) {
    inverseWithVariation = -1.*RooNumber::infinity();
    return -1.*RooNumber::infinity();
  }
  else if(nominal >= (Int_t)fSamplingDist.size()-1 ) {
    inverseWithVariation = RooNumber::infinity();
    return RooNumber::infinity();
  }
  else if(pvalue < 0.5){
    int delta = (int)(sigmaVariation*sqrt(1.0*nominal)); // note sqrt(small fraction)
    int variation = nominal+delta;

    if(variation>=(Int_t)fSamplingDist.size()-1)
      inverseWithVariation = RooNumber::infinity();
    else if(variation<=0)
      inverseWithVariation = -1.*RooNumber::infinity();
    else 
      inverseWithVariation =  fSamplingDist[ variation ];

    return fSamplingDist[nominal];
  }
  else if(pvalue >= 0.5){
    int delta = (int)(sigmaVariation*sqrt(1.0*fSamplingDist.size()- nominal)); // note sqrt(small fraction)
    int variation = nominal+delta;


    if(variation>=(Int_t)fSamplingDist.size()-1)
      inverseWithVariation = RooNumber::infinity();

    else if(variation<=0)
      inverseWithVariation = -1.*RooNumber::infinity();
    else 
      inverseWithVariation =  fSamplingDist[ variation+1 ];


    /*
      std::cout << "dgb SamplingDistribution::InverseCDF. variation = " << variation
      << " size = " << fSamplingDist.size()
      << " value = " << inverseWithVariation << std::endl;
    */

    return fSamplingDist[nominal+1];
  }
  else{
    std::cout << "problem in SamplingDistribution::InverseCDF" << std::endl;
  }
  inverseWithVariation = RooNumber::infinity();
  return RooNumber::infinity();

}


//_______________________________________________________
Double_t SamplingDistribution::InverseCDFInterpolate(Double_t pvalue)
{
   // returns the inverse of the cumulative distribution function

  // will need to deal with weights, but for now:
  std::sort(fSamplingDist.begin(), fSamplingDist.end());

  // casting will round down, eg. give i
  int nominal = (unsigned int) (pvalue*fSamplingDist.size());

  if(nominal <= 0) {
    return -1.*RooNumber::infinity();
  }
  if(nominal >= (Int_t)fSamplingDist.size()-1 ) {
    return RooNumber::infinity();
  }
  Double_t upperX = fSamplingDist[nominal+1];
  Double_t upperY = ((Double_t) (nominal+1))/fSamplingDist.size();
  Double_t lowerX =  fSamplingDist[nominal];
  Double_t lowerY = ((Double_t) nominal)/fSamplingDist.size();
  
  //  std::cout << upperX << " " << upperY << " " << lowerX << " " << lowerY << std::endl;

  return (upperX-lowerX)/(upperY-lowerY)*(pvalue-lowerY)+lowerX;

}
