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

/** \class RooStats::SamplingDistribution
    \ingroup Roostats

This class simply holds a sampling distribution of some test statistic.
The distribution can either be an empirical distribution (eg. the samples themselves) or
a weighted set of points (eg. for the FFT method).
The class supports merging.
*/

#include "RooMsgService.h"

#include "RooStats/SamplingDistribution.h"
#include "RooNumber.h"
#include "TMath.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
using namespace std ;

ClassImp(RooStats::SamplingDistribution);

using namespace RooStats;

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistribution constructor

SamplingDistribution::SamplingDistribution( const char *name, const char *title,
                   std::vector<double>& samplingDist, const char * varName) :
  TNamed(name,title)
{
  fSamplingDist = samplingDist;
  // need to check STL stuff here.  Will this = operator work as wanted, or do we need:
  //  std::copy(samplingDist.begin(), samplingDist.end(), fSamplingDist.begin());

  // WVE must fill sampleWeights vector here otherwise append behavior potentially undefined
  fSampleWeights.resize(fSamplingDist.size(),1.0) ;

  fVarName = varName;
}

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistribution constructor

SamplingDistribution::SamplingDistribution( const char *name, const char *title,
                   std::vector<double>& samplingDist, std::vector<double>& sampleWeights, const char * varName) :
  TNamed(name,title)
{
  fSamplingDist = samplingDist;
  fSampleWeights = sampleWeights;
  // need to check STL stuff here.  Will this = operator work as wanted, or do we need:
  //  std::copy(samplingDist.begin(), samplingDist.end(), fSamplingDist.begin());

  fVarName = varName;
}

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistribution constructor (with name and title)

SamplingDistribution::SamplingDistribution( const char *name, const char *title, const char * varName) :
  TNamed(name,title)
{
  fVarName = varName;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a SamplingDistribution from a RooDataSet for debugging
/// purposes; e.g. if you need a Gaussian type SamplingDistribution
/// you can generate it from a Gaussian pdf and use the resulting
/// RooDataSet with this constructor.
///
/// The result is the projected distribution onto varName
/// marginalizing the other variables.
///
/// If varName is not given, the first variable will be used.
/// This is useful mostly for RooDataSets with only one observable.

SamplingDistribution::SamplingDistribution(
   const char *name,
   const char *title,
   RooDataSet& dataSet,
   const char * _columnName,
   const char * varName
) : TNamed(name, title) {


   // check there are any meaningful entries in the given dataset
   if( dataSet.numEntries() == 0  ||  !dataSet.get()->first() ) {
      if( varName ) fVarName = varName;
      return;
   }

   TString columnName( _columnName );

   if( !columnName.Length() ) {
      columnName.Form( "%s_TS0", name );
      if( !dataSet.get()->find(columnName) ) {
         columnName = dataSet.get()->first()->GetName();
      }
   }

   if( !varName ) {
      // no leak. none of these transfers ownership.
      fVarName = (*dataSet.get())[columnName].GetTitle();
   }else{
      fVarName = varName;
   }

   for(Int_t i=0; i < dataSet.numEntries(); i++) {
      fSamplingDist.push_back(dataSet.get(i)->getRealValue(columnName));
      fSampleWeights.push_back(dataSet.weight());
   }
}


////////////////////////////////////////////////////////////////////////////////
/// SamplingDistribution default constructor

SamplingDistribution::SamplingDistribution( ) :
  TNamed("SamplingDistribution_DefaultName","SamplingDistribution")
{
}

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistribution destructor

SamplingDistribution::~SamplingDistribution()
{
   fSamplingDist.clear();
   fSampleWeights.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Merge SamplingDistributions (does nothing if nullptr is given).
/// If variable name was not set before, it is copied from the added
/// SamplingDistribution.

void SamplingDistribution::Add(const SamplingDistribution* other)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns the integral in the open/closed/mixed interval. Default is [low,high) interval.
/// Normalization can be turned off.

double SamplingDistribution::Integral(double low, double high, bool normalize, bool lowClosed, bool
                                        highClosed) const
{
   double error = 0;
   return IntegralAndError(error, low,high, normalize, lowClosed, highClosed);
}

////////////////////////////////////////////////////////////////////////////////
/// first need to sort the values and then compute the
/// running sum of the weights and of the weight square
/// needed later for computing the integral

void SamplingDistribution::SortValues() const {

   unsigned int n = fSamplingDist.size();
   std::vector<unsigned int> index(n);
   TMath::SortItr(fSamplingDist.begin(), fSamplingDist.end(), index.begin(), false );

   // compute the empirical CDF and cache in a vector
   fSumW = std::vector<double>( n );
   fSumW2 = std::vector<double>( n );

   std::vector<double> sortedDist( n);
   std::vector<double> sortedWeights( n);

   for(unsigned int i=0; i <n; i++) {
      unsigned int j = index[i];
      if (i > 0) {
         fSumW[i] += fSumW[i-1];
         fSumW2[i] += fSumW2[i-1];
      }
      fSumW[i] += fSampleWeights[j];
      fSumW2[i] += fSampleWeights[j]*fSampleWeights[j];
      // sort also the sampling distribution and the weights
      sortedDist[i] = fSamplingDist[ j] ;
      sortedWeights[i] = fSampleWeights[ j] ;
   }

   // save the sorted distribution
   fSamplingDist = sortedDist;
   fSampleWeights = sortedWeights;


}

////////////////////////////////////////////////////////////////////////////////
/// Returns the integral in the open/closed/mixed interval. Default is [low,high) interval.
/// Normalization can be turned off.
/// compute also the error on the integral

double SamplingDistribution::IntegralAndError(double & error, double low, double high, bool normalize, bool lowClosed, bool
                                                highClosed) const
{
   int n = fSamplingDist.size();
   if( n == 0 ) {
      error = numeric_limits<double>::infinity();
      return 0.0;
   }

   if (int(fSumW.size()) != n)
      SortValues();


   // use std::upper_bounds returns lower index value
   int indexLow = -1;
   int indexHigh = -1;
   if (lowClosed)  {
      // case of closed intervals want to include lower part
      indexLow = std::lower_bound( fSamplingDist.begin(), fSamplingDist.end() , low) - fSamplingDist.begin() -1;
   }
   else {
      // case of open intervals
      indexLow = std::upper_bound( fSamplingDist.begin(), fSamplingDist.end() , low) - fSamplingDist.begin() - 1;
   }


   if (highClosed) {
      indexHigh = std::upper_bound( fSamplingDist.begin(), fSamplingDist.end() , high) - fSamplingDist.begin() -1;
   }
   else {
      indexHigh = std::lower_bound( fSamplingDist.begin(), fSamplingDist.end() , high) - fSamplingDist.begin() -1;

   }


   assert(indexLow < n && indexHigh < n);

   double sum = 0;
   double sum2 = 0;

   if (indexHigh >= 0) {
      sum  = fSumW[indexHigh];
      sum2  = fSumW2[indexHigh];

      if (indexLow >= 0) {
         sum -= fSumW[indexLow];
         sum2 -= fSumW2[indexLow];
      }
   }

   if(normalize) {

      double norm  = fSumW.back();
      double norm2 = fSumW2.back();

      sum /= norm;

      // use formula for binomial error in case of weighted events
      // expression can be derived using a MLE for a weighted binomial likelihood
      error = std::sqrt( sum2 * (1. - 2. * sum) + norm2 * sum * sum ) / norm;
   }
   else {
      error = std::sqrt(sum2);
   }


   return sum;
}

////////////////////////////////////////////////////////////////////////////////
/// returns the closed integral [-inf,x]

double SamplingDistribution::CDF(double x) const {
   return Integral(-RooNumber::infinity(), x, true, true, true);
}

////////////////////////////////////////////////////////////////////////////////
/// returns the inverse of the cumulative distribution function

double SamplingDistribution::InverseCDF(double pvalue)
{
  double dummy=0;
  return InverseCDF(pvalue,0,dummy);
}

////////////////////////////////////////////////////////////////////////////////
/// returns the inverse of the cumulative distribution function, with variations depending on number of samples

double SamplingDistribution::InverseCDF(double pvalue,
                 double sigmaVariation,
                 double& inverseWithVariation)
{
   if (fSumW.size() != fSamplingDist.size())
      SortValues();

   if (!TMath::AreEqualRel(fSumW.back(), fSumW2.back(), 1.E-6) )
      Warning("InverseCDF","Estimation of Quantiles (InverseCDF) for weighted events is not yet supported");


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


////////////////////////////////////////////////////////////////////////////////
/// returns the inverse of the cumulative distribution function

double SamplingDistribution::InverseCDFInterpolate(double pvalue)
{
   if (fSumW.size() != fSamplingDist.size())
      SortValues();

   if (!TMath::AreEqualRel(fSumW.back(), fSumW2.back(), 1.E-6) )
      Warning("InverseCDFInterpolate","Estimation of Quantiles (InverseCDF) for weighted events is not yet supported.");

  // casting will round down, eg. give i
  int nominal = (unsigned int) (pvalue*fSamplingDist.size());

  if(nominal <= 0) {
    return -1.*RooNumber::infinity();
  }
  if(nominal >= (Int_t)fSamplingDist.size()-1 ) {
    return RooNumber::infinity();
  }
  double upperX = fSamplingDist[nominal+1];
  double upperY = ((double) (nominal+1))/fSamplingDist.size();
  double lowerX =  fSamplingDist[nominal];
  double lowerY = ((double) nominal)/fSamplingDist.size();

  //  std::cout << upperX << " " << upperY << " " << lowerX << " " << lowerY << std::endl;

  return (upperX-lowerX)/(upperY-lowerY)*(pvalue-lowerY)+lowerX;

}
