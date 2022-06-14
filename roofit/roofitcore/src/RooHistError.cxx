/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooHistError.cxx
\class RooHistError
\ingroup Roofitcore

RooHistError is a singleton class used to calculate the error bars
for each bin of a RooHist object. Errors are calculated by integrating
a specified area of a Poisson or Binomail error distribution.
**/

#include "RooHistError.h"
#include "RooBrentRootFinder.h"
#include "RooMsgService.h"
#include "TMath.h"

#include "Riostream.h"
#include <assert.h>

using namespace std;

ClassImp(RooHistError);
  ;



////////////////////////////////////////////////////////////////////////////////
/// Return a reference to a singleton object that is created the
/// first time this method is called. Only one object will be
/// constructed per ROOT session.

const RooHistError &RooHistError::instance()
{
  static RooHistError _theInstance;
  return _theInstance;
}


////////////////////////////////////////////////////////////////////////////////
/// Construct our singleton object.

RooHistError::RooHistError()
{
  // Initialize lookup table ;
  Int_t i ;
  for (i=0 ; i<1000 ; i++) {
    getPoissonIntervalCalc(i,_poissonLoLUT[i],_poissonHiLUT[i],1.) ;
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Return a confidence interval for the expected number of events given n
/// observed (unweighted) events. The interval will contain the same probability
/// as nSigma of a Gaussian. Uses a central interval unless this does not enclose
/// the point estimate n (ie, for small n) in which case the interval is adjusted
/// to start at n. This method uses a lookup table to return precalculated results
/// for n<1000

bool RooHistError::getPoissonInterval(Int_t n, double &mu1, double &mu2, double nSigma) const
{
  // Use lookup table for most common cases
  if (n<1000 && nSigma==1.) {
    mu1=_poissonLoLUT[n] ;
    mu2=_poissonHiLUT[n] ;
    return true ;
  }

  // Forward to calculation method
  bool ret =  getPoissonIntervalCalc(n,mu1,mu2,nSigma) ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate a confidence interval for the expected number of events given n
/// observed (unweighted) events. The interval will contain the same probability
/// as nSigma of a Gaussian. Uses a central interval unless this does not enclose
/// the point estimate n (ie, for small n) in which case the interval is adjusted
/// to start at n.

bool RooHistError::getPoissonIntervalCalc(Int_t n, double &mu1, double &mu2, double nSigma) const
{
  // sanity checks
  if(n < 0) {
    oocoutE(nullptr,Plotting) << "RooHistError::getPoissonInterval: cannot calculate interval for n = " << n << endl;
    return false;
  }

  // use assymptotic error if possible
  if(n > 100) {
    mu1= n - sqrt(n+0.25) + 0.5;
    mu2= n + sqrt(n+0.25) + 0.5;
    return true;
  }

  // create a function object to use
  PoissonSum upper(n);
  if(n > 0) {
    PoissonSum lower(n-1);
    return getInterval(&upper,&lower,(double)n,1.0,mu1,mu2,nSigma);
  }

  // Backup solution for negative numbers
  return getInterval(&upper,0,(double)n,1.0,mu1,mu2,nSigma);
}


////////////////////////////////////////////////////////////////////////////////
/// Return 'nSigma' binomial confidence interval for (n,m). The result is return in asym1 and asym2.
/// If the return values is false and error occurred.

bool RooHistError::getBinomialIntervalAsym(Int_t n, Int_t m,
                    double &asym1, double &asym2, double nSigma) const
{
  // sanity checks
  if(n < 0 || m < 0) {
    oocoutE(nullptr,Plotting) << "RooHistError::getPoissonInterval: cannot calculate interval for n,m = " << n << "," << m << endl;
    return false;
  }

  // handle the special case of no events in either category
  if(n == 0 && m == 0) {
    asym1= -1;
    asym2= +1;
    return true;
  }

  // handle cases when n,m>100 (factorials in BinomialSum will overflow around 170)
  if ((n>100&&m>100)) {
    double N = n ;
    double M = m ;
    double asym = 1.0*(N-M)/(N+M) ;
    double approxErr = sqrt(4.0*n/(N+M)*(1-N/(N+M))/(N+M)) ;

    asym1 = asym-nSigma*approxErr ;
    asym2 = asym+nSigma*approxErr ;
    return true ;
  }

  // swap n and m to ensure that n <= m
  bool swapped(false);
  if(n > m) {
    swapped= true;
    Int_t tmp(m);
    m= n;
    n= tmp;
  }

  // create the function objects to use
  bool status(false);
  BinomialSumAsym upper(n,m);
  if(n > 0) {
    BinomialSumAsym lower(n-1,m+1);
    status= getInterval(&upper,&lower,(double)(n-m)/(n+m),0.1,asym1,asym2,nSigma);
  }
  else {
    status= getInterval(&upper,0,(double)(n-m)/(n+m),0.1,asym1,asym2,nSigma);
  }

  // undo the swap here
  if(swapped) {
    double tmp(asym1);
    asym1= -asym2;
    asym2= -tmp;
  }

  return status;
}


////////////////////////////////////////////////////////////////////////////////
/// Return 'nSigma' binomial confidence interval for (n,m). The result is return in asym1 and asym2.
/// If the return values is false and error occurred.

bool RooHistError::getBinomialIntervalEff(Int_t n, Int_t m,
                    double &asym1, double &asym2, double nSigma) const
{
  // sanity checks
  if(n < 0 || m < 0) {
    oocoutE(nullptr,Plotting) << "RooHistError::getPoissonInterval: cannot calculate interval for n,m = " << n << "," << m << endl;
    return false;
  }

  // handle the special case of no events in either category
  if(n == 0 && m == 0) {
    asym1= -1;
    asym2= +1;
    return true;
  }

  // handle cases when n,m>80 (factorials in BinomialSum will overflow around 170)
  if ((n>80&&m>80)) {
    double N = n ;
    double M = m ;
    double asym = 1.0*(N)/(N+M) ;
    double approxErr = sqrt(4.0*n/(N+M)*(1-N/(N+M))/(N+M)) ;

    asym1 = asym-nSigma*0.5*approxErr ;
    asym2 = asym+nSigma*0.5*approxErr ;
    return true ;
  }

  // swap n and m to ensure that n <= m
  bool swapped(false);
  if(n > m) {
    swapped= true;
    Int_t tmp(m);
    m= n;
    n= tmp;
  }

  // create the function objects to use
  bool status(false);
  BinomialSumEff upper(n,m);
  double eff = (double)(n)/(n+m) ;
  if(n > 0) {
    BinomialSumEff lower(n-1,m+1);
    status= getInterval(&upper,&lower,eff,0.1,asym1,asym2,nSigma*0.5);
  }
  else {
    status= getInterval(&upper,0,eff,0.1,asym1,asym2,nSigma*0.5);
  }

  // undo the swap here
  if(swapped) {
    double tmp(asym1);
    asym1= 1-asym2;
    asym2= 1-tmp;
  }

  return status;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate a confidence interval using the cumulative functions provided.
/// The interval will be "central" when both cumulative functions are provided,
/// unless this would exclude the pointEstimate, in which case a one-sided interval
/// pinned at the point estimate is returned instead.

bool RooHistError::getInterval(const RooAbsFunc *Qu, const RooAbsFunc *Ql, double pointEstimate,
             double stepSize, double &lo, double &hi, double nSigma) const
{
  // sanity checks
  assert(0 != Qu || 0 != Ql);

  // convert number of sigma into a confidence level
  double beta= TMath::Erf(nSigma/sqrt(2.));
  double alpha= 0.5*(1-beta);

  // Does the central interval contain the point estimate?
  bool ok(true);
  double loProb(1),hiProb(0);
  if(0 != Ql) loProb= (*Ql)(&pointEstimate);
  if(0 != Qu) hiProb= (*Qu)(&pointEstimate);

  if (Qu && (0 == Ql || loProb > alpha + beta))  {
    // Force the low edge to be at the pointEstimate
    lo= pointEstimate;
    double target= loProb - beta;
    hi= seek(*Qu,lo,+stepSize,target);
    RooBrentRootFinder uFinder(*Qu);
    ok= uFinder.findRoot(hi,hi-stepSize,hi,target);
  }
  else if(Ql && (0 == Qu || hiProb < alpha)) {
    // Force the high edge to be at pointEstimate
    hi= pointEstimate;
    double target= hiProb + beta;
    lo= seek(*Ql,hi,-stepSize,target);
    RooBrentRootFinder lFinder(*Ql);
    ok= lFinder.findRoot(lo,lo,lo+stepSize,target);
  }
  else if (Qu && Ql) {
    // use a central interval
    lo= seek(*Ql,pointEstimate,-stepSize,alpha+beta);
    hi= seek(*Qu,pointEstimate,+stepSize,alpha);
    RooBrentRootFinder lFinder(*Ql),uFinder(*Qu);
    ok= lFinder.findRoot(lo,lo,lo+stepSize,alpha+beta);
    ok|= uFinder.findRoot(hi,hi-stepSize,hi,alpha);
  }
  if(!ok) oocoutE(nullptr,Plotting) << "RooHistError::getInterval: failed to find root(s)" << endl;

  return ok;
}


////////////////////////////////////////////////////////////////////////////////
/// Scan f(x)-value until it changes sign. Start at the specified point and take constant
/// steps of the specified size. Give up after 1000 steps.

double RooHistError::seek(const RooAbsFunc &f, double startAt, double step, double value) const
{
  Int_t steps(1000);
  double min(f.getMinLimit(1)),max(f.getMaxLimit(1));
  double x(startAt), f0= f(&startAt) - value;
  do {
    x+= step;
  }
  while(steps-- && (f0*(f(&x)-value) >= 0) && ((x-min)*(max-x) >= 0));
  assert(0 != steps);
  if(x < min) x= min;
  if(x > max) x= max;

  return x;
}



////////////////////////////////////////////////////////////////////////////////
/// Create and return a PoissonSum function binding

RooAbsFunc *RooHistError::createPoissonSum(Int_t n)
{
  return new PoissonSum(n);
}


////////////////////////////////////////////////////////////////////////////////
/// Create and return a BinomialSum function binding

RooAbsFunc *RooHistError::createBinomialSum(Int_t n, Int_t m, bool eff)
{
  if (eff) {
    return new BinomialSumEff(n,m) ;
  } else {
    return new BinomialSumAsym(n,m) ;
  }
}
