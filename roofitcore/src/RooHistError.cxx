/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.cc,v 1.8 2001/11/28 01:20:45 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   27-Apr-2001 DK Created initial version from RooMath
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PLOT] --
// RooHistError is a singleton class used to calculate the error bars
// for each bin of a RooHist object. Errors are calculated by integrating
// a specified area of a Poisson or Binomail error distribution.

#include "RooFitCore/RooHistError.hh"
#include "RooFitCore/RooBrentRootFinder.hh"
#include "TMath.h"

#include <iostream.h>
#include <assert.h>

ClassImp(RooHistError)
  ;

static const char rcsid[] =
"$Id: RooHistError.cc,v 1.8 2001/11/28 01:20:45 david Exp $";

const RooHistError &RooHistError::instance() {
  // Return a reference to a singleton object that is created the
  // first time this method is called. Only one object will be
  // constructed per ROOT session.

  static RooHistError _theInstance;
  return _theInstance;
}

RooHistError::RooHistError() {
  // Construct our singleton object.

}

Bool_t RooHistError::getPoissonInterval(Int_t n, Double_t &mu1, Double_t &mu2, Double_t nSigma) const
{
  // Calculate a confidence interval for the expected number of events given n
  // observed (unweighted) events. The interval will contain the same probability
  // as nSigma of a Gaussian. Uses a central interval unless this does not enclose
  // the point estimate n (ie, for small n) in which case the interval is adjusted
  // to start at n.

  // sanity checks
  if(n < 0) {
    cout << "RooHistError::getPoissonInterval: cannot calculate interval for n = " << n << endl;
    return kFALSE;
  }

  // use assymptotic error if possible
  if(n > 100) {
    mu1= n - sqrt(n+0.25) + 0.5;
    mu2= n + sqrt(n+0.25) + 0.5;
    return kTRUE;
  }

  // create a function object to use
  PoissonSum upper(n);
  if(n > 0) {
    PoissonSum lower(n-1);
    return getInterval(&upper,&lower,(Double_t)n,1.0,mu1,mu2,nSigma);
  }
  else {
    return getInterval(&upper,0,(Double_t)n,1.0,mu1,mu2,nSigma);
  }
}

Bool_t RooHistError::getBinomialInterval(Int_t n, Int_t m,
					 Double_t &asym1, Double_t &asym2, Double_t nSigma) const
{
  // sanity checks
  if(n < 0 || m < 0) {
    cout << "RooHistError::getPoissonInterval: cannot calculate interval for n,m = " << n << "," << m << endl;
    return kFALSE;
  }

  // handle the special case of no events in either category
  if(n == 0 && m == 0) {
    asym1= -1;
    asym2= +1;
    return kTRUE;
  }

  // handle cases when n,m>150 (factorials in BinomialSum will overflow around 170)
  if (n>150&&m>150) {
    Double_t asym = 1.0*(n-m)/(n+m) ;
    Double_t approxErr = sqrt(4.0*n/(n+m)*(1-n/(n+m))/(n+m)) ;
    asym1 = asym-nSigma*approxErr ;
    asym2 = asym+nSigma*approxErr ;
    return kTRUE ;
  }

  // swap n and m to ensure that n <= m
  Bool_t swapped(kFALSE);
  if(n > m) {
    swapped= kTRUE;
    Int_t tmp(m);
    m= n;
    n= tmp;
  }

  // create the function objects to use
  Bool_t status(kFALSE);
  BinomialSum upper(n,m);
  if(n > 0) {
    BinomialSum lower(n-1,m+1);
    status= getInterval(&upper,&lower,(Double_t)(n-m)/(n+m),0.1,asym1,asym2,nSigma);
  }
  else {
    status= getInterval(&upper,0,(Double_t)(n-m)/(n+m),0.1,asym1,asym2,nSigma);
  }

  // undo the swap here
  if(swapped) {
    Double_t tmp(asym1);
    asym1= -asym2;
    asym2= -tmp;
  }

  return status;
}

Bool_t RooHistError::getInterval(const RooAbsFunc *Qu, const RooAbsFunc *Ql, Double_t pointEstimate,
				 Double_t stepSize, Double_t &lo, Double_t &hi, Double_t nSigma) const
{
  // Calculate a confidence interval using the cummulative functions provided.
  // The interval will be "central" when both cummulative functions are provided,
  // unless this would exclude the pointEstimate, in which case a one-sided interval
  // pinned at the point estimate is returned instead.

  // sanity checks
  assert(0 != Qu || 0 != Ql);

  // convert number of sigma into a confidence level
  Double_t beta= TMath::Erf(nSigma/sqrt(2));
  Double_t alpha= 0.5*(1-beta);

  // Does the central interval contain the point estimate?
  Bool_t ok(kTRUE);
  Double_t loProb(1),hiProb(0);
  if(0 != Ql) loProb= (*Ql)(&pointEstimate);
  if(0 != Qu) hiProb= (*Qu)(&pointEstimate);

  if(0 == Ql || loProb > alpha + beta)  {
    // Force the low edge to be at the pointEstimate
    lo= pointEstimate;
    Double_t target= loProb - beta;
    hi= seek(*Qu,lo,+stepSize,target);
    RooBrentRootFinder uFinder(*Qu);
    ok= uFinder.findRoot(hi,hi-stepSize,hi,target);
  }
  else if(0 == Qu || hiProb < alpha) {
    // Force the high edge to be at pointEstimate
    hi= pointEstimate;
    Double_t target= hiProb + beta;
    lo= seek(*Ql,hi,-stepSize,target);
    RooBrentRootFinder lFinder(*Ql);
    ok= lFinder.findRoot(lo,lo,lo+stepSize,target);
  }
  else {
    // use a central interval
    lo= seek(*Ql,pointEstimate,-stepSize,alpha+beta);
    hi= seek(*Qu,pointEstimate,+stepSize,alpha);
    RooBrentRootFinder lFinder(*Ql),uFinder(*Qu);
    ok= lFinder.findRoot(lo,lo,lo+stepSize,alpha+beta);
    ok|= uFinder.findRoot(hi,hi-stepSize,hi,alpha);
  }
  if(!ok) cout << "RooHistError::getInterval: failed to find root(s)" << endl;

  return ok;
}

Double_t RooHistError::seek(const RooAbsFunc &f, Double_t startAt, Double_t step, Double_t value) const {
  // Scan f(x)-value until it changes sign. Start at the specified point and take constant
  // steps of the specified size. Give up after 1000 steps.

  Int_t steps(1000);
  Double_t min(f.getMinLimit(1)),max(f.getMaxLimit(1));
  Double_t x(startAt), f0= f(&startAt) - value;
  do {
    x+= step;
  }
  while(steps-- && (f0*(f(&x)-value) >= 0) && ((x-min)*(max-x) >= 0));
  assert(0 != steps);
  if(x < min) x= min;
  if(x > max) x= max;

  return x;
}
