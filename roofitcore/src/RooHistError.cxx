/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.cc,v 1.6 2001/11/17 01:44:51 david Exp $
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
"$Id: RooHistError.cc,v 1.6 2001/11/17 01:44:51 david Exp $";

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
  PoissonSum sum(n);

  return getInterval(sum,(Double_t)n,1.0,mu1,mu2,nSigma);
}

Bool_t RooHistError::getBinomialInterval(Int_t n, Int_t m,
					 Double_t &asym1, Double_t &asym2, Double_t nSigma) const
{
  // sanity checks
  if(n < 0 || m < 0) {
    cout << "RooHistError::getPoissonInterval: cannot calculate interval for n,m = " << n << "," << m << endl;
    return kFALSE;
  }

  // swap n and m to ensure than n <= m
  Bool_t swapped(kFALSE);
  if(0 && n > m) {
    swapped= kTRUE;
    Int_t tmp(m);
    m= n;
    n= tmp;
  }

  // create a function object to use
  BinomialSum sum(n,m);

  Bool_t status= getInterval(sum,(Double_t)(n-m)/(n+m),0.1,asym1,asym2,nSigma);

  // undo the swap here
  if(swapped) {
    Double_t tmp(asym1);
    asym1= -asym2;
    asym2= -tmp;
  }
  return status;
}

Bool_t RooHistError::getInterval(const RooAbsFunc &Q, Double_t pointEstimate, Double_t stepSize,
				 Double_t &lo, Double_t &hi, Double_t nSigma) const
{
  // Calculate a confidence interval using the cummulative function provided.

  // convert number of sigma into a confidence level
  Double_t beta= TMath::Erf(nSigma/sqrt(2));
  Double_t alpha= 0.5*(1-beta);

  // Does the central interval contain the point estimate?
  RooBrentRootFinder finder(Q);
  Double_t Q0= Q(&pointEstimate);
  Bool_t ok(kTRUE);
  if(Q0 > alpha + beta) {
    // Force the low edge to be at the pointEstimate
    lo= pointEstimate;
    hi= seek(Q,lo,+stepSize,Q0-beta);
    ok= finder.findRoot(hi,hi-stepSize,hi,Q0-beta);
  }
  else if(Q0 < alpha) {
    // Force the high edge to be at pointEstimate
    hi= pointEstimate;
    lo= seek(Q,hi,-stepSize,Q0+beta);
    ok= finder.findRoot(lo,lo,lo+stepSize,Q0+beta);
  }
  else {
    // use a central interval
    lo= seek(Q,pointEstimate,-stepSize,alpha+beta);
    hi= seek(Q,pointEstimate,+stepSize,alpha);
    ok= finder.findRoot(lo,lo,lo+stepSize,alpha+beta);
    ok|= finder.findRoot(hi,hi-stepSize,hi,alpha);
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
  while(steps-- && (f0*(f(&x)-value) > 0) && ((x-min)*(max-x) > 0));
  assert(0 != steps);
  if(x < min) x= min;
  if(x > max) x= max;

  return x;
}
