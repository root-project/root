/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.rdl,v 1.3 2001/11/15 01:49:33 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 2000 Stanford University
 *****************************************************************************/
#ifndef ROO_HIST_ERROR
#define ROO_HIST_ERROR

#include "Rtypes.h"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooAbsFunc.hh"
#include <math.h>

class RooHistError {
public:
  static const RooHistError &instance();

  Bool_t getPoissonInterval(Int_t n, Double_t &mu1, Double_t &mu2, Double_t nSigma= 1) const;
  Bool_t getBinomialInterval(Int_t n, Int_t m, Double_t &a1, Double_t &a2, Double_t nSigma= 1) const;
  Bool_t getInterval(const RooAbsFunc &Q, Double_t pointEstimate, Double_t stepSize,
		     Double_t &lo, Double_t &hi, Double_t nSigma) const;
  
  inline static RooAbsFunc *createPoissonSum(Int_t n) { return new PoissonSum(n); }
  inline static RooAbsFunc *createBinomialSum(Int_t n, Int_t m) { return new BinomialSum(n,m); }

private:
  RooHistError();
  Double_t seek(const RooAbsFunc &f, Double_t startAt, Double_t step, Double_t value) const;

  // -----------------------------------------------------------
  // Define a 1-dim RooAbsFunc of mu that evaluates the sum:
  //
  //  Q(n|mu) = Sum_{k=0}^{n} P(k|mu)
  //
  // where P(n|mu) = exp(-mu) mu**n / n! is the Poisson PDF.
  // -----------------------------------------------------------
  class PoissonSum : public RooAbsFunc {
  public:
    inline PoissonSum(Int_t n) : RooAbsFunc(1), _n(n) { }
    inline Double_t operator()(const Double_t xvec[]) const {
      Double_t mu(xvec[0]),result(1),factorial(1);
      for(Int_t k= 1; k <= _n; k++) {
	factorial*= k;
	result+= pow(mu,k)/factorial;
      }
      return exp(-mu)*result;
    };
    inline Double_t getMinLimit(UInt_t index) const { return 0; }
    inline Double_t getMaxLimit(UInt_t index) const { return RooNumber::infinity; }
  private:
    Int_t _n;
  };

  // -----------------------------------------------------------
  // Define a 1-dim RooAbsFunc of a that evaluates the sum:
  //
  //  Q(n|n+m,a) = Sum_{k=0}^{n} B(k|n+m,a)
  //
  // where B(n|n+m,a) = (n+m)!/(n!m!) ((1+a)/2)**n ((1-a)/2)**m
  // is the Binomial PDF.
  // -----------------------------------------------------------
  class BinomialSum : public RooAbsFunc {
  public:
    inline BinomialSum(Int_t n, Int_t m) : RooAbsFunc(1), _n(n), _N(n+m) { }
    inline Double_t operator()(const Double_t xvec[]) const {
      Double_t p1(0.5*(1+xvec[0])),p2(1-p1),result(0),fact1(1),fact2(1);
      for(Int_t k= 0; k <= _n; k++) {
	if(k > 0) { fact2*= k; fact1*= _N-k+1; }
	result+= fact1/fact2*pow(p1,k)*pow(p2,_N-k);
      }
      return result;
    };
    inline Double_t getMinLimit(UInt_t index) const { return -1; }
    inline Double_t getMaxLimit(UInt_t index) const { return +1; }
  private:
    Int_t _n,_N;
  };

  ClassDef(RooHistError,1) // Utility class for calculating histogram errors
};

#endif
