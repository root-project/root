/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHistError.h,v 1.14 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_HIST_ERROR
#define ROO_HIST_ERROR

#include "Rtypes.h"
#include "RooNumber.h"
#include "RooAbsFunc.h"
#include <cmath>
#include <iostream>

class RooHistError {
public:
  static const RooHistError &instance();
  virtual ~RooHistError() {} ;

  bool getPoissonInterval(Int_t n, double &mu1, double &mu2, double nSigma= 1) const;
  bool getBinomialIntervalAsym(Int_t n, Int_t m, double &a1, double &a2, double nSigma= 1) const;
  bool getBinomialIntervalEff(Int_t n, Int_t m, double &a1, double &a2, double nSigma= 1) const;
  bool getInterval(const RooAbsFunc *Qu, const RooAbsFunc *Ql, double pointEstimate, double stepSize,
           double &lo, double &hi, double nSigma) const;

  static RooAbsFunc *createPoissonSum(Int_t n) ;
  static RooAbsFunc *createBinomialSum(Int_t n, Int_t m, bool eff) ;

private:


  bool getPoissonIntervalCalc(Int_t n, double &mu1, double &mu2, double nSigma= 1) const;
  double _poissonLoLUT[1000] ;
  double _poissonHiLUT[1000] ;

  RooHistError();
  double seek(const RooAbsFunc &f, double startAt, double step, double value) const;

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
    inline double operator()(const double xvec[]) const override {
      double mu(xvec[0]),result(1),factorial(1);
      for(Int_t k= 1; k <= _n; k++) {
   factorial*= k;
   result+= pow(mu,k)/factorial;
      }
      return exp(-mu)*result;
    };
    inline double getMinLimit(UInt_t /*index*/) const override { return 0; }
    inline double getMaxLimit(UInt_t /*index*/) const override { return RooNumber::infinity() ; }
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
  class BinomialSumAsym : public RooAbsFunc {
  public:
    BinomialSumAsym(Int_t n, Int_t m) : RooAbsFunc(1), _n1(n), _N1(n+m) {
    }
    inline double operator()(const double xvec[]) const override
      {
   double p1(0.5*(1+xvec[0])),p2(1-p1),result(0),fact1(1),fact2(1);
   for(Int_t k= 0; k <= _n1; k++) {
     if(k > 0) { fact2*= k; fact1*= _N1-k+1; }
     result+= fact1/fact2*pow(p1,k)*pow(p2,_N1-k);
   }
   return result;
      };

    inline double getMinLimit(UInt_t /*index*/) const override { return -1; }
    inline double getMaxLimit(UInt_t /*index*/) const override { return +1; }

  private:
    Int_t _n1 ; ///< WVE Solaris CC5 doesn't want _n or _N here (likely compiler bug)
    Int_t _N1 ;
  } ;


  // -----------------------------------------------------------
  // Define a 1-dim RooAbsFunc of a that evaluates the sum:
  //
  //  Q(n|n+m,a) = Sum_{k=0}^{n} B(k|n+m,a)
  //
  // where B(n|n+m,a) = (n+m)!/(n!m!) ((1+a)/2)**n ((1-a)/2)**m
  // is the Binomial PDF.
  // -----------------------------------------------------------
  class BinomialSumEff : public RooAbsFunc {
  public:
    BinomialSumEff(Int_t n, Int_t m) : RooAbsFunc(1), _n1(n), _N1(n+m) {
    }
    inline double operator()(const double xvec[]) const override
      {
   double p1(xvec[0]),p2(1-p1),result(0),fact1(1),fact2(1);
   for(Int_t k= 0; k <= _n1; k++) {
     if(k > 0) { fact2*= k; fact1*= _N1-k+1; }
     result+= fact1/fact2*pow(p1,k)*pow(p2,_N1-k);
   }
   return result;
      };

    inline double getMinLimit(UInt_t /*index*/) const override { return  0; }
    inline double getMaxLimit(UInt_t /*index*/) const override { return +1; }

  private:
    Int_t _n1 ; ///< WVE Solaris CC5 doesn't want _n or _N here (likely compiler bug)
    Int_t _N1 ;
  } ;

  ClassDef(RooHistError,1) // Utility class for calculating histogram errors
};

#endif
