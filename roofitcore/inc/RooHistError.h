/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHistError.rdl,v 1.2 2001/05/14 22:54:20 verkerke Exp $
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
  
  // -----------------------------------------------------------
  // Define a 1-dim RooAbsFunc of mu that evaluates the sum:
  //
  //  Q(n|mu) = Sum_{k=0}^{n} P(n|mu)
  //
  // where P(n|mu) = exp(-mu) mu**n / n! is the Poisson PDF.
  // -----------------------------------------------------------
  class PoissonSum : public RooAbsFunc {
  public:
    inline PoissonSum(Int_t n) : RooAbsFunc(1), _n(n) { }
    inline Double_t operator()(const Double_t xvec[]) const {
      Double_t mu(xvec[0]),result(1),muton(1),factorial(1);
      for(Int_t k= 1; k <= _n; k++) {
	muton*= mu;
	factorial*= k;
	result+= muton/factorial;
      }
      return exp(-mu)*result;
    };
    inline Double_t getMinLimit(UInt_t index) const { return 0; }
    inline Double_t getMaxLimit(UInt_t index) const { return RooNumber::infinity; }
  private:
    Int_t _n;
  };

private:
  RooHistError();
  ClassDef(RooHistError,1) // Utility class for calculating histogram errors
};

#endif
