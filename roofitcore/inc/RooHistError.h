/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.rdl,v 1.3 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#ifndef ROO_HIST_ERROR
#define ROO_HIST_ERROR

#include "Rtypes.h"
#include "TMath.h"

class RooHistError {
public:
  static const RooHistError &instance();
protected:
  // Declare an abstract base class for confidence-level integrals
  struct CLIntegral {
    inline CLIntegral() { }
    virtual Double_t operator()(Double_t x) const = 0;
  };
  // Implementation for Poisson errors
  struct PoissonIntegral : CLIntegral {
    inline PoissonIntegral(Double_t nval) : np1(nval+1) { }
    inline Double_t operator()(Double_t x) const {
      return TMath::Gamma(np1,x);
    }
    Double_t np1;
  };
private:
  RooHistError();
  ClassDef(RooHistError,1) // a utility class for calculating histogram errors
};

#endif
