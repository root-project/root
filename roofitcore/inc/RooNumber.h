/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooIntegrator1D.rdl,v 1.6 2001/08/02 23:54:24 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Aug-2001 DK Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_NUMBER
#define ROO_NUMBER

#include "Rtypes.h"

class RooNumber {
public:
  static Double_t infinity;
  static inline Int_t isInfinite(Double_t x) {
    return (x >= +infinity) ? +1 : ((x <= -infinity) ? -1 : 0);
  }

  ClassDef(RooNumber,0) // wrapper class for portable numerics
};

#endif
