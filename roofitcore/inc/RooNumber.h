/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumber.rdl,v 1.6 2005/06/20 18:33:37 wverkerke Exp $
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
#ifndef ROO_NUMBER
#define ROO_NUMBER

#include "Rtypes.h"

class RooNumber {
public:

  virtual ~RooNumber() {} ;

  static Double_t infinity;
  static inline Int_t isInfinite(Double_t x) {
    return (x >= +infinity) ? +1 : ((x <= -infinity) ? -1 : 0);
  }

  ClassDef(RooNumber,0) // wrapper class for portable numerics
};

#endif
