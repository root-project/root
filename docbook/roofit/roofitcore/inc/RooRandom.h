/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRandom.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_RANDOM
#define ROO_RANDOM

#include "Rtypes.h"
#include "TRandom.h"

class RooQuasiRandomGenerator;

class RooRandom {
public:

  virtual ~RooRandom() {} ;

  static TRandom *randomGenerator();
  static Double_t uniform(TRandom *generator= randomGenerator());
  static void uniform(UInt_t dimension, Double_t vector[], TRandom *generator= randomGenerator());
  static UInt_t integer(UInt_t max, TRandom *generator= randomGenerator());
  static Double_t gaussian(TRandom *generator= randomGenerator());

  static RooQuasiRandomGenerator *quasiGenerator();
  static Bool_t quasi(UInt_t dimension, Double_t vector[],
		      RooQuasiRandomGenerator *generator= quasiGenerator());

private:
  RooRandom();
  ClassDef(RooRandom,0) // Random number generator interface
};

#endif
