/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_ABS_FUNC
#define ROO_ABS_FUNC

#include "Rtypes.h"

class RooAbsFunc {
public:
  inline RooAbsFunc(UInt_t dimension) : _dimension(dimension), _valid(kTRUE) { }
  inline virtual ~RooAbsFunc() { }
  inline UInt_t getDimension() const { return _dimension; }
  inline Bool_t isValid() const { return _valid; }

  virtual Double_t operator()(const Double_t xvector[]) const = 0;
  virtual Double_t getMinLimit(UInt_t dimension) const = 0;
  virtual Double_t getMaxLimit(UInt_t dimension) const = 0;

protected:
  UInt_t _dimension;
  Bool_t _valid;
  ClassDef(RooAbsFunc,0) // Abstract real-valued function interface
};

#endif

