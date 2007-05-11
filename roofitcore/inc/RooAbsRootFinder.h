/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRootFinder.rdl,v 1.5 2005/02/25 14:22:52 wverkerke Exp $
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
#ifndef ROO_ABS_ROOT_FINDER
#define ROO_ABS_ROOT_FINDER

#include "Rtypes.h"

class RooAbsFunc;

class RooAbsRootFinder {
public:
  RooAbsRootFinder(const RooAbsFunc& function);
  inline virtual ~RooAbsRootFinder() { }

  virtual Bool_t findRoot(Double_t &result, Double_t xlo, Double_t xhi, Double_t value= 0) const = 0;

protected:
  const RooAbsFunc *_function;
  Bool_t _valid;

  ClassDef(RooAbsRootFinder,0) // Abstract interface for 1-dim real-valued function root finders
};

#endif
