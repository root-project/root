/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRootFinder.h,v 1.6 2007/05/11 09:11:30 verkerke Exp $
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

  virtual bool findRoot(Double_t &result, Double_t xlo, Double_t xhi, Double_t value= 0) const = 0;

protected:
  const RooAbsFunc *_function; ///< Pointer to input function
  bool _valid;               ///< True if current state is valid

  ClassDef(RooAbsRootFinder,0) // Abstract interface for 1-dim real-valued function root finders
};

#endif
