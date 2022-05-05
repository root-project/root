/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooBrentRootFinder.h,v 1.6 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_BRENT_ROOT_FINDER
#define ROO_BRENT_ROOT_FINDER

#include "RooAbsRootFinder.h"

class RooBrentRootFinder : public RooAbsRootFinder {
public:
  RooBrentRootFinder(const RooAbsFunc& function);
  inline ~RooBrentRootFinder() override { }

  bool findRoot(Double_t &result, Double_t xlo, Double_t xhi, Double_t value= 0) const override;

  /// Set convergence tolerance parameter
  void setTol(Double_t tol) {
    _tol = tol ;
  }

protected:
  enum { MaxIterations = 512 };

  Double_t _tol ;

  ClassDefOverride(RooBrentRootFinder,0) // Abstract interface for 1-dim real-valued function root finders
};

#endif
