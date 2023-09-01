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

#include <Rtypes.h>

class RooAbsFunc;

class RooBrentRootFinder {
public:
  RooBrentRootFinder(const RooAbsFunc& function);
  virtual ~RooBrentRootFinder() = default;

  bool findRoot(double &result, double xlo, double xhi, double value= 0) const;

  /// Set convergence tolerance parameter
  void setTol(double tol) {
    _tol = tol ;
  }

protected:
  static constexpr int MaxIterations = 512;

  const RooAbsFunc *_function; ///< Pointer to input function
  bool _valid;               ///< True if current state is valid

  double _tol ;

  ClassDef(RooBrentRootFinder,0)
};

#endif
