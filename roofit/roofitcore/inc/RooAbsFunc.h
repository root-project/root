/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsFunc.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_FUNC
#define ROO_ABS_FUNC

#include "Rtypes.h"
#include "RooSpan.h"

#include <list>
#include <vector>

class RooAbsRealLValue ;

class RooAbsFunc {
public:
  inline RooAbsFunc(UInt_t dimension) : _ncall(0), _dimension(dimension), _valid(kTRUE) { }
  inline RooAbsFunc(const RooAbsFunc& other) : _ncall(0), _dimension(other._dimension), _valid(kTRUE) { }

  inline virtual ~RooAbsFunc() { }
  inline UInt_t getDimension() const {
    // Dimension of function
    return _dimension;
  }
  inline Bool_t isValid() const {
    // Is function in valid state
    return _valid;
  }

  virtual Double_t operator()(const Double_t xvector[]) const = 0;
  virtual Double_t getMinLimit(UInt_t dimension) const = 0;
  virtual Double_t getMaxLimit(UInt_t dimension) const = 0;

  /// Return number of function calls since last reset
  Int_t numCall() const {
    return _ncall ;
  }

  /// Reset function call counter
  void resetNumCall() const {
    _ncall = 0 ;
  }

  virtual void saveXVec() const {
    // Interface to save current values of observables (if supported by binding implementation)
  } ;
  virtual void restoreXVec() const {
    // Interface to restore observables to saved values (if supported
    // by binding implementation)
  } ;

  /// Name of function binding
  virtual const char* getName() const {
    return "(unnamed)" ;
  }

  virtual std::list<Double_t>* binBoundaries(Int_t) const { return nullptr; }

  /// Interface for returning an optional hint for initial sampling points when constructing a curve
  /// projected on observable.
  virtual std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const {
    return nullptr;
  }

protected:
  mutable Int_t _ncall ;  ///< Function call counter
  UInt_t _dimension;      ///< Number of observables
  Bool_t _valid;          ///< Is binding in valid state?
   ClassDef(RooAbsFunc,0) ///< Abstract real-valued function interface
};

#endif

