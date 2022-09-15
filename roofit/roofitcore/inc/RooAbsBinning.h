/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsBinning.h,v 1.13 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_BINNING
#define ROO_ABS_BINNING

#include "Rtypes.h"
#include "RooPrintable.h"
#include "TNamed.h"
class RooAbsRealLValue ;
class RooAbsReal ;

class RooAbsBinning : public TNamed, public RooPrintable {
public:

  RooAbsBinning(const char* name=nullptr) : TNamed{name, name} {}
  RooAbsBinning(const RooAbsBinning& other, const char* name=nullptr) : TNamed(name,name), RooPrintable(other) {
    // Copy constructor
  }
  TObject* Clone(const char* newname=nullptr) const override { return clone(newname) ; }
  virtual RooAbsBinning* clone(const char* name=nullptr) const = 0 ;

  /// Return number of bins.
  Int_t numBins() const {
    return numBoundaries()-1 ;
  }
  virtual Int_t numBoundaries() const = 0 ;

  /// Compute the bin indices for multiple values of `x`.
  ///
  /// For each element in the input, the corresponding output element will be
  /// increased by `coef * binIdx`. This is useful for aggregating
  /// multi-dimensional bin indices inplace.
  ///
  /// param[in] x The read-only input array of values of `x`.
  /// param[out] bins The output array. Note that the initial values don't get
  ///                 replaced! The result is added to the array elements.
  /// param[in] n The size of the input and output arrays.
  /// param[in] coef The multiplication factor that is applied to all calculated bin indices.
  virtual void binNumbers(double const * x, int * bins, std::size_t n, int coef=1) const = 0 ;

  /// Returns the bin number corresponding to the value `x`.
  ///
  /// \note This `inline` function is implemented by calling the vectorized
  ///       function `RooAbsBinning::binNumbers()`. If you want to calculate
  ///       the bin indices for multiple values, use that one for better
  ///       performance.
  inline int binNumber(double x) const {
    int out = 0.0;
    binNumbers(&x, &out, 1);
    return out;
  }

  virtual double binCenter(Int_t bin) const = 0 ;
  virtual double binWidth(Int_t bin) const = 0 ;
  virtual double binLow(Int_t bin) const = 0 ;
  virtual double binHigh(Int_t bin) const = 0 ;
  virtual bool isUniform() const { return false ; }

  virtual void setRange(double xlo, double xhi) = 0 ;
  /// Change lower bound to xlo.
  virtual void setMin(double xlo) {
    setRange(xlo,highBound()) ;
  }
  /// Change upper bound to xhi.
  virtual void setMax(double xhi) {
    setRange(lowBound(),xhi) ;
  }

  virtual double lowBound() const = 0 ;
  virtual double highBound() const = 0 ;
  virtual double averageBinWidth() const = 0 ;


  virtual double* array() const = 0 ;

  inline void Print(Option_t *options= nullptr) const override {
    // Printing interface
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printArgs(std::ostream& os) const override ;
  void printValue(std::ostream& os) const override ;

  /// Interface function. If true, min/max of binning is parameterized by external RooAbsReals.
  /// Default to `false`, unless overridden by a sub class.
  virtual bool isParameterized() const {
    return false ;
  }
  /// Return pointer to RooAbsReal parameterized lower bound, if any.
  virtual RooAbsReal* lowBoundFunc() const {
    return nullptr ;
  }
  /// Return pointer to RooAbsReal parameterized upper bound, if any.
  virtual RooAbsReal* highBoundFunc() const {
    return nullptr ;
  }
  /// If true (default), the range definition can be shared across clones of a RooRealVar.
  virtual bool isShareable() const {
    return true ;
  }
  /// Hook interface function to execute code upon insertion into a RooAbsRealLValue.
  virtual void insertHook(RooAbsRealLValue&) const {  }
  /// Hook interface function to execute code upon removal from a RooAbsRealLValue.
  virtual void removeHook(RooAbsRealLValue&) const {  }


protected:

  ClassDefOverride(RooAbsBinning,2) // Abstract base class for binning specification
};

#endif
