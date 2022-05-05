/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConstVar.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_CONST_VAR
#define ROO_CONST_VAR

#include "RooAbsReal.h"

class RooArgSet ;
namespace RooBatchCompute {
  struct RunContext;
}

class RooConstVar final : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooConstVar() { }
  RooConstVar(const char *name, const char *title, Double_t value);
  RooConstVar(const RooConstVar& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooConstVar(*this,newname); }
  ~RooConstVar() override = default;

  /// Return (constant) value.
  Double_t getValV(const RooArgSet*) const override {
    return _value;
  }

  RooSpan<const double> getValues(RooBatchCompute::RunContext& evalData, const RooArgSet*) const override;

  void writeToStream(std::ostream& os, bool compact) const override ;

  /// Returns false, as the value of the constant doesn't depend on other objects.
  bool isDerived() const override {
    return false;
  }

  /// Change the value of this constant.
  /// On purpose, this is not `setVal`, as this could be confused with the `setVal`
  /// that is available for variables. Constants, however, should remain mostly constant.
  /// This function is e.g. useful when reading the constant from a file.
  void changeVal(double value) {
    _value = value;
  }

protected:

  Double_t evaluate() const override {
    return _value;
  }

  ClassDefOverride(RooConstVar,2) // Constant RooAbsReal value object
};

#endif
