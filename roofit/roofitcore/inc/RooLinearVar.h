/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinearVar.h,v 1.19 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_LINEAR_VAR
#define ROO_LINEAR_VAR

#include <cmath>
#include <cfloat>
#include <string>
#include <list>
#include "RooAbsRealLValue.h"
#include "RooRealProxy.h"
#include "RooFormula.h"
#include "RooLinTransBinning.h"

class RooArgSet ;

class RooLinearVar : public RooAbsRealLValue {
public:
  // Constructors, assignment etc.
  RooLinearVar() {} ;
  RooLinearVar(const char *name, const char *title, RooAbsRealLValue& variable, const RooAbsReal& slope, const RooAbsReal& offset, const char *unit= "") ;
  RooLinearVar(const RooLinearVar& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooLinearVar(*this,newname); }
  ~RooLinearVar() override ;

  // Parameter value and error accessors
  void setVal(double value) override ;

  // Jacobian and limits
  bool hasBinning(const char* name) const override ;
  const RooAbsBinning& getBinning(const char* name=nullptr, bool verbose=true, bool createOnTheFly=false) const override ;
  RooAbsBinning& getBinning(const char* name=nullptr, bool verbose=true, bool createOnTheFly=false) override  ;
  std::list<std::string> getBinningNames() const override;

  double jacobian() const override ;
  bool isJacobianOK(const RooArgSet& depList) const override ;

  // I/O streaming interface (machine readable)
  bool readFromStream(std::istream& is, bool compact, bool verbose=false) override ;
  void writeToStream(std::ostream& os, bool compact) const override ;

  // Printing interface (human readable)

  using RooAbsRealLValue::operator= ;
  using RooAbsRealLValue::setVal ;

protected:

  double evaluate() const override ;

  mutable RooLinTransBinning _binning ;
  RooLinkedList _altBinning ; ///<!
  RooRealProxy _var ;         ///< Input observable
  RooRealProxy _slope ;       ///< Slope of transformation
  RooRealProxy _offset ;      ///< Offset of transformation

  ClassDefOverride(RooLinearVar,1) // Lvalue linear transformation function
};

#endif
