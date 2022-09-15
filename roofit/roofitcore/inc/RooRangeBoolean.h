/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_RANGE_BOOLEAN
#define ROO_RANGE_BOOLEAN

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "TString.h"

class RooRealVar;
class RooArgList ;
#include <list>

class RooRangeBoolean : public RooAbsReal {
public:

  RooRangeBoolean() ;
  RooRangeBoolean(const char* name, const char* title, RooAbsRealLValue& x, const char* rangeName) ;
  RooRangeBoolean(const RooRangeBoolean& other, const char *name = nullptr);
  TObject* clone(const char* newname) const override { return new RooRangeBoolean(*this, newname); }
  ~RooRangeBoolean() override ;


  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;

protected:

  RooRealProxy _x;
  TString _rangeName ;

  double evaluate() const override;

  ClassDefOverride(RooRangeBoolean,1) // Polynomial function
};

#endif
