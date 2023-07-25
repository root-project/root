

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id$
 * Authors:                                                                  *
 *    Tristan du Pree, Nikhef, Amsterdam, tdupree@nikhef.nl                  *
 *                                                                           *
 * Copyright (c) 2000-2005, Stanford University. All rights reserved.        *
 *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_STEP_FUNCTION
#define ROO_STEP_FUNCTION

#include <RooAbsReal.h>
#include <RooListProxy.h>
#include <RooRealProxy.h>

class RooArgList ;

class RooStepFunction : public RooAbsReal {
 public:

  RooStepFunction() {}
  RooStepFunction(const char *name, const char *title,
        RooAbsReal& x, const RooArgList& coefList, const RooArgList& limits, bool interpolate=false) ;

  RooStepFunction(const RooStepFunction& other, const char *name = nullptr);
  TObject* clone(const char* newname) const override { return new RooStepFunction(*this, newname); }

  const RooArgList& coefficients() { return _coefList; }
  const RooArgList& boundaries() { return _boundaryList; }

  std::list<double>* plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const override ;

 protected:

  double evaluate() const override;

 private:

  RooRealProxy _x;
  RooListProxy _coefList ;
  RooListProxy _boundaryList ;
  bool       _interpolate = false;

  ClassDefOverride(RooStepFunction,1) //  Step Function
};

#endif
