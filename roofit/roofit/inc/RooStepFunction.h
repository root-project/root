

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

#include "TArrayD.h"
#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooStepFunction : public RooAbsReal {
 public:

  RooStepFunction() ;
  RooStepFunction(const char *name, const char *title,
        RooAbsReal& x, const RooArgList& coefList, const RooArgList& limits, bool interpolate=false) ;

  RooStepFunction(const RooStepFunction& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooStepFunction(*this, newname); }

  const RooArgList& coefficients() { return _coefList; }
  const RooArgList& boundaries() { return _boundaryList; }

 protected:

  double evaluate() const override;

 private:

  RooRealProxy _x;
  RooListProxy _coefList ;
  RooListProxy _boundaryList ;
  bool       _interpolate ;
  TIterator* _coefIter ;  //! do not persist
  TIterator* _boundIter ;  //! do not persist

  ClassDefOverride(RooStepFunction,1) //  Step Function
};

#endif
