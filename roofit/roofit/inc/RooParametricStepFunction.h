/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooParametricStepFunction.h,v 1.5 2007/05/11 09:13:07 verkerke Exp $
 * Authors:                                                                  *
 *    Aaron Roodman, Stanford Linear Accelerator Center, Stanford University *
 *                                                                           *
 * Copyright (c) 2000-2005, Stanford University. All rights reserved.        *
 *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_PARAMETRIC_STEP_FUNCTION
#define ROO_PARAMETRIC_STEP_FUNCTION

#include "TArrayD.h"
#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooParametricStepFunction : public RooAbsPdf {
public:

   RooParametricStepFunction() : _nBins(0), _coefIter(0) {}

  RooParametricStepFunction(const char *name, const char *title,
      RooAbsReal& x, const RooArgList& coefList, TArrayD& limits, Int_t nBins=1) ;

  RooParametricStepFunction(const RooParametricStepFunction& other, const char* name = 0);
  TObject* clone(const char* newname) const override { return new RooParametricStepFunction(*this, newname); }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;
  Int_t getnBins();
  Double_t* getLimits();

protected:

  Double_t lastBinValue() const ;

  RooRealProxy _x;
  RooListProxy _coefList ;
  TArrayD _limits;
  Int_t _nBins ;
  TIterator* _coefIter ;  //! do not persist

  Double_t evaluate() const override;

  ClassDefOverride(RooParametricStepFunction,1) // Parametric Step Function Pdf
};

#endif
