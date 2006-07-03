/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 *    File: $Id: RooParametricStepFunction.rdl,v 1.3 2005/02/25 14:25:06 wverkerke Exp $
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

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooListProxy.hh"

class RooRealVar;
class RooArgList ;

class RooParametricStepFunction : public RooAbsPdf {
public:

  RooParametricStepFunction(const char *name, const char *title,
		RooAbsReal& x, const RooArgList& coefList, TArrayD& limits, Int_t nBins=1) ;

  RooParametricStepFunction(const RooParametricStepFunction& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooParametricStepFunction(*this, newname); }
  virtual ~RooParametricStepFunction() ;

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;
  Int_t getnBins();
  Double_t* getLimits();

protected:

  Double_t lastBinValue() const ;

  RooRealProxy _x;
  RooListProxy _coefList ;
  TArrayD _limits;
  Int_t _nBins ;
  TIterator* _coefIter ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooParametricStepFunction,1) // Parametric Step Function Pdf
};

#endif
