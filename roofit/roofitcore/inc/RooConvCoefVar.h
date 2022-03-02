/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvCoefVar.h,v 1.14 2007/05/14 17:56:18 brun Exp $
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
#ifndef ROO_CONV_COEF_VAR
#define ROO_CONV_COEF_VAR

#include <cmath>
#include <float.h>

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
class RooAbsAnaConvPdf ;

class RooConvCoefVar : public RooAbsReal {
public:
  // Constructors, assignment etc.
  /// Default constructor
  inline RooConvCoefVar() {
  }
  RooConvCoefVar(const char *name, const char *title, const RooAbsAnaConvPdf& input, Int_t coefIdx, const RooArgSet* varList=0) ;
  RooConvCoefVar(const RooConvCoefVar& other, const char* name=0);
  TObject* clone(const char* newname) const override { return new RooConvCoefVar(*this,newname); }
  /// Destructor
  ~RooConvCoefVar() override {
  } ;

  Double_t getValV(const RooArgSet* nset=0) const override ;

  Double_t evaluate() const override ;
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const override ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const override ;

protected:

  RooSetProxy  _varSet ;  ///< Not used anymore?
  RooRealProxy _convPdf ; ///< RooAbsAnaConv object implementing our coefficient
  Int_t    _coefIdx  ;    ///< Index code of the coefficient

  ClassDefOverride(RooConvCoefVar,1) // Auxiliary class representing the coefficient of a RooAbsAnaConvPdf as a RooAbsReal
};

#endif
