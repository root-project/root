/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvCoefVar.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
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

#include "Riostream.h"
#include <math.h>
#include <float.h>

#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
class RooAbsAnaConfPdf ;

class RooConvCoefVar : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooConvCoefVar() { }
  RooConvCoefVar(const char *name, const char *title, const RooAbsAnaConvPdf& input, Int_t coefIdx, const RooArgSet* varList=0) ;
  RooConvCoefVar(const RooConvCoefVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooConvCoefVar(*this,newname); }
  virtual ~RooConvCoefVar() {} ;

  virtual Double_t getVal(const RooArgSet* nset=0) const ;

  virtual Double_t evaluate() const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  virtual Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:

  RooSetProxy  _varSet ; 
  RooRealProxy _convPdf ; // ConvolutedPDfs implementing our coefficient
  Int_t    _coefIdx  ;    // Index code of the coefficient

  ClassDef(RooConvCoefVar,1) // Auxiliary class representing the coefficient of a RooAbsAnaConvPdf
};

#endif
