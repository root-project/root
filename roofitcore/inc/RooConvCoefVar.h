/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvCoefVar.rdl,v 1.3 2002/09/05 04:33:19 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_CONV_COEF_VAR
#define ROO_CONV_COEF_VAR

#include <iostream.h>
#include <math.h>
#include <float.h>
#include "TString.h"

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooConvCoefVar : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooConvCoefVar() { }
  RooConvCoefVar(const char *name, const char *title, const RooConvolutedPdf& input, Int_t coefIdx, const RooArgSet* varList=0) ;
  RooConvCoefVar(const RooConvCoefVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooConvCoefVar(*this,newname); }
  virtual ~RooConvCoefVar() {} ;

  virtual Double_t getVal(const RooArgSet* nset=0) const { return evaluate() ; }

  virtual Double_t evaluate() const ;
  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const ;
  virtual Double_t analyticalIntegral(Int_t code) const ;

protected:

  RooSetProxy  _varSet ; 
  RooRealProxy _convPdf ; // ConvolutedPDfs implementing our coefficient
  Int_t    _coefIdx  ;    // Index code of the coefficient

  ClassDef(RooConvCoefVar,1) // Auxiliary class representing the coefficient of a RooConvolutedPdf
};

#endif
