/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooExtendPdf.rdl,v 1.10 2005/07/12 11:29:37 wverkerke Exp $
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
#ifndef ROO_EXTEND_PDF
#define ROO_EXTEND_PDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooExtendPdf : public RooAbsPdf {
public:

  RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf, 
	       const RooAbsReal& norm, const char* rangeName=0) ;
  RooExtendPdf(const RooExtendPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooExtendPdf(*this,newname) ; }
  virtual ~RooExtendPdf() ;

  Double_t evaluate() const { return _pdf ; }

  Bool_t forceAnalyticalInt(const RooAbsArg& /*dep*/) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const {
    return ((RooAbsPdf&)_pdf.arg()).getAnalyticalIntegralWN(allVars, analVars, normSet, rangeName) ;
  }
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const {
    return ((RooAbsPdf&)_pdf.arg()).analyticalIntegralWN(code, normSet, rangeName) ;
  }
  
  virtual Bool_t selfNormalized() const { return kTRUE ; }
  virtual ExtendMode extendMode() const { return CanBeExtended ; }
  virtual Double_t expectedEvents(const RooArgSet* nset) const ;
  virtual Double_t expectedEvents(const RooArgSet& nset) const { return expectedEvents(&nset) ; }

protected:

  RooRealProxy _pdf ;        // PDF used for fractional correction factor
  RooRealProxy _n ;          // Number of expected events
  const TNamed* _rangeName ; // Name of subset range


  ClassDef(RooExtendPdf,0) // Flat PDF introducing an extended likelihood term
};

#endif
