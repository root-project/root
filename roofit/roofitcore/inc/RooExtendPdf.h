/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooExtendPdf.h,v 1.12 2007/07/16 21:04:28 wouter Exp $
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

  RooExtendPdf() ;
  RooExtendPdf(const char *name, const char *title, RooAbsPdf& pdf,
          RooAbsReal& norm, const char* rangeName=0) ;
  RooExtendPdf(const RooExtendPdf& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooExtendPdf(*this,newname) ; }
  ~RooExtendPdf() override ;

  double evaluate() const override { return _pdf ; }

  bool forceAnalyticalInt(const RooAbsArg& /*dep*/) const override { return true ; }
  /// Forward determination of analytical integration capabilities to input p.d.f
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet, const char* rangeName=0) const override {
    return _pdf->getAnalyticalIntegralWN(allVars, analVars, normSet, rangeName) ;
  }
  /// Forward calculation of analytical integrals to input p.d.f
  double analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const override {
    return _pdf->analyticalIntegralWN(code, normSet, rangeName) ;
  }

  bool selfNormalized() const override { return true ; }
  ExtendMode extendMode() const override { return CanBeExtended ; }
  double expectedEvents(const RooArgSet* nset) const override ;

protected:

  RooTemplateProxy<RooAbsPdf>  _pdf; ///< Input p.d.f
  RooTemplateProxy<RooAbsReal> _n;   ///< Number of expected events
  const TNamed* _rangeName ;         ///< Name of subset range


  ClassDefOverride(RooExtendPdf,2) // Wrapper p.d.f adding an extended likelihood term to an existing p.d.f
};

#endif
