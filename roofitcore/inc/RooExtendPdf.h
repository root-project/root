/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EXTEND_PDF
#define ROO_EXTEND_PDF

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooExtendPdf : public RooAbsPdf {
public:

  RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf, 
	       const RooAbsReal& norm) ;
  RooExtendPdf(const char *name, const char *title, const RooAbsPdf& pdf, 
	       const RooAbsReal& norm, const RooArgList& depList, const RooArgList& cutDepList) ;
  RooExtendPdf(const RooExtendPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooExtendPdf(*this,newname) ; }
  virtual ~RooExtendPdf() ;

  Double_t evaluate() const { return _pdf ; }

  Bool_t forceAnalyticalInt(const RooAbsArg& dep) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars, const RooArgSet* normSet) const {
    return ((RooAbsPdf&)_pdf.arg()).getAnalyticalIntegralWN(allVars, analVars, normSet) ;
  }
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const {
    return ((RooAbsPdf&)_pdf.arg()).analyticalIntegralWN(code, normSet) ;
  }
  
  virtual Bool_t selfNormalized() const { return kTRUE ; }
  virtual ExtendMode extendMode() const { return CanBeExtended ; }
  virtual Double_t expectedEvents() const ;

protected:

  virtual void getParametersHook(const RooArgSet* nset, RooArgSet* list) const ;
  virtual void getDependentsHook(const RooArgSet* nset, RooArgSet* list) const ;

  Bool_t _useFrac ; 
  mutable RooArgSet* _lastFracSet ;          // Normalization set for which last fracIntegral was made
  mutable RooAbsReal* _fracIntegral ;        // Fractional normalization integral
  mutable RooArgSet* _integralCompSet ;      // Owning set of components of fracIntegral and its servers
  RooRealProxy _pdf ;        // PDF used for fractional correction factor
  RooRealProxy _n ;          // Number of expected events
  RooSetProxy  _cutDepSet ;  // Set of dependents with modified fit range
  RooSetProxy  _origDepSet ; // Set of original dependents 

  void syncFracIntegral() const ;

  ClassDef(RooExtendPdf,0) // Flat PDF introducing an extended likelihood term
};

#endif
