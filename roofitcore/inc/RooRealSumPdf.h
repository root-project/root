/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealSumPdf.rdl,v 1.9 2005/04/18 21:44:50 wverkerke Exp $
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
#ifndef ROO_REAL_SUM_PDF
#define ROO_REAL_SUM_PDF

#include "RooAbsPdf.h"
#include "RooListProxy.h"
#include "RooAICRegistry.h"

class RooRealSumPdf : public RooAbsPdf {
public:

  RooRealSumPdf(const char *name, const char *title);
  RooRealSumPdf(const char *name, const char *title,
		   RooAbsReal& func1, RooAbsReal& func2, RooAbsReal& coef1) ;
  RooRealSumPdf(const char *name, const char *title, const RooArgList& funcList, const RooArgList& coefList) ;
  RooRealSumPdf(const RooRealSumPdf& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooRealSumPdf(*this,newname) ; }
  virtual ~RooRealSumPdf() ;

  Double_t evaluate() const ;
  virtual Bool_t checkObservables(const RooArgSet* nset) const ;	

  virtual Bool_t forceAnalyticalInt(const RooAbsArg&) const { return kTRUE ; }
  Int_t getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& numVars, const RooArgSet* normSet, const char* rangeName=0) const ;
  Double_t analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName=0) const ;

  const RooArgList& funcList() const { return _funcList ; }
  const RooArgList& coefList() const { return _coefList ; }


protected:
  
  mutable RooAICRegistry _codeReg ;  // Registry of component analytical integration codes

  void syncFuncIntList(const RooArgSet* intSet) const ;
  void syncFuncNormList(const RooArgSet* normSet) const ;
  mutable RooArgSet* _lastFuncIntSet ;
  mutable RooArgSet* _lastFuncNormSet ;
  mutable RooArgList* _funcIntList ;  //!
  mutable RooArgList* _funcNormList ; //!

  Bool_t _haveLastCoef ;

  RooListProxy _funcList ;   //  List of component FUNCs
  RooListProxy _coefList ;  //  List of coefficients
  TIterator* _funcIter ;     //! Iterator over FUNC list
  TIterator* _coefIter ;    //! Iterator over coefficient list
  
private:

  ClassDef(RooRealSumPdf,1) // PDF representing a sum of real functions
};

#endif
