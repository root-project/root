/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHistPdf.rdl,v 1.8 2005/02/25 14:22:57 wverkerke Exp $
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
#ifndef ROO_HIST_PDF
#define ROO_HIST_PDF

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"
#include "RooAICRegistry.h"

class RooRealVar;
class RooAbsReal;
class RooDataHist ;

class RooHistPdf : public RooAbsPdf {
public:
  RooHistPdf(const char *name, const char *title, const RooArgSet& vars, const RooDataHist& dhist, Int_t intOrder=0);
  RooHistPdf(const RooHistPdf& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooHistPdf(*this,newname); }
  inline virtual ~RooHistPdf() { }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:

  Double_t evaluate() const;

  RooSetProxy  _depList ;   // List of dependents defining dimensions of histogram
  RooDataHist* _dataHist ;  // Unowned pointer to underlying histogram
  mutable RooAICRegistry _codeReg ; // Auxiliary class keeping tracking of analytical integration code
  Int_t        _intOrder ; // Interpolation order

  ClassDef(RooHistPdf,0) // Histogram based PDF
};

#endif
